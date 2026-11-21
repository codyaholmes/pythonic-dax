"""Aggregations computed by Polars over whole columns (Phase 9).

The formula engine evaluates an iterator such as ``SUMX(Sales, Sales[Qty] *
Sales[Price])`` one row at a time, which is exact but slow. This module handles the
common shapes in one Polars pass, following the same rules as the scalar operators in
``_scalar.py``:

- row expressions built from numeric columns of the iterated table, ``RELATED`` numeric
  columns, number literals, unary minus and ``+ - *``;
- the aggregations SUM, AVERAGE, MIN, MAX, the COUNT family, COUNTROWS, and SUMX,
  AVERAGEX, MINX, MAXX and COUNTX over such row expressions.

Each aggregation is a list of Polars expressions plus a ``finish`` function that turns
their results into the DAX value, so the same code serves one filter context or many
groups at once (see ``_batch.py``). Anything outside these shapes, or close to an edge
where exactness isn't certain (whole-number overflow, mixed types with blanks), raises
:class:`NoVector`, and the caller uses the row-by-row path instead.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import polars as pl

from . import _scalar as sc
from .errors import NotSupportedError
from .expr import BinaryOp, Call, Literal_, UnaryOp, _Omitted, is_column
from .objects import Cardinality
from .types import DaxType

if TYPE_CHECKING:
    from ._evaluate import Evaluator
    from .objects import Column, Relationship, Table

Value = Any

_INT_LIMIT = 2**62  # whole-number intermediate results stay well inside 64 bits
_EXACT_FLOAT = 2**53  # whole numbers below this convert to doubles exactly
_CURRENCY_LIMIT = 10**14  # fixed decimals stay well inside Power BI's 64-bit storage

_KINDS = {
    DaxType.INTEGER: "int",
    DaxType.DECIMAL: "dec",
    DaxType.CURRENCY: "cur",
}


ENABLED = True
"""Set to False to evaluate everything row by row and group by group, as before Phase 9
(tests compare the two)."""


class NoVector(Exception):
    """The expression isn't one of the shapes this module computes exactly."""


# --- row expressions -----------------------------------------------------------------

_SCALE = 10_000  # a Fixed Decimal Number is held as a whole number of 1/10,000ths
_ROW_CACHE_SIZE = 4
"""Row expressions whose values (over a whole table) a model keeps between queries."""


@dataclass
class _Vec:
    expr: pl.Expr
    kind: str  # "int", "dec" or "cur" (held as Int64 ten-thousandths)
    nullable: bool
    bound: float  # the largest absolute value it can take, in DAX units


class RowExpression:
    """A numeric expression evaluated for every row of a model table.

    ``values(mask)`` gives the expression's value for the rows of
    ``Storage.ext(table)`` selected by ``mask``, and the kind of number it is. Values
    of kind "cur" are whole numbers of ten-thousandths (see :func:`to_currency`).
    """

    def __init__(self, ev: Evaluator, table: Table, node: Any) -> None:
        self.ev = ev
        self.table = table
        self.node = node
        self._related: dict[int, tuple[str, Column]] = {}
        self._columns: dict[int, Column] = {}
        self._collect(node)

    def _collect(self, node: Any) -> None:
        if is_column(node):
            if node.table is not self.table or node is self.ev.computing:
                raise NoVector
            if node.name not in self.ev.storage.ext(self.table).columns:
                raise NoVector
            _numeric(node)
            self._columns[id(node)] = node
        elif isinstance(node, Call) and node.name == "RELATED":
            column: Any = node.args[0]
            if not is_column(column) or column.table is self.table:
                raise NoVector
            _numeric(column)
            self._related.setdefault(id(column), (f"__related{len(self._related)}", column))
        elif isinstance(node, Literal_):
            v = node.value
            if isinstance(v, bool) or not isinstance(v, int | float | Decimal):
                raise NoVector
        elif isinstance(node, UnaryOp):
            if node.op != "-":
                raise NoVector
            self._collect(node.operand)
        elif isinstance(node, BinaryOp):
            if node.op not in ("+", "-", "*"):
                raise NoVector
            self._collect(node.left)
            self._collect(node.right)
        else:
            raise NoVector

    bound: float | None = None
    """After ``values``: the largest absolute value, in the units of the values."""

    def values(self, mask: pl.Series | None) -> tuple[pl.Series, str]:
        """The values for the rows in ``mask`` (all rows of ``ext`` when None)."""
        values, kind, bound = self._all()
        self.bound = bound
        return (values if mask is None else values.filter(mask)), kind

    def _all(self) -> tuple[pl.Series, str, float]:
        """The values for every row of ``ext``, cached on the model while the data and
        relationship lookups they come from are unchanged."""
        storage = self.ev.storage
        lookups = []
        for _, column in self._related.values():
            storage.check_text(column)
            lookups.append((column, related_index(self.ev, self.table, column.table)))
        state = (
            storage.ext(self.table),
            *(storage.ext(c.table) for c, _ in lookups),
            *(index for _, index in lookups),
        )
        cache = storage.cache.rows
        key = (id(self.node), id(self.table))
        hit = cache.get(key)
        if (
            hit is not None
            and len(hit[0]) == len(state)
            and all(x is y for x, y in zip(hit[0], state, strict=True))
        ):
            cache.move_to_end(key)
            return hit[1], hit[2], hit[4]
        ext = storage.ext(self.table)
        series = [ext[c.name] for c in self._columns.values()]
        series += [
            storage.ext(column.table)[column.name].gather(index).alias(name)
            for (name, column), (_, index) in zip(self._related.values(), lookups, strict=True)
        ]
        # A frame of no columns has no rows: keep a row index so literals broadcast.
        frame = pl.DataFrame([pl.int_range(ext.height, eager=True).alias("__row"), *series])
        vec = self._compile(self.node, frame)
        values = frame.select(vec.expr.alias("v"))["v"]
        bound = vec.bound * _SCALE if vec.kind == "cur" else vec.bound
        cache[key] = (state, values, vec.kind, self.node, bound)
        while len(cache) > _ROW_CACHE_SIZE:
            cache.popitem(last=False)
        return values, vec.kind, bound

    def _compile(self, node: Any, frame: pl.DataFrame) -> _Vec:
        if is_column(node):
            return _column_vec(node.name, node.data_type, frame)
        if isinstance(node, Call):  # RELATED
            name, column = self._related[id(node.args[0])]
            return _column_vec(name, column.data_type, frame)
        if isinstance(node, Literal_):
            v = node.value
            if isinstance(v, int):
                return _Vec(pl.lit(v, dtype=pl.Int64), "int", False, abs(v))
            return _Vec(pl.lit(float(v), dtype=pl.Float64), "dec", False, abs(float(v)))
        if isinstance(node, UnaryOp):
            inner = self._compile(node.operand, frame)
            return _Vec(-inner.expr, inner.kind, inner.nullable, inner.bound)
        a = self._compile(node.left, frame)
        b = self._compile(node.right, frame)
        if node.op == "*":
            return _multiply(a, b)
        return _add(node.op, a, b)


def to_currency(total: Any) -> Decimal:
    """A whole number of ten-thousandths as a Fixed Decimal Number."""
    return Decimal(int(total)).scaleb(-4)


def _numeric(column: Column) -> None:
    if column.data_type not in _KINDS:
        raise NoVector


def _column_vec(name: str, dax_type: DaxType, frame: pl.DataFrame) -> _Vec:
    series = frame[name]
    kind = _KINDS[dax_type]
    if kind == "dec" and series.is_nan().any():
        raise NoVector
    bound = _bound(series)
    if kind == "cur":
        if bound >= _CURRENCY_LIMIT:
            raise NoVector
        expr = pl.col(name).to_physical().cast(pl.Int64)
        return _Vec(expr, kind, series.has_nulls(), bound)
    return _check(_Vec(pl.col(name), kind, series.has_nulls(), bound))


def _check(vec: _Vec) -> _Vec:
    if vec.kind == "int" and vec.bound >= _INT_LIMIT:
        raise NoVector
    if vec.kind == "cur" and vec.bound >= _CURRENCY_LIMIT:
        raise NoVector
    return vec


def _as(vec: _Vec, kind: str) -> pl.Expr:
    if kind == vec.kind:
        return vec.expr
    if kind == "dec":
        if vec.kind == "cur":
            if vec.bound * _SCALE >= _EXACT_FLOAT:
                raise NoVector
            # Exact ten-thousandths over 10,000.0 is the double nearest the decimal,
            # as Python's float(Decimal) gives.
            return vec.expr.cast(pl.Float64) / float(_SCALE)
        if vec.bound >= _EXACT_FLOAT:
            raise NoVector
        return vec.expr.cast(pl.Float64)
    if kind == "cur" and vec.kind == "int":
        return vec.expr * _SCALE
    raise NoVector


def _multiply(a: _Vec, b: _Vec) -> _Vec:
    """``a * b``: BLANK if either side is BLANK (the rules of ``_scalar._multiply``)."""
    kinds = {a.kind, b.kind}
    bound = a.bound * b.bound
    nullable = a.nullable or b.nullable
    if kinds == {"int"}:
        return _check(_Vec(a.expr * b.expr, "int", nullable, bound))
    if kinds == {"cur"}:  # Currency times Currency is a double
        return _Vec(_as(a, "dec") * _as(b, "dec"), "dec", nullable, bound)
    if kinds == {"cur", "int"}:  # exact: a whole number times ten-thousandths
        if bound >= _CURRENCY_LIMIT:
            raise NoVector
        return _Vec(a.expr * b.expr, "cur", nullable, bound)
    if "cur" in kinds:  # Currency times a double rounds to four places: not here
        raise NoVector
    return _Vec(_as(a, "dec") * _as(b, "dec"), "dec", nullable, bound)


def _add(op: str, a: _Vec, b: _Vec) -> _Vec:
    """``a + b`` or ``a - b``: BLANK only when both sides are BLANK, else BLANK counts
    as 0 (the rules of ``_scalar.arithmetic``)."""
    kinds = {a.kind, b.kind}
    nullable = a.nullable and b.nullable
    if len(kinds) > 1 and (a.nullable or b.nullable):
        # With one side BLANK the result takes the other side's type, which changes
        # how later steps round. Leave those rows to the exact path.
        raise NoVector
    if kinds == {"int"}:
        kind = "int"
    elif "dec" in kinds:
        kind = "dec"
    else:
        kind = "cur"
    x, y = _as(a, kind), _as(b, kind)
    if a.nullable or b.nullable:
        both_blank = x.is_null() & y.is_null()
        x, y = x.fill_null(0), y.fill_null(0)
        result = pl.when(both_blank).then(None).otherwise(x + y if op == "+" else x - y)
    else:
        result = x + y if op == "+" else x - y
    return _check(_Vec(result, kind, nullable, a.bound + b.bound))


def related_index(ev: Evaluator, start: Table, target: Table) -> pl.Series:
    """For each row of ``start`` (in ``Storage.ext`` order), the row of ``target`` that
    RELATED reads: the blank row (all BLANK) or null when there is no match.

    Follows the same active relationships as ``Evaluator.related_path``, towards the
    one side only.
    """
    rels = related_relationships(ev, start, target)
    index: pl.Series | None = None
    for rel in rels:
        step = ev.storage.many_to_one(rel)
        index = step if index is None else step.gather(index)
    assert index is not None
    return index


def related_relationships(ev: Evaluator, start: Table, target: Table) -> list[Relationship]:
    """The relationships RELATED follows from ``start`` to ``target``, in order."""
    frontier: list[tuple[Table, list[Relationship]]] = [(start, [])]
    seen = {id(start)}
    while frontier:
        table, path = frontier.pop(0)
        if table is target:
            return path
        for rel in ev.model.relationships:
            if not rel.is_active:
                continue
            if rel.from_table is table:
                other = rel.to_table
            elif rel.cardinality is Cardinality.ONE_TO_ONE and rel.to_table is table:
                raise NoVector  # the reverse direction of a one-to-one: row by row
            else:
                continue
            if id(other) in seen:
                continue
            seen.add(id(other))
            frontier.append((other, [*path, rel]))
    raise NoVector  # let the row-by-row path raise Power BI's error


# --- aggregations ----------------------------------------------------------------------


@dataclass
class Aggregation:
    """Polars expressions over a frame with the values in column ``v``, and how their
    results become the DAX value."""

    build: Callable[[pl.Expr], list[pl.Expr]]
    """The Polars expressions, given the values' column."""
    finish: Callable[[tuple[Any, ...]], Value]
    combine: tuple[str, ...] | None = None
    """How partial results over parts of the rows combine ("sum", "min" or "max" per
    expression), or None when they can't (averages, distinct counts)."""

    def combined(self, prefix: str) -> list[pl.Expr]:
        assert self.combine is not None
        exprs = []
        for i, how in enumerate(self.combine):
            col = pl.col(f"{prefix}{i}")
            exprs.append(getattr(col, how)().alias(f"{prefix}{i}"))
        return exprs

    @property
    def exprs(self) -> list[pl.Expr]:
        return self.build(pl.col("v"))

    def on(self, column: str, prefix: str) -> list[pl.Expr]:
        """The expressions over the values in ``column``, named ``prefix0``, ..."""
        return [e.alias(f"{prefix}{i}") for i, e in enumerate(self.build(pl.col(column)))]

    def named(self, prefix: str) -> list[pl.Expr]:
        return self.on("v", prefix)


def _sum_parts(
    kind: str, values: pl.Series, bound: float | None = None
) -> Callable[[pl.Expr], list[pl.Expr]]:
    """A total and a count. Whole numbers (and ten-thousandths) add up exactly: as
    64-bit integers when no running total can overflow, else as 128-bit integers.
    ``bound`` is the largest absolute value, if known (in ten-thousandths for "cur")."""
    if kind not in ("int", "cur") or not values.dtype.is_integer():
        return lambda v: [v.sum(), v.count()]
    if bound is None:
        bound = _bound(values)
    fits = values.len() * bound < 2**63
    if kind == "int" and not fits:
        raise NoVector  # a running total could overflow 64 bits on the exact path
    if fits:
        return lambda v: [v.sum(), v.count()]
    return lambda v: [v.cast(pl.Int128).sum(), v.count()]


def column_aggregation(ev: Evaluator, name: str, column: Column, values: pl.Series) -> Aggregation:
    """SUM, AVERAGE, MIN, MAX or a COUNT function of a column, as its handler in
    ``_evaluate.py`` computes it. ``values`` are all the values the groups will see,
    for the checks those handlers make."""
    dax_type = column.data_type
    if name == "SUM":
        if dax_type is DaxType.DATETIME:
            raise NotSupportedError("SUM of a Date/Time column")
        if dax_type not in _KINDS:
            raise NoVector
        kind = _KINDS[dax_type]
        parts = _sum_parts(kind, values)

        def finish_sum(r: tuple[Any, ...]) -> Value:
            total, count = r
            if not count:
                return None
            if kind == "int":
                return sc.check_int(int(total))
            if kind == "cur":
                return Decimal(total).quantize(Decimal("0.0001"))
            return float(total)

        return Aggregation(parts, finish_sum, ("sum", "sum"))
    if name == "AVERAGE":
        if dax_type in (DaxType.DATETIME, DaxType.CURRENCY):
            raise NotSupportedError(f"AVERAGE of a {sc.type_label(dax_type)} column")
        if dax_type not in _KINDS:
            raise NoVector
        return Aggregation(
            lambda v: [v.cast(pl.Float64).mean(), v.count()],
            lambda r: float(r[0]) if r[1] else None,
        )
    if name in ("MIN", "MAX"):
        if dax_type is DaxType.BOOLEAN:
            raise NotSupportedError(f"{name} of a True/False column")
        if dax_type not in (*_KINDS, DaxType.DATETIME):
            raise NoVector
        how = name.lower()

        def finish_min_max(r: tuple[Any, ...]) -> Value:
            result, count = r
            if not count:
                return None
            if dax_type is DaxType.INTEGER:
                return int(result)
            if dax_type is DaxType.DECIMAL:
                return float(result)
            return result

        return Aggregation(lambda v: [getattr(v, how)(), v.count()], finish_min_max, (how, "sum"))
    if name == "COUNT" and dax_type is DaxType.BOOLEAN:
        raise NotSupportedError("COUNT of a True/False column (use COUNTA)")
    if name in ("COUNT", "COUNTA", "COUNTBLANK", "DISTINCTCOUNT", "DISTINCTCOUNTNOBLANK"):
        ev._check_empty_text(name, column, values)
        if name.startswith("DISTINCT"):
            ev.storage.check_text(column)
        if dax_type is DaxType.DECIMAL and name.startswith("DISTINCT") and values.is_nan().any():
            raise NoVector
        count: Callable[[pl.Expr], pl.Expr] = {
            "COUNT": lambda v: v.count(),
            "COUNTA": lambda v: v.count(),
            "COUNTBLANK": lambda v: v.null_count(),
            "DISTINCTCOUNT": lambda v: v.n_unique(),
            "DISTINCTCOUNTNOBLANK": lambda v: v.drop_nulls().n_unique(),
        }[name]
        combine = None if name.startswith("DISTINCT") else ("sum",)
        return Aggregation(lambda v: [count(v)], lambda r: int(r[0]) or None, combine)
    raise NoVector


def iterator_aggregation(
    name: str, kind: str, values: pl.Series, result_type: Any, bound: float | None = None
) -> Aggregation:
    """SUMX, AVERAGEX, MINX, MAXX or COUNTX of row values of one kind, with the rules of
    their handlers in ``_iterators.py``."""
    from ._evaluate import convert

    if name == "SUMX":
        parts = _sum_parts(kind, values, bound)

        def finish_sumx(r: tuple[Any, ...]) -> Value:
            total, count = r
            if not count:
                return None
            if kind == "int":
                value: Value = sc.check_int(int(total))
            elif kind == "cur":
                value = to_currency(total)
            else:
                value = float(total)
            return convert(value, result_type)

        return Aggregation(parts, finish_sumx, ("sum", "sum"))
    if name == "AVERAGEX":
        if kind == "int":
            parts = _sum_parts(kind, values, bound)
        else:
            parts = lambda v: [v.cast(pl.Float64).sum(), v.count()]  # noqa: E731

        def finish_averagex(r: tuple[Any, ...]) -> Value:
            total, count = r
            if not count:
                return None
            if kind == "cur":
                raise NotSupportedError("AVERAGEX of Currency values")
            if kind == "int":
                return int(total) / int(count)
            return float(float(total) / int(count))

        return Aggregation(parts, finish_averagex)
    if name in ("MINX", "MAXX"):
        how = name[:3].lower()

        def finish_minx(r: tuple[Any, ...]) -> Value:
            result, count = r
            if not count:
                return None
            if kind == "int":
                result = int(result)
            elif kind == "dec":
                result = float(result)
            else:
                result = to_currency(result)
            return convert(result, result_type)

        return Aggregation(lambda v: [getattr(v, how)(), v.count()], finish_minx, (how, "sum"))
    if name == "COUNTX":
        return Aggregation(lambda v: [v.count()], lambda r: int(r[0]) or None, ("sum",))
    raise NoVector


def _bound(values: pl.Series) -> float:
    if values.len() == values.null_count():
        return 0.0
    if not values.dtype.is_numeric() and not isinstance(values.dtype, pl.Decimal):
        return 0.0
    return max(abs(float(values.min())), abs(float(values.max())))  # type: ignore[arg-type]


ITERATORS = ("SUMX", "AVERAGEX", "MINX", "MAXX", "COUNTX")
COLUMN_AGGREGATIONS = (
    "SUM",
    "AVERAGE",
    "MIN",
    "MAX",
    "COUNT",
    "COUNTA",
    "COUNTBLANK",
    "DISTINCTCOUNT",
    "DISTINCTCOUNTNOBLANK",
)


def aggregate(frame: pl.DataFrame, agg: Aggregation) -> Value:
    """The aggregation over all of ``frame``."""
    row = frame.select(agg.named("a")).row(0)
    return agg.finish(row)


def iterate_rowset(ev: Evaluator, name: str, value: Any, args: list[Any]) -> Value:
    """An iterator over the visible rows of a model table, in one Polars pass.

    Raises :class:`NoVector` when the expression isn't a supported shape.
    """
    from ._evaluate import infer_type

    if len(args) != 2 or isinstance(args[1], _Omitted):
        raise NoVector
    row_expr = RowExpression(ev, value.table, args[1])
    values, kind = row_expr.values(ev._rowset_mask(value))
    result_type = infer_type(Call(name, args))
    agg = iterator_aggregation(name, kind, values, result_type, row_expr.bound)
    return aggregate(values.to_frame("v"), agg)
