"""The formula engine: evaluates expression trees in filter context and row context.

Phase 3 covers the evaluation core: filters flowing across relationships, the basic
aggregations, operators, IF/AND/OR/NOT, VAR, measure references (with context
transition), RELATED, RELATEDTABLE, calculated columns, ROW, and the simple form of
SUMMARIZECOLUMNS. Everything else raises :class:`NotSupportedError` naming its phase.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from decimal import Decimal
from functools import partial
from typing import TYPE_CHECKING, Any

import polars as pl

from . import _scalar as sc
from ._coverage import PLANNED
from ._filters import ColumnFilter, FilterContext, Storage
from .errors import DaxError, NotSupportedError
from .expr import (
    BinaryOp,
    Call,
    Literal_,
    Node,
    TableCall,
    TableConstructor,
    TableVarBlock,
    TableVarRef,
    UnaryOp,
    VarBlock,
    VarRef,
    _Omitted,
    is_column,
    is_measure,
)
from .objects import Cardinality
from .types import DaxType

if TYPE_CHECKING:
    from .model import Model
    from .objects import Column, Relationship, Table

Value = Any


# --- static result types ------------------------------------------------------------

BLANK_TYPE = "blank"
"""The type of an expression that is always BLANK, like ``BLANK()``."""

StaticType = DaxType | str | None  # a DaxType, BLANK_TYPE, or None when unknown

_COMPARISONS = ("=", "==", "<>", "<", "<=", ">", ">=", "IN", "&&", "||")
_COUNTS = ("COUNT", "COUNTA", "COUNTBLANK", "COUNTROWS", "DISTINCTCOUNT", "DISTINCTCOUNTNOBLANK")


def infer_type(node: Any) -> StaticType:
    """The DAX type an expression returns, worked out without evaluating it."""
    if isinstance(node, Literal_):
        v = node.value
        if v is None:
            return BLANK_TYPE
        if isinstance(v, Decimal):
            return DaxType.DECIMAL
        return node.dax_type
    if is_column(node):
        return node.data_type  # type: ignore[no-any-return]
    if is_measure(node):
        return infer_type(node.expression)
    if isinstance(node, VarRef):
        return infer_type(node.value)
    if isinstance(node, VarBlock):
        return infer_type(node.body)
    if isinstance(node, BinaryOp):
        if node.op in _COMPARISONS:
            return DaxType.BOOLEAN
        if node.op == "&":
            return DaxType.STRING
        return arithmetic_type(node.op, infer_type(node.left), infer_type(node.right))
    if isinstance(node, UnaryOp):
        return infer_type(node.operand)
    if isinstance(node, Call):
        name = node.name
        args = node.args
        if name == "BLANK":
            return BLANK_TYPE
        if name in ("TRUE", "FALSE", "AND", "OR", "NOT"):
            return DaxType.BOOLEAN
        if name in _COUNTS:
            return DaxType.INTEGER
        if name in ("SUM", "RELATED") or (name in ("MIN", "MAX") and len(args) == 1):
            return infer_type(args[0])
        if name == "AVERAGE":
            return DaxType.DECIMAL
        if name in ("MIN", "MAX"):
            return unify([infer_type(args[0]), infer_type(args[1])])
        if name == "IF":
            otherwise = args[2] if len(args) > 2 and not isinstance(args[2], _Omitted) else None
            return unify(
                [
                    infer_type(args[1]),
                    infer_type(otherwise) if otherwise is not None else BLANK_TYPE,
                ]
            )
        return node.dax_type
    return getattr(node, "dax_type", None)


def arithmetic_type(op: str, a: StaticType, b: StaticType) -> StaticType:
    """The type of ``a op b``, matching what Power BI reports for each combination."""
    if op == "^":
        return DaxType.DECIMAL
    if a == BLANK_TYPE and b == BLANK_TYPE:
        return DaxType.DECIMAL if op == "/" else DaxType.INTEGER
    kinds = []
    for t in (a, b):
        if t == BLANK_TYPE or t is DaxType.BOOLEAN:
            t = DaxType.INTEGER
        if t is None or t is DaxType.STRING:
            return None
        kinds.append(t)
    x, y = kinds
    if DaxType.DATETIME in (x, y):
        return DaxType.DATETIME if op in ("+", "-") else DaxType.DECIMAL
    if op in ("+", "-"):
        if x is DaxType.INTEGER and y is DaxType.INTEGER:
            return DaxType.INTEGER
        if DaxType.DECIMAL not in (x, y):
            return DaxType.CURRENCY
        return DaxType.DECIMAL
    if op == "*":
        if x is DaxType.INTEGER and y is DaxType.INTEGER:
            return DaxType.INTEGER
        if x is DaxType.CURRENCY and y is DaxType.CURRENCY:
            return DaxType.DECIMAL
        if DaxType.CURRENCY in (x, y):
            return DaxType.CURRENCY
        return DaxType.DECIMAL
    if op == "/":
        if x is DaxType.CURRENCY and y in (DaxType.INTEGER, DaxType.DECIMAL):
            return DaxType.CURRENCY
        return DaxType.DECIMAL
    raise ValueError(op)


def unify(types: Sequence[StaticType]) -> StaticType:
    """The type of an expression that returns one of several branches (IF, MIN)."""
    known = {t for t in types if t != BLANK_TYPE}
    if not known:
        return BLANK_TYPE
    if None in known:
        return None
    if len(known) == 1:
        return known.pop()
    if known == {DaxType.INTEGER, DaxType.DECIMAL}:
        return DaxType.DECIMAL
    return None


def convert(value: Value, dax_type: StaticType) -> Value:
    """Convert a branch's value to the expression's type, as IF does (1 becomes 1.0)."""
    if value is None or not isinstance(dax_type, DaxType):
        return value
    if (
        dax_type is DaxType.DECIMAL
        and isinstance(value, int | Decimal)
        and not isinstance(value, bool)
    ):
        return float(value)
    return value


# --- evaluation state -----------------------------------------------------------------


@dataclass(frozen=True)
class RowRef:
    """A row context: one row of a table (an index into ``Storage.ext``)."""

    table: Table
    index: int


class _Lazy:
    """A variable's value, evaluated the first time it is used."""

    __slots__ = ("compute", "done", "value")

    def __init__(self, compute: Callable[[], Any]) -> None:
        self.compute = compute
        self.done = False
        self.value: Any = None

    def get(self) -> Any:
        if not self.done:
            self.value = self.compute()
            self.done = True
        return self.value


@dataclass(frozen=True)
class Context:
    filters: FilterContext = field(default_factory=FilterContext)
    rows: tuple[RowRef, ...] = ()
    env: dict[int, _Lazy] = field(default_factory=dict)


@dataclass
class RowSet:
    """The visible rows of a model table."""

    table: Table
    mask: pl.Series | None


@dataclass
class ResultColumn:
    name: str
    dax_type: StaticType


@dataclass
class Frame:
    """A table computed by the engine, such as the result of SUMMARIZECOLUMNS."""

    columns: list[ResultColumn]
    rows: list[list[Value]]

    def to_polars(self) -> pl.DataFrame:
        series = []
        for i, col in enumerate(self.columns):
            values = [row[i] for row in self.rows]
            if col.dax_type == BLANK_TYPE:
                dax_type = DaxType.INTEGER  # Power BI types an always-BLANK result as Int64
            elif isinstance(col.dax_type, DaxType):
                dax_type = col.dax_type
            else:
                dax_type = _type_of(values)
            values = [convert(v, dax_type) for v in values]
            series.append(pl.Series(col.name, values, dtype=dax_type.polars_dtype))
        return pl.DataFrame(series)


def _type_of(values: list[Value]) -> DaxType:
    kinds = {sc.type_of(v) for v in values if v is not None}
    if kinds == {DaxType.INTEGER, DaxType.DECIMAL}:
        return DaxType.DECIMAL
    if len(kinds) == 1:
        return kinds.pop()  # type: ignore[return-value]
    if not kinds:
        raise NotSupportedError(
            "a result column whose values are all BLANK and whose type is unknown"
        )
    raise NotSupportedError("a result column with values of different types")


TableValue = RowSet | Frame


# --- the evaluator ----------------------------------------------------------------------


class Evaluator:
    def __init__(self, model: Model, computing: Column | None = None) -> None:
        self.model = model
        self.storage = Storage(model)
        self.computing = computing  # the calculated column being computed, if any

    # --- entry points ---------------------------------------------------------------

    def evaluate(self, node: Node) -> pl.DataFrame:
        """``EVALUATE <table expression>``."""
        value = self.table(node, Context())
        return self.to_frame(value).to_polars()

    def column_values(self, table: Table, node: Node) -> list[Value]:
        """A calculated column: the expression evaluated in the row context of each row."""
        ext = self.storage.ext(table)
        count = ext.height - (1 if self.storage.blank_index(table) is not None else 0)
        return [self.scalar(node, Context(rows=(RowRef(table, i),))) for i in range(count)]

    def to_frame(self, value: TableValue) -> Frame:
        if isinstance(value, Frame):
            return value
        table = value.table
        real = self.storage.real_rows(table)
        rows = self.storage.ext(table).filter(real if value.mask is None else value.mask & real)
        for col in table.columns:
            self.storage.check_text(col)
        columns = [ResultColumn(f"{table.name}[{c.name}]", c.data_type) for c in table.columns]
        names = [c.name for c in table.columns]
        return Frame(columns, [list(r) for r in rows.select(names).rows()])

    # --- scalars -------------------------------------------------------------------------

    def scalar(self, node: Any, ctx: Context) -> Value:
        if isinstance(node, Literal_):
            v = node.value
            if isinstance(v, Decimal):
                return float(v)
            if isinstance(v, dt.date) and not isinstance(v, dt.datetime):
                return dt.datetime(v.year, v.month, v.day)
            return v
        if is_column(node):
            return self._column_value(node, ctx)
        if is_measure(node):
            return self.scalar(node.expression, self._transition(ctx, keep_env=False))
        if isinstance(node, VarRef):
            return self._variable(node, ctx)
        if isinstance(node, VarBlock):
            return self.scalar(node.body, self._bind(node.variables, ctx))
        if isinstance(node, BinaryOp):
            return self._binary(node, ctx)
        if isinstance(node, UnaryOp):
            value = self.scalar(node.operand, ctx)
            return sc.negate(value) if node.op == "-" else value
        if isinstance(node, Call):
            return self._call(node, ctx)
        if getattr(node, "kind", None) == "table":
            raise NotSupportedError("using a table as a single value", phase=5)
        raise TypeError(f"can't evaluate {node!r} as a value")

    def _column_value(self, column: Column, ctx: Context) -> Value:
        for row in reversed(ctx.rows):
            if row.table is column.table:
                self.storage.check_text(column)
                return self.storage.ext(column.table)[column.name][row.index]
        raise DaxError(
            f"A single value for column '{column.name}' in table '{column.table.name}' cannot "
            "be determined. This can happen when a measure formula refers to a column that "
            "contains many values without specifying an aggregation such as min, max, count, "
            "or sum to get a single result."
        )

    def _variable(self, node: Any, ctx: Context) -> Any:
        lazy = ctx.env.get(id(node.value))
        if lazy is None:
            raise DaxError(
                f"The variable {node.name} is used outside the VAR block that defines it."
            )
        return lazy.get()

    def _bind(self, variables: list[tuple[str, Node]], ctx: Context) -> Context:
        env = dict(ctx.env)
        for _, value in variables:
            here = replace(ctx, env=dict(env))
            compute = self.table if value.kind == "table" else self.scalar
            env[id(value)] = _Lazy(partial(compute, value, here))
        return replace(ctx, env=env)

    def _binary(self, node: BinaryOp, ctx: Context) -> Value:
        op = node.op
        if op == "&&":
            return sc.to_bool(self.scalar(node.left, ctx)) and sc.to_bool(
                self.scalar(node.right, ctx)
            )
        if op == "||":
            return sc.to_bool(self.scalar(node.left, ctx)) or sc.to_bool(
                self.scalar(node.right, ctx)
            )
        if op == "IN":
            value = self.scalar(node.left, ctx)
            return any(sc.compare("==", value, v) for v in self._single_column(node.right, ctx))
        a = self.scalar(node.left, ctx)
        b = self.scalar(node.right, ctx)
        if op in ("+", "-", "*", "/", "^"):
            return sc.arithmetic(op, a, b)
        if op == "&":
            return sc.concatenate(a, b)
        return sc.compare(op, a, b)

    def _single_column(self, node: Any, ctx: Context) -> list[Value]:
        if isinstance(node, TableConstructor):
            if node.rows and len(node.rows[0]) != 1:
                raise DaxError(
                    "The number of columns in the table must match the number of values on the "
                    "left of IN."
                )
            return [self.scalar(row[0], ctx) for row in node.rows]
        raise NotSupportedError("IN with a table expression other than a list", phase=5)

    # --- function calls ----------------------------------------------------------------

    def _call(self, node: Call, ctx: Context) -> Value:
        name = node.name
        args = node.args
        handler = _SCALAR_FUNCTIONS.get(name)
        if handler is None:
            raise NotSupportedError(name, PLANNED.get(name))
        return handler(self, args, ctx)

    def _visible_column(self, column: Column, ctx: Context) -> pl.Series:
        """The column's values in the visible rows. Aggregations never see the blank row."""
        mask = self.storage.visible(column.table, ctx.filters)
        real = self.storage.real_rows(column.table)
        values = self.storage.ext(column.table)[column.name]
        return values.filter(real if mask is None else mask & real)

    def _check_empty_text(self, function: str, column: Column, values: pl.Series) -> None:
        if column.data_type is DaxType.STRING and (values == "").any():
            raise NotSupportedError(f"{function} over empty text values in {column.ref}")

    def _transition(self, ctx: Context, *, keep_env: bool) -> Context:
        """Context transition: each row context becomes a filter on its row's values."""
        filters = ctx.filters
        if ctx.rows:
            new: list[ColumnFilter] = []
            for row in ctx.rows:
                new.append(self._row_filter(row))
            filters = filters.apply(new)
        return Context(filters=filters, env=ctx.env if keep_env else {})

    def _row_filter(self, row: RowRef) -> ColumnFilter:
        table = row.table
        present = set(self.storage.ext(table).columns)
        key = self.model.row_identifier(table)
        if key is not None:
            columns = [key]
        else:
            columns = [c for c in table.columns if c.name in present and c is not self.computing]
        return self.storage.row_filter(table, row.index, columns)

    # --- tables --------------------------------------------------------------------------

    def table(self, node: Any, ctx: Context) -> TableValue:
        if getattr(node, "is_table_ref", False):
            if node.model is not self.model:
                raise DaxError(f"{node.ref} belongs to a different model")
            return RowSet(node, self.storage.visible(node, ctx.filters))
        if isinstance(node, TableVarRef):
            value: TableValue = self._variable(node, ctx)
            return value
        if isinstance(node, TableVarBlock):
            return self.table(node.body, self._bind(node.variables, ctx))
        if isinstance(node, TableCall):
            if node.name == "ROW":
                return self._row(node.args, ctx)
            if node.name == "SUMMARIZECOLUMNS":
                return self._summarizecolumns(node.args, ctx)
            if node.name == "RELATEDTABLE":
                return self._relatedtable(node.args, ctx)
            raise NotSupportedError(node.name, PLANNED.get(node.name))
        if isinstance(node, TableConstructor):
            raise NotSupportedError("table constructors outside IN", phase=5)
        name = getattr(node, "name", type(node).__name__)
        raise NotSupportedError(str(name), PLANNED.get(str(name)))

    def _row(self, args: list[Any], ctx: Context) -> Frame:
        columns: list[ResultColumn] = []
        values: list[Value] = []
        for name, expr in zip(args[::2], args[1::2], strict=True):
            columns.append(ResultColumn(f"[{name.value}]", infer_type(expr)))
            values.append(self.scalar(expr, ctx))
        return Frame(columns, [values])

    def _relatedtable(self, args: list[Any], ctx: Context) -> RowSet:
        table = args[0]
        inner = self._transition(ctx, keep_env=True)
        return RowSet(table, self.storage.visible(table, inner.filters))

    def _summarizecolumns(self, args: list[Any], ctx: Context) -> Frame:
        group: list[Column] = []
        named: list[tuple[str, Any]] = []
        i = 0
        while i < len(args) and is_column(args[i]):
            group.append(args[i])
            i += 1
        rest = args[i:]
        if any(getattr(a, "kind", None) == "table" for a in rest):
            raise NotSupportedError("SUMMARIZECOLUMNS with filter tables", phase=5)
        for name, expr in zip(rest[::2], rest[1::2], strict=True):
            named.append((name.value, expr))
        if not named:
            raise NotSupportedError("SUMMARIZECOLUMNS without expressions", phase=5)

        tables: dict[int, list[Column]] = {}
        for col in group:
            self.storage.check_text(col)
            tables.setdefault(id(col.table), []).append(col)
        combos = pl.DataFrame()
        for cols in tables.values():
            table = cols[0].table
            values = (
                self.storage.visible_rows(table, ctx.filters)
                .select([c.name for c in cols])
                .unique(maintain_order=True)
            )
            values = values.rename({c.name: f"{id(c)}" for c in cols})
            combos = values if combos.width == 0 else combos.join(values, how="cross")

        columns = [ResultColumn(f"{c.table.name}[{c.name}]", c.data_type) for c in group]
        columns += [ResultColumn(f"[{name}]", infer_type(expr)) for name, expr in named]
        rows: list[list[Value]] = []
        combo_rows = combos.rows() if combos.width else [()]
        for combo in combo_rows:
            by_column = dict(zip((id(c) for c in group), combo, strict=True))
            filters = [
                ColumnFilter(
                    cols[0].table,
                    cols,
                    pl.DataFrame(
                        [
                            pl.Series(c.name, [by_column[id(c)]], dtype=c.data_type.polars_dtype)
                            for c in cols
                        ]
                    ),
                )
                for cols in tables.values()
            ]
            inner = Context(filters=ctx.filters.apply(filters), env=ctx.env)
            results = [self.scalar(expr, inner) for _, expr in named]
            if all(r is None for r in results):
                continue
            rows.append([*combo, *results])
        return Frame(columns, rows)

    # --- relationships ------------------------------------------------------------------

    def related_path(self, start: Table, target: Table) -> list[tuple[Column, Column]]:
        """The chain of (this side, other side) keys from ``start`` to ``target``,
        following active relationships towards their one side."""
        frontier: list[tuple[Table, list[tuple[Column, Column]]]] = [(start, [])]
        seen = {id(start)}
        while frontier:
            table, path = frontier.pop(0)
            if table is target:
                return path
            for rel in self.model.relationships:
                if not rel.is_active:
                    continue
                step = _towards_one(rel, table)
                if step is None or id(step[1].table) in seen:
                    continue
                seen.add(id(step[1].table))
                frontier.append((step[1].table, [*path, step]))
        raise DaxError(
            f"The column '{target.name}' either doesn't exist or doesn't have a relationship to "
            "any table available in the current context."
        )


def _towards_one(rel: Relationship, table: Table) -> tuple[Column, Column] | None:
    if rel.from_table is table:
        return rel.from_column, rel.to_column
    if rel.cardinality is Cardinality.ONE_TO_ONE and rel.to_table is table:
        return rel.to_column, rel.from_column
    return None


# --- scalar functions ---------------------------------------------------------------------

Handler = Callable[[Evaluator, list[Any], Context], Value]


def _blank(ev: Evaluator, args: list[Any], ctx: Context) -> Value:
    return None


def _true(ev: Evaluator, args: list[Any], ctx: Context) -> Value:
    return True


def _false(ev: Evaluator, args: list[Any], ctx: Context) -> Value:
    return False


def _if(ev: Evaluator, args: list[Any], ctx: Context) -> Value:
    condition = sc.to_bool(ev.scalar(args[0], ctx))
    otherwise = args[2] if len(args) > 2 and not isinstance(args[2], _Omitted) else None
    branch = args[1] if condition else otherwise
    value = ev.scalar(branch, ctx) if branch is not None else None
    return convert(value, infer_type(Call("IF", args)))


def _and(ev: Evaluator, args: list[Any], ctx: Context) -> Value:
    return sc.to_bool(ev.scalar(args[0], ctx)) and sc.to_bool(ev.scalar(args[1], ctx))


def _or(ev: Evaluator, args: list[Any], ctx: Context) -> Value:
    return sc.to_bool(ev.scalar(args[0], ctx)) or sc.to_bool(ev.scalar(args[1], ctx))


def _not(ev: Evaluator, args: list[Any], ctx: Context) -> Value:
    return not sc.to_bool(ev.scalar(args[0], ctx))


def _sum(ev: Evaluator, args: list[Any], ctx: Context) -> Value:
    column = args[0]
    if column.data_type is DaxType.DATETIME:
        raise NotSupportedError("SUM of a Date/Time column")
    values = ev._visible_column(column, ctx).drop_nulls()
    if values.len() == 0:
        return None
    total = values.sum()
    if column.data_type is DaxType.INTEGER:
        return sc.check_int(int(total))
    if column.data_type is DaxType.CURRENCY:
        return Decimal(total).quantize(Decimal("0.0001"))
    return float(total)


def _average(ev: Evaluator, args: list[Any], ctx: Context) -> Value:
    column = args[0]
    if column.data_type in (DaxType.DATETIME, DaxType.CURRENCY):
        raise NotSupportedError(f"AVERAGE of a {sc.type_label(column.data_type)} column")
    values = ev._visible_column(column, ctx).drop_nulls()
    if values.len() == 0:
        return None
    return float(values.cast(pl.Float64).mean())  # type: ignore[arg-type]


def _min_max(name: str) -> Handler:
    def handler(ev: Evaluator, args: list[Any], ctx: Context) -> Value:
        if len(args) == 2:
            return sc.scalar_min_max(name, ev.scalar(args[0], ctx), ev.scalar(args[1], ctx))
        column = args[0]
        if column.data_type is DaxType.BOOLEAN:
            raise NotSupportedError(f"{name} of a True/False column")
        values = ev._visible_column(column, ctx).drop_nulls()
        if values.len() == 0:
            return None
        if column.data_type is DaxType.STRING:
            ev.storage.check_text(column)
            if not all(sc._ORDERABLE_TEXT.match(v) for v in values.unique().to_list()):
                raise NotSupportedError(
                    f"{name} of text that isn't plain letters, digits and spaces"
                )
            ordered = sorted(values.unique().to_list(), key=str.lower)
            return ordered[0] if name == "MIN" else ordered[-1]
        result = values.min() if name == "MIN" else values.max()
        if column.data_type is DaxType.INTEGER:
            return int(result)  # type: ignore[arg-type]
        if column.data_type is DaxType.DECIMAL:
            return float(result)  # type: ignore[arg-type]
        return result

    return handler


def _count(name: str) -> Handler:
    def handler(ev: Evaluator, args: list[Any], ctx: Context) -> Value:
        column = args[0]
        if name == "COUNT" and column.data_type is DaxType.BOOLEAN:
            raise NotSupportedError("COUNT of a True/False column (use COUNTA)")
        values = ev._visible_column(column, ctx)
        ev._check_empty_text(name, column, values)
        if name in ("COUNT", "COUNTA"):
            count = values.len() - values.null_count()
        elif name == "COUNTBLANK":
            count = values.null_count()
        elif name == "DISTINCTCOUNT":
            ev.storage.check_text(column)
            count = values.n_unique()
        else:  # DISTINCTCOUNTNOBLANK
            ev.storage.check_text(column)
            count = values.drop_nulls().n_unique()
        return count or None

    return handler


def _countrows(ev: Evaluator, args: list[Any], ctx: Context) -> Value:
    if not args or isinstance(args[0], _Omitted):
        raise NotSupportedError("COUNTROWS without a table")
    value = ev.table(args[0], ctx)
    if isinstance(value, Frame):
        return len(value.rows) or None
    # A table reference never includes the blank row, even when the blank row is visible.
    real = ev.storage.real_rows(value.table)
    count = int((real if value.mask is None else value.mask & real).sum())
    return count or None


def _related(ev: Evaluator, args: list[Any], ctx: Context) -> Value:
    column: Column = args[0]
    if not ctx.rows:
        raise DaxError(
            f"The column '{column.table.name}[{column.name}]' either doesn't exist or doesn't "
            "have a relationship to any table available in the current context."
        )
    last_error: DaxError | None = None
    for row in reversed(ctx.rows):
        try:
            path = ev.related_path(row.table, column.table)
        except DaxError as error:
            last_error = error
            continue
        index: int | None = row.index
        table = row.table
        for here, there in path:
            if index is None:
                break
            ev.storage.check_text(here)
            key = ev.storage.ext(table)[here.name][index]
            index = _lookup(ev, there, key)
            table = there.table
        if index is None:
            return None
        ev.storage.check_text(column)
        return ev.storage.ext(column.table)[column.name][index]
    assert last_error is not None
    raise last_error


def _lookup(ev: Evaluator, column: Column, key: Value) -> int | None:
    if key is None:
        return None
    ext = ev.storage.ext(column.table)
    values = ext[column.name]
    if column.data_type is DaxType.STRING:
        hits = (values.str.to_lowercase() == str(key).lower()).arg_true()
    else:
        hits = (values == key).arg_true()
    real = [i for i in hits.to_list() if i != ev.storage.blank_index(column.table)]
    return real[0] if real else None


_SCALAR_FUNCTIONS: dict[str, Handler] = {
    "BLANK": _blank,
    "TRUE": _true,
    "FALSE": _false,
    "IF": _if,
    "AND": _and,
    "OR": _or,
    "NOT": _not,
    "SUM": _sum,
    "AVERAGE": _average,
    "MIN": _min_max("MIN"),
    "MAX": _min_max("MAX"),
    "COUNT": _count("COUNT"),
    "COUNTA": _count("COUNTA"),
    "COUNTBLANK": _count("COUNTBLANK"),
    "DISTINCTCOUNT": _count("DISTINCTCOUNT"),
    "DISTINCTCOUNTNOBLANK": _count("DISTINCTCOUNTNOBLANK"),
    "COUNTROWS": _countrows,
    "RELATED": _related,
}

EVALUATED = frozenset(_SCALAR_FUNCTIONS) | {"ROW", "SUMMARIZECOLUMNS", "RELATEDTABLE"}
"""Functions this engine evaluates (some only in their simple forms)."""
