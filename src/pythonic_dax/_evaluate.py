"""The formula engine: evaluates expression trees in filter context and row context.

Phase 3 built the evaluation core: filters flowing across relationships, the basic
aggregations, operators, IF/AND/OR/NOT, VAR, measure references (with context
transition), RELATED, RELATEDTABLE, calculated columns, ROW, and the simple form of
SUMMARIZECOLUMNS. Phase 4 adds CALCULATE, CALCULATETABLE, the filter modifiers and the
functions that inspect the filter context. Phase 5 adds iterators, table and window
functions (_iterators.py, _tables.py, _window.py), Phase 6 the scalar function library
(_library.py, _format.py), and Phase 7 time intelligence (_time.py). What the engine
can't reproduce exactly raises :class:`NotSupportedError`.
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
from ._build import row_context_columns
from ._coverage import PLANNED
from ._filters import (
    ColumnFilter,
    FilterContext,
    RelationshipState,
    Storage,
    bool_series,
    normalized,
)
from .errors import DaxError, NotSupportedError
from .expr import (
    BinaryOp,
    Call,
    ColumnName,
    GroupingCall,
    Literal_,
    ModifierCall,
    Node,
    TableCall,
    TableConstructor,
    TableVarBlock,
    TableVarRef,
    UnaryOp,
    UnsupportedDax,
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


LIBRARY_TYPES: dict[str, Callable[[list[Any]], StaticType]] = {}
"""Result-type rules of the Phase 6 functions whose type depends on their arguments."""


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
        rule = LIBRARY_TYPES.get(name)
        if rule is not None:
            return rule(args)
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
        if name in ("CALCULATE", "EARLIER", "EARLIEST"):
            return infer_type(args[0])
        if name in ("SUMX", "MINX", "MAXX"):
            return infer_type(args[1])
        if name in ("AVERAGEX", "PRODUCTX"):
            return DaxType.DECIMAL
        if name in ("SELECTEDVALUE", "LOOKUPVALUE"):
            alternate = _alternate(name, args)
            return unify(
                [infer_type(args[0]), BLANK_TYPE if alternate is None else infer_type(alternate)]
            )
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
    if (
        isinstance(node, TableCall)
        and node.name in ("VALUES", "DISTINCT")
        and is_column(node.args[0])
    ):
        return node.args[0].data_type  # type: ignore[union-attr,no-any-return]
    return getattr(node, "dax_type", None)


def _alternate(name: str, args: list[Any]) -> Any:
    """The alternate result of SELECTEDVALUE or LOOKUPVALUE, or None."""
    if name == "SELECTEDVALUE":
        value = args[1] if len(args) > 1 else None
    else:  # LOOKUPVALUE: result column, then search pairs, then an optional alternate
        value = args[-1] if len(args) % 2 == 0 else None
    return None if value is None or isinstance(value, _Omitted) else value


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
    """A row context: one row of a table (an index into ``Storage.ext``).

    ``columns`` limits the row to some columns, as when a True/False filter iterates
    the values of ``ALL(<columns>)``. Context transition then filters only those.
    """

    table: Table
    index: int
    columns: tuple[Column, ...] | None = None


@dataclass(frozen=True)
class FrameRow:
    """A row context on a row of a computed table (ADDCOLUMNS, VALUES(...), ...)."""

    frame: Frame
    index: int


Row = RowRef | FrameRow


class Precomputed:
    """An argument whose value is already known (SUMMARIZECOLUMNS computes some
    arguments for all groups at once, then calls the function once per group).
    It reports the type of the expression it stands for."""

    __slots__ = ("node", "value")

    def __init__(self, node: Any, value: Value) -> None:
        self.node = node
        self.value = value

    @property
    def dax_type(self) -> StaticType:
        return infer_type(self.node)


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
    rows: tuple[Row, ...] = ()
    env: dict[int, _Lazy] = field(default_factory=dict)
    shadow: FilterContext | None = None  # what ALLSELECTED restores
    group: TableValue | None = None  # the current group of GROUPBY (CURRENTGROUP)


@dataclass
class RowSet:
    """The visible rows of a model table.

    ``blank`` says whether the hidden blank row counts as one of the rows when it is
    visible: it does for VALUES(<table>) and ALL(<table>), not for a table reference.
    """

    table: Table
    mask: pl.Series | None
    blank: bool = False


@dataclass
class ResultColumn:
    name: str
    dax_type: StaticType
    column: Column | None = None  # the model column the values come from (lineage)


@dataclass
class Frame:
    """A table computed by the engine, such as the result of SUMMARIZECOLUMNS.

    ``filterable`` is False for tables whose use as a CALCULATE filter isn't verified.
    """

    columns: list[ResultColumn]
    rows: list[list[Value]]
    filterable: bool = True

    def find(self, column: Column) -> int | None:
        """The position of the column with this lineage, if the table has one."""
        for i, c in enumerate(self.columns):
            if c.column is column:
                return i
        return None

    def find_name(self, name: str) -> int | None:
        """The position of the column named ``[name]`` (an added or renamed column),
        ignoring case."""
        wanted = f"[{name}]".lower()
        for i, c in enumerate(self.columns):
            if c.name.lower() == wanted:
                return i
        return None

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
        self._measures: dict[tuple[Any, ...], Value] = {}

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
        rows = self.storage.ext(table).filter(self._rowset_mask(value))
        present = self.storage.present_columns(table)
        for col in present:
            self.storage.check_text(col)
        columns = [ResultColumn(f"{table.name}[{c.name}]", c.data_type, c) for c in present]
        names = [c.name for c in present]
        return Frame(columns, [list(r) for r in rows.select(names).rows()])

    def _rowset_mask(self, value: RowSet) -> pl.Series:
        """The rows of a RowSet, with the blank row only when the RowSet includes it."""
        mask = value.mask
        if not value.blank:
            real = self.storage.real_rows(value.table)
            return real if mask is None else mask & real
        if mask is None:
            return bool_series(self.storage.ext(value.table).height, True)
        return mask

    # --- scalars -------------------------------------------------------------------------

    def scalar(self, node: Any, ctx: Context) -> Value:
        if isinstance(node, Precomputed):
            return node.value
        if isinstance(node, Literal_):
            v = node.value
            if isinstance(v, Decimal):
                return float(v)
            if isinstance(v, dt.date) and not isinstance(v, dt.datetime):
                return dt.datetime(v.year, v.month, v.day)
            return v
        if is_column(node):
            return self._column_value(node, ctx)
        if isinstance(node, ColumnName):
            return self._named_value(node, ctx)
        if is_measure(node):
            return self._measure(node, ctx)
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
        if isinstance(node, UnsupportedDax):
            raise NotSupportedError(node.reason)
        if getattr(node, "kind", None) == "table":
            return self._table_as_value(self.table(node, ctx))
        raise TypeError(f"can't evaluate {node!r} as a value")

    def _measure(self, measure: Any, ctx: Context) -> Value:
        """A measure: its expression after context transition. A measure's value
        depends only on the filter context it ends up in, so each is computed once per
        filter context in an evaluation."""
        inner = self._transition(ctx, keep_env=False)
        shadow = inner.shadow.signature if inner.shadow is not None else None
        key = (id(measure), inner.filters.signature, shadow)
        if key in self._measures:
            return self._measures[key]
        value = self.scalar(measure.expression, inner)
        self._measures[key] = value
        return value

    def _column_value(self, column: Column, ctx: Context) -> Value:
        """A column's value in the innermost row context that has the column."""
        for row in reversed(ctx.rows):
            if isinstance(row, FrameRow):
                i = row.frame.find(column)
                if i is not None:
                    return row.frame.rows[row.index][i]
                continue
            if row.table is column.table:
                if row.columns is not None and not any(c is column for c in row.columns):
                    continue
                self.storage.check_text(column)
                return self.storage.ext(column.table)[column.name][row.index]
        raise DaxError(
            f"A single value for column '{column.name}' in table '{column.table.name}' cannot "
            "be determined. This can happen when a measure formula refers to a column that "
            "contains many values without specifying an aggregation such as min, max, count, "
            "or sum to get a single result."
        )

    def _named_value(self, node: ColumnName, ctx: Context) -> Value:
        for row in reversed(ctx.rows):
            if isinstance(row, FrameRow):
                i = row.frame.find_name(node.name)
                if i is not None:
                    return row.frame.rows[row.index][i]
            elif node.name in row.table:
                raise NotSupportedError(
                    f"the unqualified column reference [{node.name}] to a model column "
                    f"(write it as {row.table.ref}[{node.name}])"
                )
        raise DaxError(
            f"Column '{node.name}' cannot be found or may not be used in this expression."
        )

    def rows_of(self, value: TableValue) -> list[Row]:
        """One row context per row of a table, as an iterator like SUMX makes them."""
        if isinstance(value, Frame):
            return [FrameRow(value, i) for i in range(len(value.rows))]
        mask = self._rowset_mask(value)
        return [RowRef(value.table, int(i)) for i in mask.arg_true().to_list()]

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
        frame = self.to_frame(self.table(node, ctx))
        if len(frame.columns) != 1:
            raise DaxError(
                "The number of columns in the table must match the number of values on the "
                "left of IN."
            )
        return [row[0] for row in frame.rows]

    # --- function calls ----------------------------------------------------------------

    def scalar_handler(self, name: str) -> Handler:
        handler = _SCALAR_FUNCTIONS.get(name)
        if handler is None:
            raise NotSupportedError(name, PLANNED.get(name))
        return handler

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
        for row in ctx.rows:  # an inner row context replaces an outer one's filters
            filters = filters.apply(self._row_filters(row))
        return Context(filters=filters, env=ctx.env if keep_env else {}, shadow=ctx.shadow)

    def _row_filters(self, row: Row) -> list[ColumnFilter]:
        if isinstance(row, RowRef):
            own = self._row_filter(row)
            if row.columns is not None:
                return [own]
            # A row of a model table filters its expanded table, like a table filter:
            # the related rows on the one side of each relationship are filtered too.
            height = self.storage.ext(row.table).height
            mask = bool_series(height, False).scatter(row.index, True)
            return [own, *self._expanded_filters(row.table, mask)[1:]]
        # A computed row filters each model column it has (its lineage), table by table.
        by_table: dict[int, list[tuple[Column, Value]]] = {}
        for col, value in zip(row.frame.columns, row.frame.rows[row.index], strict=True):
            if col.column is None:
                continue
            pairs = by_table.setdefault(id(col.column.table), [])
            if not any(c is col.column for c, _ in pairs):
                pairs.append((col.column, value))
        filters = []
        for pairs in by_table.values():
            columns = [c for c, _ in pairs]
            for c in columns:
                self.storage.check_text(c)
            values = pl.DataFrame(
                [pl.Series(c.name, [v], dtype=c.data_type.polars_dtype) for c, v in pairs]
            )
            filters.append(ColumnFilter(columns[0].table, columns, values))
        return filters

    def _row_filter(self, row: RowRef) -> ColumnFilter:
        table = row.table
        if row.columns is not None:
            columns = list(row.columns)
        else:
            # Every column of the row, so the filter also replaces outer filters on any
            # of them (a filter on the key alone wouldn't replace a filter on Name).
            columns = [c for c in self.storage.present_columns(table) if c is not self.computing]
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
            handler = _TABLE_FUNCTIONS.get(node.name)
            if handler is not None:
                return handler(self, node.args, ctx)
            raise NotSupportedError(node.name, PLANNED.get(node.name))
        if isinstance(node, ModifierCall):
            raise DaxError(
                f"{node.to_dax()} can only be used as a filter argument of CALCULATE or "
                "CALCULATETABLE."
            )
        if isinstance(node, TableConstructor):
            return self._table_constructor(node, ctx)
        if type(node).__name__ == "_DataTable":
            from ._tables import datatable

            return datatable(self, node, ctx)
        name = getattr(node, "name", type(node).__name__)
        raise NotSupportedError(str(name), PLANNED.get(str(name)))

    def _table_constructor(self, node: TableConstructor, ctx: Context) -> Frame:
        """``{1, 2}`` or ``{(1, "a"), (2, "b")}``: columns named Value, or Value1, Value2..."""
        width = len(node.rows[0]) if node.rows else 1
        names = ["[Value]"] if width == 1 else [f"[Value{i + 1}]" for i in range(width)]
        types = [unify([infer_type(row[i]) for row in node.rows]) for i in range(width)]
        rows = [[self.scalar(v, ctx) for v in row] for row in node.rows]
        rows = [[convert(v, t) for v, t in zip(row, types, strict=True)] for row in rows]
        return Frame([ResultColumn(n, t) for n, t in zip(names, types, strict=True)], rows)

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
        plain: list[Column] = []
        rollup: list[tuple[Column, str]] = []
        i = 0
        while i < len(args) and (is_column(args[i]) or isinstance(args[i], GroupingCall)):
            if isinstance(args[i], GroupingCall):
                items = args[i].args
                rollup += [(items[k], items[k + 1].value) for k in range(0, len(items), 2)]
            else:
                plain.append(args[i])
            i += 1
        filter_args = []
        while i < len(args) and getattr(args[i], "kind", None) == "table":
            filter_args.append(args[i])
            i += 1
        rest = args[i:]
        named = [(name.value, expr) for name, expr in zip(rest[::2], rest[1::2], strict=True)]
        if not named:
            raise NotSupportedError("SUMMARIZECOLUMNS without expressions")
        if ctx.rows:
            raise NotSupportedError("SUMMARIZECOLUMNS inside a row context")

        # Filter tables work like CALCULATETABLE's. ALLSELECTED inside restores this.
        if filter_args:
            ctx = self.calculate_context(filter_args, ctx)
        group = plain + [c for c, _ in rollup]
        for col in group:
            self.storage.check_text(col)
        columns = [ResultColumn(f"{c.table.name}[{c.name}]", c.data_type, c) for c in group]
        columns += [ResultColumn(f"[{n}]", DaxType.BOOLEAN) for _, n in rollup]
        columns += [ResultColumn(f"[{name}]", infer_type(expr)) for name, expr in named]
        rows: list[list[Value]] = []
        # The detail rows, then one level of subtotals per rollup column, last first.
        for rolled in range(len(rollup) + 1):
            active = plain + [c for c, _ in rollup[: len(rollup) - rolled]]
            flags = [k >= len(rollup) - rolled for k in range(len(rollup))]
            for combo, results in self._group_rows(active, named, ctx):
                by_column = dict(zip((id(c) for c in active), combo, strict=True))
                values = [by_column.get(id(c)) for c in group]
                rows.append([*values, *flags, *results])
        return Frame(columns, rows)

    def _group_rows(
        self, group: list[Column], named: list[tuple[str, Any]], ctx: Context
    ) -> list[tuple[tuple[Value, ...], list[Value]]]:
        """Each existing combination of the group-by values, with the expressions'
        results, leaving out combinations where every result is BLANK.

        Columns of one table give the combinations that exist in it, under the filters
        on that table's own columns only (relationships don't narrow them); the tables
        are then combined in every way.
        """
        tables, combos = self.group_combinations(group, ctx)
        from ._batch import batch_group_rows

        batched = batch_group_rows(self, tables, combos, named, ctx)
        if batched is not None:
            return batched
        result = []
        for combo in combos.rows() if combos.width else [()]:
            inner = self.group_context(tables, group, combo, ctx)
            results = [self.scalar(expr, inner) for _, expr in named]
            if any(r is not None for r in results):
                result.append((combo, results))
        return result

    def group_combinations(
        self, group: list[Column], ctx: Context
    ) -> tuple[list[list[Column]], pl.DataFrame]:
        """The group-by columns by table, and their existing combinations: one column
        per group-by column, named by its ``id``, in the order of ``group``."""
        tables: dict[int, list[Column]] = {}
        for col in group:
            tables.setdefault(id(col.table), []).append(col)
        combos = pl.DataFrame()
        for cols in tables.values():
            values = (
                self.storage.own_rows(cols[0].table, ctx.filters)
                .select([c.name for c in cols])
                .unique(maintain_order=True)
                .rename({c.name: f"{id(c)}" for c in cols})
            )
            combos = values if combos.width == 0 else combos.join(values, how="cross")
        if combos.width:
            combos = combos.select([f"{id(c)}" for c in group])
        return list(tables.values()), combos

    def group_context(
        self,
        tables: list[list[Column]],
        group: list[Column],
        combo: tuple[Value, ...],
        ctx: Context,
    ) -> Context:
        """The context SUMMARIZECOLUMNS evaluates its expressions in for one combination
        (values in the order of ``group``)."""
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
                grouping=True,
            )
            for cols in tables
        ]
        return Context(filters=ctx.filters.apply(filters), env=ctx.env, shadow=ctx.filters)

    # --- CALCULATE -------------------------------------------------------------------

    def calculate_context(self, filter_args: Sequence[Any], ctx: Context) -> Context:
        """The context CALCULATE evaluates its expression in.

        The steps follow Power BI: filter arguments are evaluated in the original
        context; then context transition turns row contexts into filters; then the
        modifiers (USERELATIONSHIP, CROSSFILTER, then ALL and friends) change the
        result; last, the filter arguments are applied. Row contexts end here.
        """
        changes = _Changes()
        for arg in filter_args:
            self._filter_argument(arg, ctx, changes, keep=False)

        filters = self._transition(ctx, keep_env=True).filters
        for rel in changes.use:
            filters = self._use_relationship(filters, rel)
        for rel, direction in changes.cross:
            filters = self._cross_filter(filters, rel, direction)
        if changes.remove_all:
            filters = filters.cleared()
        if changes.remove:
            filters = filters.remove(changes.remove)
        if changes.restore_all or changes.restore:
            filters = self._restore(filters, changes, ctx)
        filters = filters.apply(changes.overwrite, changes.keep)
        return Context(filters=filters, env=ctx.env, shadow=ctx.shadow)

    def _filter_argument(self, arg: Any, ctx: Context, changes: _Changes, *, keep: bool) -> None:
        name = getattr(arg, "name", None) if isinstance(arg, TableCall | ModifierCall) else None
        args = getattr(arg, "args", [])
        if name == "KEEPFILTERS":
            if keep:
                raise NotSupportedError("KEEPFILTERS inside KEEPFILTERS")
            inner = args[0]
            inner_name = getattr(inner, "name", None)
            if isinstance(inner, TableCall | ModifierCall) and inner_name in _REMOVERS:
                raise NotSupportedError(f"KEEPFILTERS({inner_name}(...))")
            self._filter_argument(inner, ctx, changes, keep=True)
            return
        if name in ("ALL", "REMOVEFILTERS"):
            if not args:
                changes.remove_all = True
            else:
                changes.remove |= {id(c) for c in self._removal_columns(args)}
            return
        if name == "ALLEXCEPT":
            kept = {id(c) for c in args[1:]}
            changes.remove |= {id(c) for c in self.expanded_columns(args[0]) if id(c) not in kept}
            return
        if name == "ALLSELECTED":
            if ctx.shadow is None:
                raise NotSupportedError("ALLSELECTED outside an iteration such as SUMMARIZECOLUMNS")
            if not args:
                changes.restore_all = True
            else:
                changes.restore.extend(self._removal_columns(args))
            return
        if name == "ALLNOBLANKROW" and not is_column(args[0]) and self._expansion_steps(args[0]):
            # Verified as a plain filter (all rows but the blank one) only for tables
            # that don't expand to other tables.
            raise NotSupportedError("ALLNOBLANKROW of a table on the many side as a filter")
        if name == "USERELATIONSHIP":
            changes.use.append(self._relationship(args[0], args[1]))
            return
        if name == "CROSSFILTER":
            rel = self._relationship(args[0], args[1])
            changes.cross.append((rel, (args[0], args[2].name)))
            return

        target = changes.keep if keep else changes.overwrite
        if getattr(arg, "kind", None) == "table":
            new = self.table_filters(self.table(arg, ctx))
        else:
            new = [self._boolean_filter(arg, ctx)]
        target.extend(new)
        if not keep:
            for f in new:
                if any(self._clears_table(c) for c in f.columns):
                    changes.remove |= {id(c) for c in self.expanded_columns(f.table)}

    def _clears_table(self, column: Column) -> bool:
        """Whether a CALCULATE filter on this column also removes the filters on the
        rest of its table, as ALL(<table>) does.

        Power BI does this for the date column of a marked date table, and for a
        Date/Time column that is the one side of a relationship, marked or not. It
        doesn't for KEEPFILTERS filters.
        """
        if column.data_type is not DaxType.DATETIME:
            return False
        if column.table.date_column is column:
            return True
        sides = [r for r in self.model.relationships if r.to_column is column]
        if any(r.is_active for r in sides):
            return True
        if sides:
            raise NotSupportedError(
                f"a filter on {column.ref}, the one side of only inactive relationships"
            )
        return False

    def _removal_columns(self, args: list[Any]) -> list[Column]:
        columns: list[Column] = []
        for a in args:
            columns.extend([a] if is_column(a) else self.expanded_columns(a))
        return columns

    def _boolean_filter(self, node: Any, ctx: Context) -> ColumnFilter:
        """A True/False filter: ``FILTER(ALL(<its columns>), <condition>)``."""
        columns: list[Column] = []
        for c in row_context_columns(node):
            if not any(c is d for d in columns):
                columns.append(c)
        table = columns[0].table
        for c in columns:
            self.storage.check_text(c)
        names = [c.name for c in columns]
        combos = (
            self.storage.ext(table)
            .select(names)
            .with_row_index("__i")
            .unique(subset=names, keep="first", maintain_order=True)
        )
        hits = _vector_condition(node, combos, columns)
        if hits is None:
            keep = []
            for index in combos["__i"].to_list():
                row = RowRef(table, index, tuple(columns))
                inner = replace(ctx, rows=(*ctx.rows, row))
                keep.append(sc.to_bool(self.scalar(node, inner)))
            hits = pl.Series(keep, dtype=pl.Boolean)
        values = combos.filter(hits).select(names)
        return ColumnFilter(table, columns, values)

    def table_filters(self, value: TableValue) -> list[ColumnFilter]:
        """The filters a table applies when it is a CALCULATE filter argument.

        The rows of a model table filter its expanded table: the table itself and,
        following relationships towards their one side, the related rows of each
        table it reaches (the blank row for keys with no match).
        """
        if isinstance(value, RowSet):
            return self._expanded_filters(value.table, self._rowset_mask(value))
        if not value.filterable:
            raise NotSupportedError("this table expression as a CALCULATE filter")
        lineage = [c.column for c in value.columns]
        if any(c is None for c in lineage):
            raise NotSupportedError("a filter table with columns that aren't model columns")
        columns: list[Column] = [c for c in lineage if c is not None]
        if len({id(c.table) for c in columns}) > 1:
            raise NotSupportedError("a filter table with columns from several tables")
        if len({id(c) for c in columns}) < len(columns):
            raise NotSupportedError("a filter table that names a column twice")
        for c in columns:
            self.storage.check_text(c)
        frame = pl.DataFrame(
            [
                pl.Series(c.name, [row[i] for row in value.rows], dtype=c.data_type.polars_dtype)
                for i, c in enumerate(columns)
            ]
        )
        return [ColumnFilter(columns[0].table, columns, frame)]

    def _expanded_filters(self, table: Table, mask: pl.Series) -> list[ColumnFilter]:
        present = self.storage.present_columns(table)
        for c in present:
            self.storage.check_text(c)
        names = [c.name for c in present]
        result = [ColumnFilter(table, present, self.storage.ext(table).filter(mask).select(names))]
        for many, one in self._expansion_steps(table):
            other = one.table
            self.storage._check_keys(self._relationship(many, one))
            keys = self.storage.ext(table).filter(mask).select(normalized(many).alias("k"))["k"]
            one_ext = self.storage.ext(other)
            one_real = self.storage.real_rows(other)
            one_keys = one_ext.select(normalized(one).alias("k"))["k"]
            other_mask = one_keys.is_in(keys.drop_nulls().implode()).fill_null(False) & one_real
            blank = self.storage.blank_index(other)
            if blank is not None:
                matched = keys.is_in(one_keys.filter(one_real).drop_nulls().implode())
                if bool((~matched.fill_null(False)).any()):
                    other_mask = other_mask.scatter(blank, True)
            result.extend(self._expanded_filters(other, other_mask))
        return result

    def expanded_columns(self, table: Table) -> list[Column]:
        """The columns of a table's expanded table (ALL(<table>) removes all of them)."""
        columns = list(table.columns)
        for _, one in self._expansion_steps(table):
            columns.extend(self.expanded_columns(one.table))
        return columns

    def _expansion_steps(self, table: Table) -> list[tuple[Column, Column]]:
        """Relationships from ``table`` towards their one side, as (many, one) keys."""
        steps = []
        reached = set()
        for rel in self.model.relationships:
            if rel.from_table is not table and rel.to_table is not table:
                continue
            if rel.cardinality is not Cardinality.MANY_TO_ONE:
                raise NotSupportedError(
                    f"table expansion across a {rel.cardinality.value.replace('_', '-')} "
                    "relationship"
                )
            if rel.from_table is table and rel.is_active:
                steps.append((rel.from_column, rel.to_column))
                reached.add(id(rel.to_table))
        for rel in self.model.relationships:
            if rel.from_table is table and not rel.is_active and id(rel.to_table) not in reached:
                raise NotSupportedError(
                    f"table expansion when {table.name} has an inactive relationship to "
                    f"{rel.to_table.name}"
                )
        return steps

    def _relationship(self, a: Column, b: Column) -> Relationship:
        for rel in self.model.relationships:
            if {id(rel.from_column), id(rel.to_column)} == {id(a), id(b)}:
                return rel
        raise DaxError(f"No relationship between {a.ref} and {b.ref}.")

    def _use_relationship(self, filters: FilterContext, rel: Relationship) -> FilterContext:
        if rel.cardinality is Cardinality.MANY_TO_MANY:
            raise NotSupportedError("USERELATIONSHIP on a many-to-many relationship")
        pair = {id(rel.from_table), id(rel.to_table)}
        for other in self.model.relationships:
            if other is rel or {id(other.from_table), id(other.to_table)} != pair:
                continue
            state = filters.state(other)
            if state.active:
                filters = filters.override(other, replace(state, active=False))
        return filters.override(rel, replace(filters.state(rel), active=True))

    def _cross_filter(
        self, filters: FilterContext, rel: Relationship, how: tuple[Column, str]
    ) -> FilterContext:
        _, direction = how
        state = filters.state(rel)
        if not state.active:
            raise NotSupportedError("CROSSFILTER on an inactive relationship")
        if direction == "NONE":
            new = RelationshipState(True, False, False)
        elif direction == "BOTH":
            new = RelationshipState(True, True, True)
        elif direction == "ONEWAY":
            new = RelationshipState(True, True, False)
        else:
            raise NotSupportedError(f"CROSSFILTER with {direction}")
        return filters.override(rel, new)

    def _restore(self, filters: FilterContext, changes: _Changes, ctx: Context) -> FilterContext:
        """ALLSELECTED: put back the filters of the last shadow filter context."""
        assert ctx.shadow is not None
        if changes.restore_all:
            return FilterContext(ctx.shadow.filters, filters.overrides)
        ids = {id(c) for c in changes.restore}
        restored = []
        for f in ctx.shadow.filters:
            rest = [c for c in f.columns if id(c) in ids]
            if not rest:
                continue
            if len(rest) < len(f.columns):
                raise NotSupportedError("ALLSELECTED of part of a filter over several columns")
            restored.append(f)
        return FilterContext((*filters.remove(ids).filters, *restored), filters.overrides)

    # --- tables as values ---------------------------------------------------------------

    def _table_as_value(self, value: TableValue) -> Value:
        """A one-column table used as a single value: BLANK when empty, an error when
        it has more than one row."""
        frame = self.to_frame(value)
        if len(frame.columns) != 1:
            raise DaxError(
                "The expression refers to multiple columns. Multiple columns cannot be "
                "converted to a scalar value."
            )
        if len(frame.rows) > 1:
            raise _multiple_values(frame.columns[0].column)
        return frame.rows[0][0] if frame.rows else None

    def column_distinct(self, column: Column, ctx: Context, *, blank: bool) -> pl.Series:
        """The distinct visible values of a column: VALUES (``blank=True``, including
        the blank row's BLANK when that row is visible) or DISTINCT."""
        self.storage.check_text(column)
        mask = self._rowset_mask(
            RowSet(column.table, self.storage.visible(column.table, ctx.filters), blank)
        )
        return self.storage.ext(column.table)[column.name].filter(mask).unique(maintain_order=True)

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
    # VALUES(<table>) and ALL(<table>) do.
    return int(ev._rowset_mask(value).sum()) or None


def _related(ev: Evaluator, args: list[Any], ctx: Context) -> Value:
    column: Column = args[0]
    if not ctx.rows:
        raise DaxError(
            f"The column '{column.table.name}[{column.name}]' either doesn't exist or doesn't "
            "have a relationship to any table available in the current context."
        )
    last_error: DaxError | None = None
    for row in reversed(ctx.rows):
        if isinstance(row, FrameRow):
            raise NotSupportedError("RELATED in a row context on a computed table")
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


_MULTIPLE_VALUES = "A table of multiple values was supplied where a single value was expected."

_REMOVERS = ("ALL", "REMOVEFILTERS", "ALLEXCEPT", "ALLSELECTED", "ALLNOBLANKROW")


def _multiple_values(column: Column | None) -> DaxError:
    suffix = f" for column '{column.table.name}'[{column.name}]" if column is not None else ""
    return DaxError(f"{_MULTIPLE_VALUES[:-1]}{suffix}.")


@dataclass
class _Changes:
    """What CALCULATE's filter arguments ask for, gathered before any is applied."""

    overwrite: list[ColumnFilter] = field(default_factory=list)
    keep: list[ColumnFilter] = field(default_factory=list)
    remove: set[int] = field(default_factory=set)
    remove_all: bool = False
    restore: list[Column] = field(default_factory=list)
    restore_all: bool = False
    use: list[Relationship] = field(default_factory=list)
    cross: list[tuple[Relationship, tuple[Column, str]]] = field(default_factory=list)


class _NoVector(Exception):
    """The condition isn't one of the simple shapes :func:`_vector_condition` handles."""


_FLIPPED = {"=": "=", "==": "==", "<>": "<>", "<": ">", "<=": ">=", ">": "<", ">=": "<="}
_NUMERIC = (DaxType.INTEGER, DaxType.DECIMAL, DaxType.CURRENCY)


def _vector_condition(node: Any, frame: pl.DataFrame, columns: list[Column]) -> pl.Series | None:
    """A True/False filter evaluated over all value combinations at once.

    Handles comparisons of a column with a number or text literal, IN with a list of
    literals, and &&, || and NOT of those, with the same rules as the scalar
    operators. Anything else returns None and is evaluated one row at a time.
    """
    try:
        expr = _vector_expr(node, frame, columns)
    except _NoVector:
        return None
    return frame.select(expr.fill_null(False).alias("r"))["r"]


def _vector_expr(node: Any, frame: pl.DataFrame, columns: list[Column]) -> pl.Expr:
    if isinstance(node, BinaryOp) and node.op in ("&&", "||"):
        left = _vector_expr(node.left, frame, columns)
        right = _vector_expr(node.right, frame, columns)
        return left & right if node.op == "&&" else left | right
    if isinstance(node, Call) and node.name == "NOT":
        return ~_vector_expr(node.args[0], frame, columns)
    if not isinstance(node, BinaryOp):
        raise _NoVector
    if node.op == "IN" and isinstance(node.right, TableConstructor):
        column = _vector_column(node.left, frame, columns)
        if any(len(row) != 1 or not isinstance(row[0], Literal_) for row in node.right.rows):
            raise _NoVector
        literals = [row[0].value for row in node.right.rows]  # type: ignore[attr-defined]
        present = [v for v in literals if v is not None]
        for v in present:
            _vector_literal(column, v)
        col = pl.col(column.name)
        if column.data_type is DaxType.STRING:
            col = col.str.to_lowercase()
            present = [v.lower() for v in present]
        elif column.data_type is DaxType.CURRENCY:
            col = col.cast(pl.Float64)
        result = col.is_in(present)
        if None in literals:
            result = result | col.is_null()
        return result
    if node.op not in _FLIPPED:
        raise _NoVector
    op, subject, literal = node.op, node.left, node.right
    if isinstance(subject, Literal_):
        op, subject, literal = _FLIPPED[op], literal, subject
    if not isinstance(literal, Literal_) or literal.value is None:
        raise _NoVector
    column = _vector_column(subject, frame, columns)
    value = _vector_literal(column, literal.value)
    col = pl.col(column.name)
    if column.data_type is DaxType.STRING:
        if op not in ("=", "<>", "=="):
            raise _NoVector
        col, value = col.str.to_lowercase(), value.lower()
        blank: Any = ""
    else:
        if column.data_type is DaxType.CURRENCY:
            col = col.cast(pl.Float64)
        blank = 0
    lit = pl.lit(value)
    if op == "==":
        return col == lit  # BLANK only equals BLANK, so a null row is False
    col = col.fill_null(blank)
    return {
        "=": col == lit,
        "<>": col != lit,
        "<": col < lit,
        "<=": col <= lit,
        ">": col > lit,
        ">=": col >= lit,
    }[op]


def _vector_column(node: Any, frame: pl.DataFrame, columns: list[Column]) -> Column:
    if not is_column(node) or not any(node is c for c in columns):
        raise _NoVector
    column: Column = node
    if column.data_type is DaxType.DECIMAL and frame[column.name].is_nan().any():
        raise _NoVector  # comparing NaN isn't supported; let the scalar path say so
    return column


def _vector_literal(column: Column, value: Any) -> Any:
    if column.data_type is DaxType.STRING and isinstance(value, str):
        return value
    if (
        column.data_type in _NUMERIC
        and isinstance(value, int | float)
        and not isinstance(value, bool)
    ):
        return value
    raise _NoVector


def _calculate(ev: Evaluator, args: list[Any], ctx: Context) -> Value:
    return ev.scalar(args[0], ev.calculate_context(args[1:], ctx))


def _calculatetable(ev: Evaluator, args: list[Any], ctx: Context) -> TableValue:
    return ev.table(args[0], ev.calculate_context(args[1:], ctx))


def _all_table(name: str) -> Callable[[Evaluator, list[Any], Context], TableValue]:
    def handler(ev: Evaluator, args: list[Any], ctx: Context) -> TableValue:
        blank = name == "ALL"
        if len(args) == 1 and not is_column(args[0]):
            return RowSet(args[0], None, blank)
        columns: list[Column] = list(args)
        table = columns[0].table
        for c in columns:
            ev.storage.check_text(c)
        names = [c.name for c in columns]
        ext = ev.storage.ext(table)
        rows = ext if blank else ext.filter(ev.storage.real_rows(table))
        values = rows.select(names).unique(maintain_order=True)
        return Frame(
            [ResultColumn(f"{table.name}[{c.name}]", c.data_type, c) for c in columns],
            [list(r) for r in values.rows()],
        )

    return handler


def _values_table(name: str) -> Callable[[Evaluator, list[Any], Context], TableValue]:
    def handler(ev: Evaluator, args: list[Any], ctx: Context) -> TableValue:
        arg = args[0]
        if is_column(arg):
            values = ev.column_distinct(arg, ctx, blank=name == "VALUES")
            column = ResultColumn(f"{arg.table.name}[{arg.name}]", arg.data_type, arg)
            return Frame([column], [[v] for v in values.to_list()])
        if not getattr(arg, "is_table_ref", False):
            from ._tables import _distinct_table

            return _distinct_table(ev, ev.table(arg, ctx))
        mask = ev.storage.visible(arg, ctx.filters)
        if name == "VALUES":
            return RowSet(arg, mask, blank=True)
        frame = ev.to_frame(RowSet(arg, mask))
        unique = (
            pl.DataFrame(frame.rows, orient="row").unique(maintain_order=True)
            if frame.rows
            else None
        )
        rows = [list(r) for r in unique.rows()] if unique is not None else []
        return Frame(frame.columns, rows, filterable=False)

    return handler


def _filters_table(ev: Evaluator, args: list[Any], ctx: Context) -> TableValue:
    column: Column = args[0]
    ev.storage.check_text(column)
    mask = ev.storage.direct_mask(column, ctx.filters)
    values = ev.storage.ext(column.table)[column.name]
    if mask is not None:
        values = values.filter(mask)
    result = ResultColumn(f"{column.table.name}[{column.name}]", column.data_type, column)
    return Frame([result], [[v] for v in values.unique(maintain_order=True).to_list()])


def _treatas(ev: Evaluator, args: list[Any], ctx: Context) -> TableValue:
    columns: list[Column] = list(args[1:])
    frame = ev.to_frame(ev.table(args[0], ctx))
    if len(frame.columns) != len(columns):
        raise DaxError(
            f"The number of columns in the table ({len(frame.columns)}) must match the number "
            f"of column references in TREATAS ({len(columns)})."
        )
    for source, target in zip(frame.columns, columns, strict=True):
        if source.dax_type != target.data_type:
            raise NotSupportedError(
                f"TREATAS from a column of type {source.dax_type} to {target.ref}"
            )
    result = [ResultColumn(f"{c.table.name}[{c.name}]", c.data_type, c) for c in columns]
    return Frame(result, frame.rows)


def _hasonevalue(ev: Evaluator, args: list[Any], ctx: Context) -> Value:
    return ev.column_distinct(args[0], ctx, blank=True).len() == 1


def _hasonefilter(ev: Evaluator, args: list[Any], ctx: Context) -> Value:
    column: Column = args[0]
    if not ctx.filters.direct(column):
        return False
    frame = _filters_table(ev, args, ctx)
    assert isinstance(frame, Frame)
    return len(frame.rows) == 1


def _selectedvalue(ev: Evaluator, args: list[Any], ctx: Context) -> Value:
    values = ev.column_distinct(args[0], ctx, blank=True)
    if values.len() == 1:
        value = values[0]
    else:
        alternate = _alternate("SELECTEDVALUE", args)
        value = None if alternate is None else ev.scalar(alternate, ctx)
    return convert(value, infer_type(Call("SELECTEDVALUE", args)))


def _isfiltered(ev: Evaluator, args: list[Any], ctx: Context) -> Value:
    arg = args[0]
    if is_column(arg):
        return bool(ctx.filters.direct(arg))
    return any(f.table is arg for f in ctx.filters.filters)


def _iscrossfiltered(ev: Evaluator, args: list[Any], ctx: Context) -> Value:
    arg = args[0]
    table = arg.table if is_column(arg) else arg
    return ev.storage.visible(table, ctx.filters) is not None


def _isinscope(ev: Evaluator, args: list[Any], ctx: Context) -> Value:
    column: Column = args[0]
    return any(f.grouping for f in ctx.filters.direct(column))


def _lookupvalue(ev: Evaluator, args: list[Any], ctx: Context) -> Value:
    result: Column = args[0]
    alternate = _alternate("LOOKUPVALUE", args)
    end = len(args) - 1 if alternate is not None else len(args)
    table = result.table
    ext = ev.storage.ext(table)
    # Filters on the table's expanded table are ignored, as with REMOVEFILTERS(<table>);
    # filters that reach it through a two-way relationship still apply.
    filters = ctx.filters.remove({id(c) for c in ev.expanded_columns(table)})
    visible = ev.storage.visible(table, filters)
    mask = ev.storage.real_rows(table)
    if visible is not None:
        mask = mask & visible
    for i in range(1, end, 2):
        column: Column = args[i]
        if column.table is not table:
            raise NotSupportedError("LOOKUPVALUE with search columns from another table")
        wanted = ev.scalar(args[i + 1], ctx)
        if wanted is None:
            raise NotSupportedError("LOOKUPVALUE searching for BLANK")
        ev.storage.check_text(column)
        values = ext[column.name].to_list()
        hits = [v is not None and sc.compare("=", v, wanted) for v in values]
        mask = mask & pl.Series(hits, dtype=pl.Boolean)
    ev.storage.check_text(result)
    found = ext[result.name].filter(mask).unique()
    if found.len() == 1:
        value = found[0]
    elif found.len() == 0 or alternate is not None:
        value = None if alternate is None else ev.scalar(alternate, ctx)
    else:
        raise _multiple_values(result)
    return convert(value, infer_type(Call("LOOKUPVALUE", args)))


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
    "CALCULATE": _calculate,
    "HASONEVALUE": _hasonevalue,
    "HASONEFILTER": _hasonefilter,
    "SELECTEDVALUE": _selectedvalue,
    "ISFILTERED": _isfiltered,
    "ISCROSSFILTERED": _iscrossfiltered,
    "ISINSCOPE": _isinscope,
    "LOOKUPVALUE": _lookupvalue,
}

_TABLE_FUNCTIONS: dict[str, Callable[[Evaluator, list[Any], Context], TableValue]] = {
    "CALCULATETABLE": _calculatetable,
    "ALL": _all_table("ALL"),
    "ALLNOBLANKROW": _all_table("ALLNOBLANKROW"),
    "VALUES": _values_table("VALUES"),
    "DISTINCT": _values_table("DISTINCT"),
    "FILTERS": _filters_table,
    "TREATAS": _treatas,
}

EVALUATED = (
    frozenset(_SCALAR_FUNCTIONS)
    | frozenset(_TABLE_FUNCTIONS)
    | {
        "ROW", "SUMMARIZECOLUMNS", "RELATEDTABLE", "ALLEXCEPT", "ALLSELECTED",
        "REMOVEFILTERS", "KEEPFILTERS", "USERELATIONSHIP", "CROSSFILTER",
    }
)  # fmt: skip
"""Functions this engine evaluates (some only in their simple forms)."""


def _register() -> None:
    from . import _library, _time, _window
    from ._iterators import SCALAR
    from ._tables import TABLES

    _SCALAR_FUNCTIONS.update(SCALAR)
    _SCALAR_FUNCTIONS.update(_window.SCALAR)
    _TABLE_FUNCTIONS.update(TABLES)
    _TABLE_FUNCTIONS.update(_window.TABLES)
    _SCALAR_FUNCTIONS.update(_library.SCALAR)
    _TABLE_FUNCTIONS.update(_library.TABLES)
    LIBRARY_TYPES.update(_library.TYPES)
    _SCALAR_FUNCTIONS.update(_time.SCALAR)
    _TABLE_FUNCTIONS.update(_time.TABLES)


_register()
EVALUATED = (
    EVALUATED
    | frozenset(_SCALAR_FUNCTIONS)
    | frozenset(_TABLE_FUNCTIONS)
    | {"DATATABLE", "ORDERBY", "PARTITIONBY", "ROLLUPADDISSUBTOTAL"}
)
