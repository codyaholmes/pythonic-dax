"""DAX functions whose arguments don't fit a fixed list: name/value pairs, optional
sort orders, and functions with two forms (MIN, MAX).

Name/expression pairs can be written as DAX does (``"Total", total``) or as a dict
(``{"Total": total}``). The two forms can be mixed, and the order is kept.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from ._build import check_arg
from ._specs import Param, Spec
from .errors import DaxError
from .expr import (
    OMITTED,
    Arg,
    Call,
    Expr,
    Keyword,
    Literal_,
    ModifierExpr,
    Node,
    TableCall,
    TableExpr,
    coerce,
    coerce_any,
    coerce_table,
    is_column,
    keyword,
)
from .types import DaxType

__all__ = [
    "ADDCOLUMNS",
    "CONCATENATEX",
    "CONTAINS",
    "DATATABLE",
    "GROUPBY",
    "IN",
    "LOOKUPVALUE",
    "MAX",
    "MIN",
    "ORDERBY",
    "ROW",
    "SELECTCOLUMNS",
    "SUMMARIZE",
    "SUMMARIZECOLUMNS",
    "SWITCH",
    "TOPN",
]


def _spec(name: str) -> Spec:
    return Spec(name=name, params=(), returns="S", summary="")


def _column(function: str, value: Any, position: int) -> Arg:
    return check_arg(_spec(function), Param("column", "C"), value, position)


def _scalar(function: str, value: Any, name: str = "value") -> Node:
    return check_arg(_spec(function), Param(name, "S"), value, 0)  # type: ignore[return-value]


def _pairs(function: str, items: tuple[Any, ...]) -> Iterator[tuple[str, Any]]:
    """Name/expression pairs from ``"Name", expr`` sequences and dicts."""
    i = 0
    while i < len(items):
        item = items[i]
        if isinstance(item, dict):
            for key, value in item.items():
                if not isinstance(key, str):
                    raise TypeError(f"{function}: column names must be text, got {key!r}")
                yield key, value
            i += 1
        elif isinstance(item, str):
            if i + 1 >= len(items):
                raise TypeError(f"{function}: the name {item!r} has no expression after it")
            yield item, items[i + 1]
            i += 2
        else:
            raise TypeError(
                f"{function}: expected a name and an expression (or a dict of them), got {item!r}"
            )


def _named_args(function: str, items: tuple[Any, ...]) -> list[Arg]:
    args: list[Arg] = []
    names: set[str] = set()
    for name, value in _pairs(function, items):
        if name.casefold() in names:
            raise DaxError(f"{function}: the column name {name!r} is used more than once.")
        names.add(name.casefold())
        node = coerce_any(value)
        if node.kind == "modifier":
            raise DaxError(f"{function}: {node.to_dax()} can only be used inside CALCULATE.")
        args += [Literal_(name), node]
    return args


def _orders(function: str, items: tuple[Any, ...]) -> list[Arg]:
    """``expr[, ASC|DESC]`` repeated, as in TOPN, CONCATENATEX and ORDERBY."""
    args: list[Arg] = []
    i = 0
    while i < len(items):
        args.append(_scalar(function, items[i], "orderBy_expression"))
        i += 1
        if i < len(items) and _is_order(items[i]):
            args.append(keyword(items[i], ("ASC", "DESC"), function, "order"))
            i += 1
    return args


def _is_order(value: Any) -> bool:
    if isinstance(value, Keyword):
        return value.name in ("ASC", "DESC")
    return isinstance(value, str) and value.upper() in ("ASC", "DESC")


# --- two-form aggregations -------------------------------------------------------


def MIN(column_or_value1: Any, value2: Any = OMITTED) -> Expr:
    """DAX ``MIN(<column>)`` or ``MIN(<value1>, <value2>)``: the smallest value."""
    return _min_max("MIN", column_or_value1, value2)


def MAX(column_or_value1: Any, value2: Any = OMITTED) -> Expr:
    """DAX ``MAX(<column>)`` or ``MAX(<value1>, <value2>)``: the largest value."""
    return _min_max("MAX", column_or_value1, value2)


def _min_max(name: str, a: Any, b: Any) -> Expr:
    if b is OMITTED:
        col = _column(name, a, 1)
        return Call(name, [col], getattr(col, "dax_type", None))
    x, y = _scalar(name, a), _scalar(name, b)
    same = x.dax_type if x.dax_type is y.dax_type else None
    return Call(name, [x, y], same)


# --- logic -------------------------------------------------------------------------


def SWITCH(expression: Any, *values_and_results: Any) -> Expr:
    """DAX ``SWITCH(<expression>, <value>, <result>[, ...][, <else>])``.

    The result for the first value equal to the expression, or the else value (BLANK if
    there's none). ``SWITCH(TRUE(), cond1, result1, ...)`` tests conditions in order.
    """
    if len(values_and_results) < 2:
        raise TypeError("SWITCH needs at least one value and result after the expression")
    args: list[Arg] = [_scalar("SWITCH", expression, "expression")]
    args += [_scalar("SWITCH", v) for v in values_and_results]
    results = list(args[2::2])
    if len(values_and_results) % 2 == 1:
        results.append(args[-1])
    types = {getattr(r, "dax_type", None) for r in results}
    return Call("SWITCH", args, types.pop() if len(types) == 1 else None)


def IN(value: Any, table: Any) -> Expr:
    """DAX ``<value> IN <table>``, the same as ``value.IN(table)``."""
    return coerce(value).IN(table)


# --- table functions with name/expression pairs -----------------------------------


def ADDCOLUMNS(table: Any, *names_and_expressions: Any) -> TableExpr:
    """DAX ``ADDCOLUMNS(<table>, <name>, <expression>[, ...])``.

    ``ADDCOLUMNS(Product, "Sales", total)`` or ``ADDCOLUMNS(Product, {"Sales": total})``.
    """
    args: list[Arg] = [coerce_table(table, "ADDCOLUMNS")]
    args += _named_args("ADDCOLUMNS", names_and_expressions)
    if len(args) == 1:
        raise TypeError("ADDCOLUMNS needs at least one name and expression")
    return TableCall("ADDCOLUMNS", args)


def SELECTCOLUMNS(table: Any, *columns: Any) -> TableExpr:
    """DAX ``SELECTCOLUMNS(<table>, [<name>], <expression>[, ...])``.

    A bare column keeps its own name: ``SELECTCOLUMNS(Sales, Sales["Amount"])``.
    """
    args: list[Arg] = [coerce_table(table, "SELECTCOLUMNS")]
    rest: list[Any] = []
    for item in columns:
        if is_column(item) and (not rest or not isinstance(rest[-1], str)):
            args.append(item)
        else:
            rest.append(item)
    args += _named_args("SELECTCOLUMNS", tuple(rest))
    if len(args) == 1:
        raise TypeError("SELECTCOLUMNS needs at least one column")
    return TableCall("SELECTCOLUMNS", args)


def ROW(*names_and_expressions: Any) -> TableExpr:
    """DAX ``ROW(<name>, <expression>[, ...])``: a one-row table."""
    args = _named_args("ROW", names_and_expressions)
    if not args:
        raise TypeError("ROW needs at least one name and expression")
    for value in args[1::2]:
        if getattr(value, "kind", None) == "table":
            raise DaxError("ROW: each expression must return a single value, not a table.")
    return TableCall("ROW", args)


def _group_and_named(function: str, items: tuple[Any, ...]) -> tuple[list[Arg], tuple[Any, ...]]:
    cols: list[Arg] = []
    i = 0
    while i < len(items) and is_column(items[i]):
        cols.append(items[i])
        i += 1
    return cols, items[i:]


def SUMMARIZE(table: Any, *groupby_and_named: Any) -> TableExpr:
    """DAX ``SUMMARIZE(<table>, <groupBy_column>[, ...][, <name>, <expression>]...)``."""
    cols, named = _group_and_named("SUMMARIZE", groupby_and_named)
    return TableCall(
        "SUMMARIZE", [coerce_table(table, "SUMMARIZE"), *cols, *_named_args("SUMMARIZE", named)]
    )


def GROUPBY(table: Any, *groupby_and_named: Any) -> TableExpr:
    """DAX ``GROUPBY(<table>, <groupBy_column>...[, <name>, <expression>]...)``.

    Expressions use ``CURRENTGROUP()``, as in ``SUMX(CURRENTGROUP(), Sales["Amount"])``.
    """
    cols, named = _group_and_named("GROUPBY", groupby_and_named)
    return TableCall(
        "GROUPBY", [coerce_table(table, "GROUPBY"), *cols, *_named_args("GROUPBY", named)]
    )


def SUMMARIZECOLUMNS(*args: Any) -> TableExpr:
    """DAX ``SUMMARIZECOLUMNS``: group by columns and add named expressions.

    ``SUMMARIZECOLUMNS(<groupBy_column>...[, <filter_table>]...[, <name>, <expr>]...)``.

    Groups by the columns, applies the filter tables, and adds the named expressions.
    Rows where every expression is BLANK are removed, as in Power BI visuals::

        SUMMARIZECOLUMNS(Product["Color"], {"Total": total, "Share": share})
    """
    group: list[Arg] = []
    filters: list[Arg] = []
    i = 0
    while i < len(args) and is_column(args[i]):
        group.append(args[i])
        i += 1
    while i < len(args) and isinstance(args[i], TableExpr | list):
        filters.append(coerce_table(args[i], "SUMMARIZECOLUMNS"))
        i += 1
    rest = args[i:]
    for item in rest:
        if is_column(item):
            raise DaxError("SUMMARIZECOLUMNS: group-by columns must come before filters and names.")
        if isinstance(item, ModifierExpr):
            raise DaxError(f"SUMMARIZECOLUMNS: {item.to_dax()} can only be used inside CALCULATE.")
    named = _named_args("SUMMARIZECOLUMNS", tuple(rest))
    if not group and not named:
        raise TypeError("SUMMARIZECOLUMNS needs a group-by column or a name and expression")
    for value in named[1::2]:
        if getattr(value, "kind", None) == "table":
            raise DaxError("SUMMARIZECOLUMNS: each expression must return a single value.")
    return TableCall("SUMMARIZECOLUMNS", [*group, *filters, *named])


def TOPN(n_value: Any, table: Any, *order_by: Any) -> TableExpr:
    """DAX ``TOPN(<n_value>, <table>[, <orderBy_expression>[, <order>]]...)``.

    ``TOPN(5, Product, total, DESC)``. The order is ``ASC`` or ``DESC`` (DESC when left
    out, as in DAX).
    """
    args: list[Arg] = [_scalar("TOPN", n_value, "n_value"), coerce_table(table, "TOPN")]
    return TableCall("TOPN", args + _orders("TOPN", order_by))


def CONCATENATEX(table: Any, expression: Any, delimiter: Any = OMITTED, *order_by: Any) -> Expr:
    """DAX ``CONCATENATEX(<table>, <expression>[, <delimiter>[, <orderBy>[, <order>]]...])``."""
    args: list[Arg] = [coerce_table(table, "CONCATENATEX"), _scalar("CONCATENATEX", expression)]
    if delimiter is not OMITTED or order_by:
        args.append(OMITTED if delimiter is OMITTED else _scalar("CONCATENATEX", delimiter))
    return Call("CONCATENATEX", args + _orders("CONCATENATEX", order_by), DaxType.STRING)


def ORDERBY(*expressions_and_orders: Any) -> TableExpr:
    """DAX ``ORDERBY(<expression>[, <order>]...)`` for window functions."""
    if not expressions_and_orders:
        raise TypeError("ORDERBY needs at least one expression")
    return TableCall("ORDERBY", _orders("ORDERBY", expressions_and_orders))


# --- lookups -------------------------------------------------------------------------


def LOOKUPVALUE(result_column: Any, *search_pairs_and_alternate: Any) -> Expr:
    """DAX ``LOOKUPVALUE``: the value of a column in the row matching the search values.

    ``LOOKUPVALUE(<result_column>, <search_column>, <search_value>[, ...][, <alternate>])``.
    """
    items = search_pairs_and_alternate
    if len(items) < 2:
        raise TypeError("LOOKUPVALUE needs at least one search column and value")
    result = _column("LOOKUPVALUE", result_column, 1)
    args: list[Arg] = [result]
    pairs, alternate = (items[:-1], items[-1]) if len(items) % 2 else (items, OMITTED)
    for i in range(0, len(pairs), 2):
        args.append(_column("LOOKUPVALUE", pairs[i], len(args) + 1))
        args.append(_scalar("LOOKUPVALUE", pairs[i + 1], "search_value"))
    if alternate is not OMITTED:
        args.append(_scalar("LOOKUPVALUE", alternate, "alternate_result"))
    return Call("LOOKUPVALUE", args, getattr(result, "dax_type", None))


def CONTAINS(table: Any, *columns_and_values: Any) -> Expr:
    """DAX ``CONTAINS(<table>, <column>, <value>[, <column>, <value>]...)``."""
    items = columns_and_values
    if not items or len(items) % 2:
        raise TypeError("CONTAINS needs column and value pairs after the table")
    args: list[Arg] = [coerce_table(table, "CONTAINS")]
    for i in range(0, len(items), 2):
        args.append(_column("CONTAINS", items[i], len(args) + 1))
        args.append(_scalar("CONTAINS", items[i + 1]))
    return Call("CONTAINS", args, DaxType.BOOLEAN)


# --- inline tables ------------------------------------------------------------------

_DATATABLE_TYPES = {
    "INTEGER": "INTEGER",
    "DOUBLE": "DOUBLE",
    "DECIMAL": "DOUBLE",
    "STRING": "STRING",
    "BOOLEAN": "BOOLEAN",
    "CURRENCY": "CURRENCY",
    "DATETIME": "DATETIME",
}


class _DataTable(TableExpr):
    def __init__(self, columns: list[tuple[str, str]], rows: list[list[Node]]) -> None:
        self.columns = columns
        self.rows = rows

    def children(self) -> list[Node]:
        return [v for r in self.rows for v in r]

    def _dax(self) -> tuple[str, int]:
        cols = ", ".join(f'"{n}", {t}' for n, t in self.columns)
        rows = ", ".join("{" + ", ".join(v.to_dax() for v in r) + "}" for r in self.rows)
        return f"DATATABLE({cols}, {{{rows}}})", 100


def DATATABLE(columns: dict[str, Any], rows: list[list[Any]]) -> TableExpr:
    """DAX ``DATATABLE``: a table written out with its column types and rows.

    ``DATATABLE({"Color": "STRING", "Qty": "INTEGER"}, [["Red", 1], ["Blue", 2]])``.
    Types are INTEGER, DOUBLE, STRING, BOOLEAN, CURRENCY and DATETIME (a DaxType works
    too). Values must be constants, as in DAX.
    """
    cols: list[tuple[str, str]] = []
    for name, dtype in columns.items():
        raw = getattr(dtype, "_dax_name", None) or getattr(dtype, "name", None) or str(dtype)
        if isinstance(dtype, DaxType):
            raw = {"decimal": "DOUBLE", "datetime": "DATETIME"}.get(dtype.value, dtype.value)
        key = str(raw).upper()
        if key not in _DATATABLE_TYPES:
            raise ValueError(f"DATATABLE: unknown type {dtype!r} for column {name!r}")
        cols.append((name, _DATATABLE_TYPES[key]))
    if not cols:
        raise TypeError("DATATABLE needs at least one column")
    out: list[list[Node]] = []
    for row in rows:
        if len(row) != len(cols):
            raise ValueError(f"DATATABLE: row {row!r} needs {len(cols)} values")
        values = [coerce(v) for v in row]
        if any(not isinstance(v, Literal_) for v in values):
            raise DaxError("DATATABLE: values must be constants.")
        out.append(values)  # type: ignore[arg-type]
    return _DataTable(cols, out)
