"""Building function-call expressions, with the argument checks DAX itself makes.

Errors here match what Power BI reports when a formula is written. For example, a
non-column argument to SUM, or columns from two tables in a CALCULATE filter.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ._specs import SPECS, Param, Spec
from .errors import DaxError
from .expr import (
    OMITTED,
    Arg,
    Call,
    GroupingCall,
    Keyword,
    Node,
    TableConstructor,
    _ExprBase,
    _Omitted,
    coerce_any,
    coerce_table,
    is_column,
    is_measure,
    keyword,
)
from .types import DaxType

# Functions whose arguments aren't evaluated in the row context of a CALCULATE
# boolean filter: aggregations, iterators, table functions and CALCULATE itself.
_AGGREGATING = {
    name for name, s in SPECS.items() if s.params and s.params[0].kind in ("T", "C", "TC", "D")
} | {
    "CALCULATE", "CALCULATETABLE", "MIN", "MAX", "CONCATENATEX", "LOOKUPVALUE", "CONTAINS",
    "TOPN", "SUMMARIZE", "SUMMARIZECOLUMNS", "ADDCOLUMNS", "SELECTCOLUMNS", "GROUPBY", "ROW",
    "DATATABLE",
}  # fmt: skip

# Functions whose table argument must name a model table, not a table expression.
_TABLE_NAME_ONLY = {
    "ALL", "ALLEXCEPT", "ALLNOBLANKROW", "ALLSELECTED", "REMOVEFILTERS", "RELATEDTABLE",
    "VALUES", "ISFILTERED", "ISCROSSFILTERED",
}  # fmt: skip

# Functions whose third argument, when it is text, is the year end date.
_YEAR_END_THIRD = {"TOTALYTD", "OPENINGBALANCEYEAR", "CLOSINGBALANCEYEAR"}

_SAME_TABLE_COLUMNS = {"ALL", "ALLNOBLANKROW", "ALLSELECTED", "REMOVEFILTERS"}


def build(name: str, values: Sequence[Any]) -> Node:
    """Build a call to a generic DAX function from its bound argument values.

    ``values`` has one entry per parameter of the spec, in order. A repeating
    parameter's entry is a sequence. A missing optional argument is ``OMITTED``.
    """
    spec = SPECS[name]
    args: list[Arg] = []
    for param, value in zip(spec.params, values, strict=True):
        if param.repeat:
            items = list(value)
            if param.repeat == "+" and not items:
                raise TypeError(f"{name} needs at least one {param.name}")
            for item in items:
                args.append(check_arg(spec, param, item, len(args) + 1))
        elif value is OMITTED:
            if not param.optional:
                raise TypeError(f"{name} is missing its {param.name} argument")
            args.append(OMITTED)
        else:
            args.append(check_arg(spec, param, value, len(args) + 1))

    _validate(spec, args)
    _fill_blanks(spec, args)
    if name in _YEAR_END_THIRD and len(args) == 4 and args[2] is OMITTED and args[3] is not OMITTED:
        # DAX can't skip the filter; a text constant in its place is the year end date.
        args = [args[0], args[1], args[3]]

    kind = {"S": "scalar", "T": "table", "M": "modifier"}[spec.returns]
    if name == "KEEPFILTERS" and getattr(args[0], "kind", None) == "table":
        kind = "table"  # KEEPFILTERS(<table>) can also wrap an iterator's table
    if name == "ALL" and not args:
        kind = "modifier"  # ALL() with no arguments only works inside CALCULATE
    return Call.raw(name, args, kind, _result_type(spec, args))  # type: ignore[arg-type]


def check_arg(spec: Spec, param: Param, value: Any, position: int) -> Arg:
    """Check one argument against its parameter kind and convert it to a node."""
    name = spec.name
    kind = param.kind

    if kind == "K":
        return keyword(value, param.keywords, name, param.name)
    if value is OMITTED:
        omitted: Arg = OMITTED
        return omitted

    if kind == "C":
        if not is_column(value):
            raise DaxError(
                f"The {name} function only accepts a column reference as argument number "
                f"{position}."
            )
        if spec.numeric_column and value.data_type in (DaxType.STRING, DaxType.BOOLEAN):
            label = "String" if value.data_type is DaxType.STRING else "Boolean"
            raise DaxError(f"The function {name} cannot work with values of type {label}.")
        column: Arg = value
        return column

    if kind == "T":
        table = coerce_table(value, name)
        if name in _TABLE_NAME_ONLY and not getattr(table, "is_table_ref", False):
            raise DaxError(f"The {name} function needs a table name, not a table expression.")
        return table

    if kind in ("TC", "D"):
        if is_column(value):
            return value  # type: ignore[no-any-return]
        node = coerce_any(value)
        if node.kind != "table":
            what = "a table or a column" if kind == "TC" else "a date column or a table of dates"
            raise TypeError(f"{name}: {param.name} must be {what}, got {node.to_dax()}")
        if kind == "TC" and name in _TABLE_NAME_ONLY and not getattr(node, "is_table_ref", False):
            raise DaxError(
                f"The {name} function needs a table name or a column, not a table expression."
            )
        return node

    node = coerce_any(value)
    if isinstance(node, GroupingCall):
        raise DaxError(
            f"{node.name} can only be used among the group-by columns of SUMMARIZECOLUMNS."
        )
    if kind == "F":
        if node.kind == "scalar":
            check_boolean_filter(name, node)
        return node

    # S: a scalar. A table is allowed too, because DAX converts a one-row, one-column
    # table to its value when the formula runs.
    if node.kind == "modifier":
        raise DaxError(
            f"{node.to_dax()} can only be used as a filter argument of CALCULATE or "
            f"CALCULATETABLE, not as the {param.name} of {name}."
        )
    return node


def _fill_blanks(spec: Spec, args: list[Arg]) -> None:
    """Window functions don't accept an empty ``blanks`` argument in the middle of the
    list (Power BI requires it), so write the default, DEFAULT, there instead."""
    names = [p.name for p in spec.params]
    if "blanks" not in names or "partitionBy" not in names:
        return
    i = names.index("blanks")
    if args[i] is OMITTED and any(a is not OMITTED for a in args[i + 1 :]):
        args[i] = Keyword("DEFAULT")


def row_context_columns(node: Node) -> list[Any]:
    """Columns a boolean filter reads from its own rows (not inside aggregations)."""
    found: list[Any] = []

    def visit(n: Any) -> None:
        if is_column(n):
            found.append(n)
            return
        if is_measure(n) or getattr(n, "kind", None) == "table":
            return
        if getattr(n, "name", None) in _AGGREGATING and hasattr(n, "args"):
            return
        if isinstance(n, _ExprBase):
            for child in n.children():
                visit(child)

    visit(node)
    return found


def check_boolean_filter(function: str, node: Node) -> None:
    """A True/False filter argument must read columns of exactly one table."""
    tables = {id(c.table): c.table for c in row_context_columns(node)}
    if len(tables) > 1:
        raise DaxError(
            "The expression contains columns from multiple tables, but only columns from a "
            "single table can be used in a True/False expression that is used as a table "
            f"filter expression. (In {function}: {node.to_dax()})"
        )
    if not tables:
        raise DaxError(
            "A True/False expression used as a table filter expression must refer to a column. "
            f"(In {function}: {node.to_dax()})"
        )


def _validate(spec: Spec, args: list[Arg]) -> None:
    name = spec.name
    given = [a for a in args if not isinstance(a, _Omitted | Keyword)]

    if name in _SAME_TABLE_COLUMNS and given:
        tables = [a for a in given if not is_column(a)]
        if tables and len(given) > 1:
            raise DaxError(f"{name} accepts one table, or columns of one table, but not both.")
        owners = {id(a.table) for a in given if is_column(a)}  # type: ignore[union-attr]
        if len(owners) > 1:
            raise DaxError(
                f"All column arguments of the {name} function must be from the same table."
            )

    if name in ("USERELATIONSHIP", "CROSSFILTER"):
        a, b = args[0], args[1]
        model = a.table.model  # type: ignore[union-attr]
        found = any(
            {id(r.from_column), id(r.to_column)} == {id(a), id(b)} for r in model.relationships
        )
        if not found:
            raise DaxError(
                f"{name} function can only use the two columns references participating in "
                "relationship."
            )

    if name == "TREATAS":
        table = args[0]
        if isinstance(table, TableConstructor) and table.rows:
            width = len(table.rows[0])
            if width != len(args) - 1:
                raise DaxError(
                    f"The number of columns in the table ({width}) must match the number of "
                    f"column references in TREATAS ({len(args) - 1})."
                )


def _result_type(spec: Spec, args: list[Arg]) -> DaxType | None:
    rule = spec.type_rule
    if isinstance(rule, DaxType):
        return rule
    if isinstance(rule, str) and rule.startswith("arg"):
        index = int(rule[3:])
        if index < len(args):
            arg = args[index]
            return getattr(arg, "dax_type", None)
    return None
