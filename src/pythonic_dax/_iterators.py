"""Iterators (Phase 5): SUMX and friends, RANKX, CONCATENATEX, EARLIER and EARLIEST.

An iterator evaluates an expression once per row of a table, each time with a row
context on that row added to the current context.
"""

from __future__ import annotations

import functools
import itertools
from collections.abc import Callable
from dataclasses import replace
from decimal import Decimal
from typing import Any

from . import _scalar as sc
from ._evaluate import Context, Evaluator, FrameRow, Value, convert, infer_type
from .errors import DaxError, NotSupportedError
from .expr import Call, Keyword, _Omitted, is_column

Handler = Callable[[Evaluator, list[Any], Context], Value]


def iterate(ev: Evaluator, table: Any, expr: Any, ctx: Context) -> list[Value]:
    """The expression's value for each row of the table."""
    value = ev.table(table, ctx)
    return [ev.scalar(expr, replace(ctx, rows=(*ctx.rows, row))) for row in ev.rows_of(value)]


def _no_dates_or_booleans(name: str, values: list[Value]) -> None:
    for v in values:
        if isinstance(v, bool) or not isinstance(v, int | float | Decimal):
            raise NotSupportedError(f"{name} of {sc.type_label(sc.type_of(v))} values")


def _sumx(ev: Evaluator, args: list[Any], ctx: Context) -> Value:
    values = [v for v in iterate(ev, args[0], args[1], ctx) if v is not None]
    if not values:
        return None
    _no_dates_or_booleans("SUMX", values)
    total = functools.reduce(lambda a, b: sc.arithmetic("+", a, b), values)
    return convert(total, infer_type(Call("SUMX", args)))


def _averagex(ev: Evaluator, args: list[Any], ctx: Context) -> Value:
    values = [v for v in iterate(ev, args[0], args[1], ctx) if v is not None]
    if not values:
        return None
    _no_dates_or_booleans("AVERAGEX", values)
    if any(isinstance(v, Decimal) for v in values):
        raise NotSupportedError("AVERAGEX of Currency values")
    if all(isinstance(v, int) for v in values):
        return sum(values) / len(values)
    return float(sum(float(v) for v in values) / len(values))


def _min_maxx(name: str) -> Handler:
    def handler(ev: Evaluator, args: list[Any], ctx: Context) -> Value:
        values = [v for v in iterate(ev, args[0], args[1], ctx) if v is not None]
        if not values:
            return None
        kinds = {sc._compare_kind(v) for v in values}
        if len(kinds) > 1 or "bool" in kinds:
            raise NotSupportedError(f"{name} of mixed or True/False values")
        if kinds == {"text"}:
            if not all(sc._ORDERABLE_TEXT.match(v) for v in values):
                raise NotSupportedError(
                    f"{name} of text that isn't plain letters, digits and spaces"
                )
            folded = {v.lower() for v in values}
            if len(folded) != len(set(values)):
                raise NotSupportedError(f"{name} of text values that differ only by case")
            ordered = sorted(set(values), key=str.lower)
            return ordered[0] if name == "MINX" else ordered[-1]
        result = min(values) if name == "MINX" else max(values)
        return convert(result, infer_type(Call(name, args)))

    return handler


def _countx(name: str) -> Handler:
    def handler(ev: Evaluator, args: list[Any], ctx: Context) -> Value:
        values = iterate(ev, args[0], args[1], ctx)
        if any(v == "" for v in values if isinstance(v, str)):
            raise NotSupportedError(f"{name} over empty text values")
        return sum(v is not None for v in values) or None

    return handler


def _productx(ev: Evaluator, args: list[Any], ctx: Context) -> Value:
    values = [v for v in iterate(ev, args[0], args[1], ctx) if v is not None]
    if not values:
        return None
    _no_dates_or_booleans("PRODUCTX", values)
    if any(isinstance(v, Decimal) for v in values):
        raise NotSupportedError("PRODUCTX of Currency values")
    product = 1.0
    for v in values:
        product *= float(v)
    return product


def _sort_key(values: list[Value], descending: list[bool]) -> Callable[[Any], Any]:
    """A sort key for rows given as (keys, ...) tuples, using DAX comparisons."""

    def cmp(a: tuple[list[Value], Any], b: tuple[list[Value], Any]) -> int:
        for x, y, desc in zip(a[0], b[0], descending, strict=True):
            if sc.compare("<", x, y):
                return 1 if desc else -1
            if sc.compare(">", x, y):
                return -1 if desc else 1
        return 0

    return functools.cmp_to_key(cmp)


def order_args(items: list[Any], default: str) -> list[tuple[Any, bool]]:
    """``expr[, ASC|DESC]...`` as (expression, descending) pairs."""
    result: list[tuple[Any, bool]] = []
    for item in items:
        if isinstance(item, _Omitted):
            continue
        if isinstance(item, Keyword):
            result[-1] = (result[-1][0], item.name == "DESC")
        else:
            result.append((item, default == "DESC"))
    return result


def sorted_rows(
    ev: Evaluator, rows: list[Any], orders: list[tuple[Any, bool]], ctx: Context
) -> list[tuple[list[Value], Any]]:
    """Rows with their sort keys, sorted the way DAX orders them."""
    keyed = [
        ([ev.scalar(e, replace(ctx, rows=(*ctx.rows, row))) for e, _ in orders], row)
        for row in rows
    ]
    descending = [d for _, d in orders]
    return sorted(keyed, key=_sort_key([], descending))


def same_keys(a: list[Value], b: list[Value]) -> bool:
    return all(sc.compare("=", x, y) for x, y in zip(a, b, strict=True))


def _concatenatex(ev: Evaluator, args: list[Any], ctx: Context) -> Value:
    table, expr = args[0], args[1]
    delimiter = ""
    if len(args) > 2 and not isinstance(args[2], _Omitted):
        delimiter = sc._text(ev.scalar(args[2], ctx))
    rows = ev.rows_of(ev.table(table, ctx))
    texts = [sc._text(ev.scalar(expr, replace(ctx, rows=(*ctx.rows, r)))) for r in rows]
    orders = order_args(args[3:], "ASC")
    if not orders:
        if len(set(texts)) > 1:
            raise NotSupportedError("CONCATENATEX without an order (Power BI's order is internal)")
        return delimiter.join(texts) or None
    keyed = sorted_rows(ev, rows, orders, ctx)
    text_of = {id(r): t for r, t in zip(rows, texts, strict=True)}
    ordered = [text_of[id(r)] for _, r in keyed]
    for (ka, ra), (kb, rb) in itertools.pairwise(keyed):
        if same_keys(ka, kb) and text_of[id(ra)] != text_of[id(rb)]:
            raise NotSupportedError("CONCATENATEX with ties in its order")
    return delimiter.join(ordered) or None


def _rankx(ev: Evaluator, args: list[Any], ctx: Context) -> Value:
    table, expr = args[0], args[1]
    value_node = args[2] if len(args) > 2 and not isinstance(args[2], _Omitted) else expr
    order = args[3].name if len(args) > 3 and isinstance(args[3], Keyword) else "DESC"
    ties = args[4].name if len(args) > 4 and isinstance(args[4], Keyword) else "SKIP"
    values = iterate(ev, table, expr, ctx)
    value = ev.scalar(value_node, ctx)
    for v in [*values, value]:
        if v is not None and (isinstance(v, bool) or not isinstance(v, int | float | Decimal)):
            raise NotSupportedError(f"RANKX of {sc.type_label(sc.type_of(v))} values")
    op = ">" if order == "DESC" else "<"
    better = [v for v in values if sc.compare(op, v, value)]
    if ties == "SKIP":
        return len(better) + 1
    distinct: list[Value] = []
    for v in better:
        if not any(sc.compare("=", v, d) for d in distinct):
            distinct.append(v)
    return len(distinct) + 1


def _earlier(ev: Evaluator, args: list[Any], ctx: Context) -> Value:
    column = args[0]
    n = 1
    if len(args) > 1 and not isinstance(args[1], _Omitted):
        number = ev.scalar(args[1], ctx)
        if not isinstance(number, int) or isinstance(number, bool) or number < 1:
            raise NotSupportedError("EARLIER with a number that isn't a positive whole number")
        n = number
    holders = [i for i, row in enumerate(ctx.rows) if _has(ev, row, column)]
    if len(holders) <= n:
        raise DaxError("EARLIER/EARLIEST refer to an earlier row context which doesn't exist.")
    index = holders[-1 - n]
    return ev.scalar(column, replace(ctx, rows=ctx.rows[: index + 1]))


def _earliest(ev: Evaluator, args: list[Any], ctx: Context) -> Value:
    column = args[0]
    holders = [i for i, row in enumerate(ctx.rows) if _has(ev, row, column)]
    if len(holders) < 2:
        raise DaxError("EARLIER/EARLIEST refer to an earlier row context which doesn't exist.")
    return ev.scalar(column, replace(ctx, rows=ctx.rows[: holders[0] + 1]))


def _has(ev: Evaluator, row: Any, column: Any) -> bool:
    if not is_column(column):
        return False
    if isinstance(row, FrameRow):
        return row.frame.find(column) is not None
    if row.table is not column.table:
        return False
    return row.columns is None or any(c is column for c in row.columns)


SCALAR: dict[str, Handler] = {
    "SUMX": _sumx,
    "AVERAGEX": _averagex,
    "MINX": _min_maxx("MINX"),
    "MAXX": _min_maxx("MAXX"),
    "COUNTX": _countx("COUNTX"),
    "COUNTAX": _countx("COUNTAX"),
    "PRODUCTX": _productx,
    "CONCATENATEX": _concatenatex,
    "RANKX": _rankx,
    "EARLIER": _earlier,
    "EARLIEST": _earliest,
}
