"""Table functions (Phase 5): FILTER, ADDCOLUMNS, SELECTCOLUMNS, TOPN, set operations,
joins, SUMMARIZE, GROUPBY, GENERATESERIES and DATATABLE.

Tables come back as a :class:`RowSet` (some rows of a model table, which keeps Power
BI's expanded-table behavior when used as a filter) or a :class:`Frame` (computed rows,
whose columns remember the model column they came from).
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from dataclasses import replace
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import polars as pl

from . import _scalar as sc
from ._evaluate import (
    BLANK_TYPE,
    Context,
    Evaluator,
    Frame,
    FrameRow,
    ResultColumn,
    RowRef,
    RowSet,
    TableValue,
    Value,
    _type_of,
    _vector_condition,
    convert,
    infer_type,
    unify,
)
from ._iterators import order_args, same_keys, sorted_rows
from .errors import DaxError, NotSupportedError
from .expr import Call, Literal_, _Omitted, is_column
from .types import DaxType

if TYPE_CHECKING:
    from .objects import Column

TableHandler = Callable[[Evaluator, list[Any], Context], TableValue]


def _with_row(ctx: Context, row: Any) -> Context:
    return replace(ctx, rows=(*ctx.rows, row))


def _check_name(frame_columns: list[ResultColumn], name: str) -> None:
    wanted = name.lower()
    for c in frame_columns:
        short = c.name[c.name.index("[") + 1 : -1] if "[" in c.name else c.name
        if short.lower() == wanted:
            raise DaxError(f"The column '{name}' already exists in the table.")


def _result_type(expr: Any, values: list[Value]) -> Any:
    t = infer_type(expr)
    if isinstance(t, DaxType) or t == BLANK_TYPE:
        return t
    known = [v for v in values if v is not None]
    return _type_of(known) if known else None


# --- FILTER ------------------------------------------------------------------------


def _filter(ev: Evaluator, args: list[Any], ctx: Context) -> TableValue:
    value = ev.table(args[0], ctx)
    cond = args[1]
    if isinstance(value, RowSet):
        mask = ev._rowset_mask(value)
        present = ev.storage.present_columns(value.table)
        ext = ev.storage.ext(value.table)
        indices = mask.arg_true()
        hits = _vector_condition(cond, ext[indices.to_list()], present) if present else None
        if hits is None:
            keep = [
                sc.to_bool(ev.scalar(cond, _with_row(ctx, RowRef(value.table, int(i)))))
                for i in indices.to_list()
            ]
            hits = pl.Series(keep, dtype=pl.Boolean)
        chosen = indices.filter(hits)
        new = pl.Series([False] * ext.height, dtype=pl.Boolean).scatter(chosen, True)
        return RowSet(value.table, new, value.blank)
    rows = [
        r
        for i, r in enumerate(value.rows)
        if sc.to_bool(ev.scalar(cond, _with_row(ctx, FrameRow(value, i))))
    ]
    return Frame(value.columns, rows, value.filterable)


# --- adding and selecting columns -------------------------------------------------


def _named_pairs(args: list[Any]) -> list[tuple[str, Any]]:
    return [(n.value, e) for n, e in zip(args[::2], args[1::2], strict=True)]


def _addcolumns(ev: Evaluator, args: list[Any], ctx: Context) -> TableValue:
    value = ev.table(args[0], ctx)
    base = ev.to_frame(value)
    rows = ev.rows_of(value)
    named = _named_pairs(args[1:])
    columns = list(base.columns)
    for name, _ in named:
        _check_name(columns, name)
        columns.append(ResultColumn(f"[{name}]", None))
    computed = [[ev.scalar(e, _with_row(ctx, r)) for _, e in named] for r in rows]
    for j, (_, expr) in enumerate(named):
        t = _result_type(expr, [row[j] for row in computed])
        columns[len(base.columns) + j] = ResultColumn(f"[{named[j][0]}]", t)
        for row in computed:
            row[j] = convert(row[j], t)
    out = [list(b) + c for b, c in zip(base.rows, computed, strict=True)]
    return Frame(columns, out, base.filterable)


def _selectcolumns(ev: Evaluator, args: list[Any], ctx: Context) -> TableValue:
    value = ev.table(args[0], ctx)
    rows = ev.rows_of(value)
    items: list[tuple[str, Any]] = []
    headers: list[str] = []
    rest = args[1:]
    i = 0
    while i < len(rest):
        if is_column(rest[i]):  # a bare column keeps its full name, Table[Column]
            items.append((rest[i].name, rest[i]))
            headers.append(f"{rest[i].table.name}[{rest[i].name}]")
            i += 1
        else:
            items.append((rest[i].value, rest[i + 1]))
            headers.append(f"[{rest[i].value}]")
            i += 2
    columns: list[ResultColumn] = []
    for (name, expr), header in zip(items, headers, strict=True):
        _check_name(columns, name)
        lineage = expr if is_column(expr) else None
        columns.append(ResultColumn(header, None, lineage))
    computed = [[ev.scalar(e, _with_row(ctx, r)) for _, e in items] for r in rows]
    for j, (_, expr) in enumerate(items):
        t = _result_type(expr, [row[j] for row in computed])
        columns[j] = replace(columns[j], dax_type=t)
        for row in computed:
            row[j] = convert(row[j], t)
    return Frame(columns, computed)


# --- constant tables ------------------------------------------------------------------

_DATATABLE_TYPES = {
    "INTEGER": DaxType.INTEGER,
    "DOUBLE": DaxType.DECIMAL,
    "STRING": DaxType.STRING,
    "BOOLEAN": DaxType.BOOLEAN,
    "CURRENCY": DaxType.CURRENCY,
    "DATETIME": DaxType.DATETIME,
}


def datatable(ev: Evaluator, node: Any, ctx: Context) -> Frame:
    columns = [ResultColumn(f"[{n}]", _DATATABLE_TYPES[t]) for n, t in node.columns]
    rows = []
    for row in node.rows:
        out = []
        for col, literal in zip(columns, row, strict=True):
            out.append(_datatable_value(col.dax_type, literal))
        rows.append(out)
    return Frame(columns, rows)


def _datatable_value(dax_type: Any, literal: Literal_) -> Value:
    v = literal.value
    if v is None:
        return None
    if dax_type is DaxType.INTEGER and isinstance(v, int) and not isinstance(v, bool):
        return v
    if dax_type is DaxType.DECIMAL and isinstance(v, int | float) and not isinstance(v, bool):
        return float(v)
    if dax_type is DaxType.CURRENCY and isinstance(v, int | float | Decimal):
        return sc.to_currency(v)
    if dax_type is DaxType.STRING and isinstance(v, str):
        return v
    if dax_type is DaxType.BOOLEAN and isinstance(v, bool):
        return v
    if dax_type is DaxType.DATETIME and isinstance(v, dt.date):
        return v if isinstance(v, dt.datetime) else dt.datetime(v.year, v.month, v.day)
    raise NotSupportedError(f"DATATABLE value {v!r} in a {dax_type} column")


def _generateseries(ev: Evaluator, args: list[Any], ctx: Context) -> TableValue:
    start = ev.scalar(args[0], ctx)
    end = ev.scalar(args[1], ctx)
    step = 1
    if len(args) > 2 and not isinstance(args[2], _Omitted):
        step = ev.scalar(args[2], ctx)
    for v in (start, end, step):
        if not isinstance(v, int) or isinstance(v, bool):
            raise NotSupportedError("GENERATESERIES with values that aren't whole numbers")
    if step <= 0:
        raise DaxError("The increment value must be greater than zero.")
    values = list(range(start, end + 1, step)) if end >= start else []
    return Frame([ResultColumn("[Value]", DaxType.INTEGER)], [[v] for v in values])


# --- TOPN ----------------------------------------------------------------------------


def _topn(ev: Evaluator, args: list[Any], ctx: Context) -> TableValue:
    n = ev.scalar(args[0], ctx)
    if n is not None and (not isinstance(n, int) or isinstance(n, bool)):
        raise NotSupportedError("TOPN with a count that isn't a whole number")
    value = ev.table(args[1], ctx)
    rows = ev.rows_of(value)
    count = n or 0
    orders = order_args(args[2:], "DESC")
    if count <= 0:
        return _subset(ev, value, [])
    if count >= len(rows):
        return value
    if not orders:
        raise NotSupportedError("TOPN without an order, when it keeps only some rows")
    keyed = sorted_rows(ev, rows, orders, ctx)
    last = keyed[count - 1][0]
    kept = [r for k, r in keyed[:count]]
    kept += [r for k, r in keyed[count:] if same_keys(k, last)]  # ties are all kept
    return _subset(ev, value, kept)


def _subset(ev: Evaluator, value: TableValue, rows: list[Any]) -> TableValue:
    if isinstance(value, RowSet):
        height = ev.storage.ext(value.table).height
        mask = pl.Series([False] * height, dtype=pl.Boolean)
        if rows:
            mask = mask.scatter([r.index for r in rows], True)
        return RowSet(value.table, mask, value.blank)
    wanted = sorted(r.index for r in rows)
    return Frame(value.columns, [value.rows[i] for i in wanted], value.filterable)


# --- combining tables -------------------------------------------------------------


def _crossjoin(ev: Evaluator, args: list[Any], ctx: Context) -> TableValue:
    frames = [ev.to_frame(ev.table(a, ctx)) for a in args]
    columns: list[ResultColumn] = []
    for f in frames:
        for c in f.columns:
            if any(c.name.lower() == d.name.lower() for d in columns):
                raise DaxError(f"The column {c.name} appears in more than one table of CROSSJOIN.")
            columns.append(c)
    rows: list[list[Value]] = [[]]
    for f in frames:
        rows = [a + list(b) for a in rows for b in f.rows]
    return Frame(columns, rows)


def _generate(keep_empty: bool) -> TableHandler:
    def handler(ev: Evaluator, args: list[Any], ctx: Context) -> TableValue:
        value = ev.table(args[0], ctx)
        base = ev.to_frame(value)
        right_columns: list[ResultColumn] | None = None
        out: list[list[Value]] = []
        for left, row in zip(base.rows, ev.rows_of(value), strict=True):
            right = ev.to_frame(ev.table(args[1], _with_row(ctx, row)))
            if right_columns is None:
                right_columns = right.columns
            elif [c.name for c in right.columns] != [c.name for c in right_columns]:
                raise NotSupportedError("GENERATE whose second table changes shape")
            if right.rows:
                out += [list(left) + list(r) for r in right.rows]
            elif keep_empty:
                out.append(list(left) + [None] * len(right.columns))
        if right_columns is None:
            right = ev.to_frame(ev.table(args[1], ctx))
            right_columns = right.columns
        for c in right_columns:
            if any(c.name.lower() == d.name.lower() for d in base.columns):
                raise DaxError(f"The column {c.name} appears in both tables of GENERATE.")
        return Frame([*base.columns, *right_columns], out)

    return handler


def _same_shape(frames: list[Frame], function: str) -> None:
    widths = {len(f.columns) for f in frames}
    if len(widths) > 1:
        raise DaxError(f"Each table argument of '{function}' must have the same number of columns.")


def _keys(frame: Frame) -> list[tuple[Any, ...]]:
    def norm(v: Value) -> Any:
        if isinstance(v, str):
            return ("text", v.lower())
        if isinstance(v, bool):
            return ("bool", v)
        if isinstance(v, int | float | Decimal):
            return ("number", float(v))
        return ("other", v)

    return [tuple(norm(v) for v in row) for row in frame.rows]


def _check_case(frames: list[Frame], function: str) -> None:
    for f in frames:
        for i in range(len(f.columns)):
            texts = {r[i] for r in f.rows if isinstance(r[i], str)}
            if len({t.lower() for t in texts}) != len(texts):
                raise NotSupportedError(f"{function} with text values that differ only by case")


def _union(ev: Evaluator, args: list[Any], ctx: Context) -> TableValue:
    frames = [ev.to_frame(ev.table(a, ctx)) for a in args]
    _same_shape(frames, "UNION")
    columns = []
    for i, first in enumerate(frames[0].columns):
        types = [f.columns[i].dax_type for f in frames]
        t = unify(types)
        if t is None and any(x is not None for x in types):
            raise NotSupportedError("UNION of columns with different types")
        lineages = {id(f.columns[i].column) for f in frames}
        lineage = first.column if len(lineages) == 1 else None
        columns.append(ResultColumn(first.name, t, lineage))
    rows = [
        [convert(v, c.dax_type) for v, c in zip(r, columns, strict=True)]
        for f in frames
        for r in f.rows
    ]
    return Frame(columns, rows)


def _intersect_except(name: str) -> TableHandler:
    def handler(ev: Evaluator, args: list[Any], ctx: Context) -> TableValue:
        left = ev.to_frame(ev.table(args[0], ctx))
        right = ev.to_frame(ev.table(args[1], ctx))
        _same_shape([left, right], name)
        _check_case([left, right], name)
        found = set(_keys(right))
        keep = name == "INTERSECT"
        rows = [r for r, k in zip(left.rows, _keys(left), strict=True) if (k in found) == keep]
        return Frame(left.columns, rows, left.filterable)

    return handler


def _natural_join(outer: bool) -> TableHandler:
    def handler(ev: Evaluator, args: list[Any], ctx: Context) -> TableValue:
        left = ev.to_frame(ev.table(args[0], ctx))
        right = ev.to_frame(ev.table(args[1], ctx))
        common = [
            (i, j)
            for i, a in enumerate(left.columns)
            for j, b in enumerate(right.columns)
            if a.column is not None and a.column is b.column
        ]
        if not common:
            raise NotSupportedError("NATURAL joins without columns from the same model column")
        joined = {j for _, j in common}
        extra = [j for j in range(len(right.columns)) if j not in joined]
        for j in extra:
            if any(right.columns[j].name.lower() == c.name.lower() for c in left.columns):
                raise DaxError(f"The column {right.columns[j].name} appears in both tables.")
        _check_case([left, right], "NATURALINNERJOIN")
        index: dict[tuple[Any, ...], list[list[Value]]] = {}
        right_keys = _keys(Frame(right.columns, [[r[j] for _, j in common] for r in right.rows]))
        for r, k in zip(right.rows, right_keys, strict=True):
            if any(v[1] is None for v in k):
                continue  # BLANK keys never match
            index.setdefault(k, []).append(r)
        left_keys = _keys(Frame(left.columns, [[r[i] for i, _ in common] for r in left.rows]))
        out = []
        for r, k in zip(left.rows, left_keys, strict=True):
            matches = index.get(k, [])
            for m in matches:
                out.append(list(r) + [m[j] for j in extra])
            if not matches and outer:
                out.append(list(r) + [None] * len(extra))
        return Frame([*left.columns, *(right.columns[j] for j in extra)], out)

    return handler


# --- grouping ------------------------------------------------------------------------


def _group_values(ev: Evaluator, value: TableValue, columns: list[Column]) -> list[list[Value]]:
    """For each row of the table, the values of the group-by columns. Columns of
    tables on the one side of a relationship are looked up like RELATED."""
    from ._evaluate import _lookup

    if isinstance(value, Frame):
        positions = []
        for c in columns:
            i = value.find(c)
            if i is None:
                raise NotSupportedError(f"grouping a computed table by {c.ref}, which it lacks")
            positions.append(i)
        return [[row[i] for i in positions] for row in value.rows]
    out = []
    for c in columns:
        ev.storage.check_text(c)
    paths = {
        id(c): ([] if c.table is value.table else ev.related_path(value.table, c.table))
        for c in columns
    }
    for row in ev.rows_of(value):
        assert isinstance(row, RowRef)
        values = []
        for c in columns:
            index: int | None = row.index
            table = value.table
            for here, there in paths[id(c)]:
                if index is None:
                    break
                key = ev.storage.ext(table)[here.name][index]
                index = _lookup(ev, there, key)
                table = there.table
            values.append(None if index is None else ev.storage.ext(c.table)[c.name][index])
        out.append(values)
    return out


def _distinct_groups(values: list[list[Value]]) -> list[tuple[list[Value], list[int]]]:
    groups: dict[tuple[Any, ...], tuple[list[Value], list[int]]] = {}
    for i, row in enumerate(values):
        key = tuple(v.lower() if isinstance(v, str) else v for v in row)
        if key not in groups:
            groups[key] = (row, [])
        groups[key][1].append(i)
    return list(groups.values())


def _summarize(ev: Evaluator, args: list[Any], ctx: Context) -> TableValue:
    value = ev.table(args[0], ctx)
    columns = []
    i = 1
    while i < len(args) and is_column(args[i]):
        columns.append(args[i])
        i += 1
    if i < len(args):
        raise NotSupportedError(
            "SUMMARIZE with named expressions (use ADDCOLUMNS or SUMMARIZECOLUMNS)"
        )
    groups = _distinct_groups(_group_values(ev, value, columns))
    result = [ResultColumn(f"{c.table.name}[{c.name}]", c.data_type, c) for c in columns]
    return Frame(result, [list(g) for g, _ in groups])


def _groupby(ev: Evaluator, args: list[Any], ctx: Context) -> TableValue:
    value = ev.table(args[0], ctx)
    columns = []
    i = 1
    while i < len(args) and is_column(args[i]):
        columns.append(args[i])
        i += 1
    named = _named_pairs(args[i:])
    groups = _distinct_groups(_group_values(ev, value, columns))
    rows_all = ev.rows_of(value)
    result = [ResultColumn(f"{c.table.name}[{c.name}]", c.data_type, c) for c in columns]
    out: list[list[Value]] = []
    for keys, members in groups:
        group = _subset(ev, value, [rows_all[m] for m in members])
        inner = replace(ctx, group=group)
        out.append(list(keys) + [ev.scalar(e, inner) for _, e in named])
    for j, (name, expr) in enumerate(named):
        t = _result_type(expr, [row[len(columns) + j] for row in out])
        result.append(ResultColumn(f"[{name}]", t))
        for row in out:
            row[len(columns) + j] = convert(row[len(columns) + j], t)
    return Frame(result, out)


def _currentgroup(ev: Evaluator, args: list[Any], ctx: Context) -> TableValue:
    if ctx.group is None:
        raise DaxError("Function CURRENTGROUP can only be used inside GROUPBY.")
    return ctx.group


def _distinct_table(ev: Evaluator, value: TableValue) -> Frame:
    frame = ev.to_frame(value)
    _check_case([frame], "DISTINCT")
    seen: set[tuple[Any, ...]] = set()
    rows = []
    for row, key in zip(frame.rows, _keys(frame), strict=True):
        if key not in seen:
            seen.add(key)
            rows.append(row)
    return Frame(frame.columns, rows, filterable=False)


def _allselected(ev: Evaluator, args: list[Any], ctx: Context) -> TableValue:
    """ALLSELECTED as a table: the rows visible once the filters on its columns are put
    back to the last shadow filter context (the filters from outside SUMMARIZECOLUMNS)."""
    if ctx.shadow is None:
        raise NotSupportedError("ALLSELECTED outside an iteration such as SUMMARIZECOLUMNS")
    if not args:
        raise NotSupportedError("ALLSELECTED() as a table")
    restored = ev.calculate_context([Call.raw("ALLSELECTED", list(args), "modifier")], ctx)
    if len(args) == 1 and not is_column(args[0]):
        return RowSet(args[0], ev.storage.visible(args[0], restored.filters), blank=True)
    columns: list[Column] = list(args)
    table = columns[0].table
    for c in columns:
        ev.storage.check_text(c)
    values = (
        ev.storage.visible_rows(table, restored.filters)
        .select([c.name for c in columns])
        .unique(maintain_order=True)
    )
    return Frame(
        [ResultColumn(f"{table.name}[{c.name}]", c.data_type, c) for c in columns],
        [list(r) for r in values.rows()],
    )


TABLES: dict[str, TableHandler] = {
    "ALLSELECTED": _allselected,
    "FILTER": _filter,
    "ADDCOLUMNS": _addcolumns,
    "SELECTCOLUMNS": _selectcolumns,
    "GENERATESERIES": _generateseries,
    "TOPN": _topn,
    "CROSSJOIN": _crossjoin,
    "GENERATE": _generate(False),
    "GENERATEALL": _generate(True),
    "UNION": _union,
    "INTERSECT": _intersect_except("INTERSECT"),
    "EXCEPT": _intersect_except("EXCEPT"),
    "NATURALINNERJOIN": _natural_join(False),
    "NATURALLEFTOUTERJOIN": _natural_join(True),
    "SUMMARIZE": _summarize,
    "GROUPBY": _groupby,
    "CURRENTGROUP": _currentgroup,
}
