"""Window functions (Phase 5): OFFSET, INDEX, WINDOW, RANK and ROWNUMBER.

Each works on a relation (a table) sorted by ORDERBY and split by PARTITIONBY. The
current row is found by matching the relation's columns against the current context:
a row context gives one value, otherwise the filter context gives the visible values.
When several rows of the relation match, OFFSET, INDEX and WINDOW return the union of
their results, and RANK and ROWNUMBER return BLANK, as in Power BI.
"""

from __future__ import annotations

import functools
import itertools
from collections.abc import Callable
from dataclasses import dataclass, replace
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from . import _scalar as sc
from ._evaluate import Context, Evaluator, Frame, TableValue, Value
from ._iterators import _has, order_args
from .errors import DaxError, NotSupportedError
from .expr import Keyword, _Omitted

if TYPE_CHECKING:
    from .objects import Column


def _pad(args: list[Any], size: int) -> list[Any]:
    """The arguments with trailing ones left out filled in as omitted."""
    return [*args, *[_Omitted()] * (size - len(args))]


def _given(arg: Any) -> bool:
    return not isinstance(arg, _Omitted)


def _keyword(arg: Any, default: str) -> str:
    return arg.name if isinstance(arg, Keyword) else default


@dataclass
class _Window:
    frame: Frame
    partitions: list[list[int]]  # row indexes of the frame, each partition sorted
    keys: list[list[Value]]  # the order keys of each frame row
    compare: Callable[[list[Value], list[Value]], int]
    current: list[int]  # frame rows matching the current context

    def position(self, row: int) -> tuple[list[int], int]:
        for part in self.partitions:
            if row in part:
                return part, part.index(row)
        raise AssertionError("row not in any partition")  # pragma: no cover

    def current_partitions(self, partitioned: bool) -> list[list[int]]:
        if not partitioned:
            return self.partitions
        seen: list[list[int]] = []
        for row in self.current:
            part, _ = self.position(row)
            if not any(p is part for p in seen):
                seen.append(part)
        return seen

    def table(self, rows: set[int]) -> Frame:
        return Frame(
            self.frame.columns,
            [self.frame.rows[i] for i in sorted(rows)],
            filterable=self.frame.filterable,
        )


def _norm(value: Value) -> Any:
    return value.lower() if isinstance(value, str) else value


def _cmp(descending: list[bool], blanks: str) -> Callable[[list[Value], list[Value]], int]:
    def cmp(a: list[Value], b: list[Value]) -> int:
        for x, y, desc in zip(a, b, descending, strict=True):
            if x is None and y is None:
                continue
            if x is None or y is None:
                first = -1 if x is None else 1  # -1 when the BLANK comes first
                if blanks == "FIRST":
                    return first
                if blanks == "LAST":
                    return -first
                return -first if desc else first  # DEFAULT: BLANK is the smallest value
            if sc.compare("<", x, y):
                return 1 if desc else -1
            if sc.compare(">", x, y):
                return -1 if desc else 1
        return 0

    return cmp


def _check_default_blanks(keys: list[list[Value]], blanks: str) -> None:
    """BLANK sorts first by default; that is verified only next to positive numbers,
    dates and non-empty text."""
    if blanks != "DEFAULT" or not keys:
        return
    for j in range(len(keys[0])):
        column = [k[j] for k in keys]
        if None not in column:
            continue
        for v in column:
            if isinstance(v, bool):
                raise NotSupportedError("ORDERBY of True/False values with BLANK")
            if (isinstance(v, int | float | Decimal) and v < 0) or v == "":
                raise NotSupportedError(
                    "ORDERBY with BLANK next to negative numbers or empty text "
                    "(use BLANKS FIRST or LAST)"
                )


def _window(ev: Evaluator, name: str, parts: list[Any], ctx: Context, *, unique: bool) -> _Window:
    """``parts`` is (relation, orderBy, blanks, partitionBy, matchBy)."""
    relation, order_by, blanks_arg, partition_by, match_by = parts
    if not _given(relation):
        raise NotSupportedError(f"{name} without a relation")
    if not _given(order_by):
        raise NotSupportedError(f"{name} without ORDERBY")
    if _given(match_by):
        raise NotSupportedError(f"{name} with MATCHBY")
    frame = ev.to_frame(ev.table(relation, ctx))
    lineage = [c.column for c in frame.columns]
    if all(c is None for c in lineage):
        raise DaxError(
            f"{name}'s Relation parameter only contains columns added by DAX table "
            "functions. This is not supported."
        )
    if any(c is None for c in lineage):
        raise NotSupportedError(f"{name} over a relation with columns added by DAX functions")
    columns: list[Column] = [c for c in lineage if c is not None]

    rows = ev.rows_of(frame)
    orders = order_args(order_by.args, "ASC")
    keys = [[ev.scalar(e, replace(ctx, rows=(*ctx.rows, row))) for e, _ in orders] for row in rows]
    blanks = _keyword(blanks_arg, "DEFAULT")
    _check_default_blanks(keys, blanks)
    compare = _cmp([d for _, d in orders], blanks)

    part_index: list[int] = []
    if _given(partition_by):
        for col in partition_by.args:
            i = frame.find(col)
            if i is None:
                raise DaxError(f"{name}: the PARTITIONBY column {col.ref} isn't in the relation.")
            part_index.append(i)
    groups: dict[tuple[Any, ...], list[int]] = {}
    for r, row in enumerate(frame.rows):
        groups.setdefault(tuple(_norm(row[i]) for i in part_index), []).append(r)

    def by_key(a: int, b: int) -> int:
        return compare(keys[a], keys[b])

    partitions = []
    for members in groups.values():
        ordered = sorted(members, key=functools.cmp_to_key(by_key))
        if unique:
            for a, b in itertools.pairwise(ordered):
                if compare(keys[a], keys[b]) == 0:
                    raise NotSupportedError(
                        f"{name} whose ORDERBY and PARTITIONBY don't identify each row "
                        "(Power BI picks extra columns to order by)"
                    )
        partitions.append(ordered)

    return _Window(frame, partitions, keys, compare, _current_rows(ev, frame, columns, ctx))


def _current_rows(ev: Evaluator, frame: Frame, columns: list[Column], ctx: Context) -> list[int]:
    """The relation's rows that match the current context."""
    fixed: dict[int, Any] = {}
    by_table: dict[int, list[int]] = {}
    for i, col in enumerate(columns):
        if any(_has(ev, row, col) for row in ctx.rows):
            fixed[i] = _norm(ev.scalar(col, ctx))
        else:
            by_table.setdefault(id(col.table), []).append(i)
    allowed: list[tuple[list[int], set[tuple[Any, ...]]]] = []
    for indexes in by_table.values():
        table = columns[indexes[0]].table
        names = [columns[i].name for i in indexes]
        visible = ev.storage.visible_rows(table, ctx.filters).select(names).unique()
        allowed.append((indexes, {tuple(_norm(v) for v in r) for r in visible.rows()}))
    result = []
    for r, row in enumerate(frame.rows):
        if any(_norm(row[i]) != v for i, v in fixed.items()):
            continue
        if all(tuple(_norm(row[i]) for i in idx) in ok for idx, ok in allowed):
            result.append(r)
    return result


def _whole_number(ev: Evaluator, node: Any, ctx: Context, name: str) -> int:
    value = ev.scalar(node, ctx)
    if isinstance(value, bool) or not isinstance(value, int):
        raise NotSupportedError(f"{name} with a position that isn't a whole number")
    return value


def _offset(ev: Evaluator, args: list[Any], ctx: Context) -> TableValue:
    args = _pad(args, 6)
    delta = _whole_number(ev, args[0], ctx, "OFFSET")
    w = _window(ev, "OFFSET", args[1:6], ctx, unique=True)
    selected = set()
    for row in w.current:
        part, pos = w.position(row)
        if 0 <= pos + delta < len(part):
            selected.add(part[pos + delta])
    return w.table(selected)


def _absolute(position: int, size: int, name: str) -> int:
    if position == 0:
        raise NotSupportedError(f"{name} with position 0")
    return position - 1 if position > 0 else size + position


def _index(ev: Evaluator, args: list[Any], ctx: Context) -> TableValue:
    args = _pad(args, 6)
    position = _whole_number(ev, args[0], ctx, "INDEX")
    w = _window(ev, "INDEX", args[1:6], ctx, unique=True)
    selected = set()
    for part in w.current_partitions(_given(args[4])):
        i = _absolute(position, len(part), "INDEX")
        if 0 <= i < len(part):
            selected.add(part[i])
    return w.table(selected)


def _window_table(ev: Evaluator, args: list[Any], ctx: Context) -> TableValue:
    args = _pad(args, 9)
    start = _whole_number(ev, args[0], ctx, "WINDOW")
    end = _whole_number(ev, args[2], ctx, "WINDOW")
    start_type, end_type = args[1].name, args[3].name
    w = _window(ev, "WINDOW", args[4:9], ctx, unique=True)

    def bounds(part: list[int], pos: int | None) -> tuple[int, int]:
        def at(value: int, kind: str) -> int:
            if kind == "ABS":
                return _absolute(value, len(part), "WINDOW")
            assert pos is not None
            return pos + value

        return max(at(start, start_type), 0), min(at(end, end_type), len(part) - 1)

    selected: set[int] = set()
    if start_type == end_type == "ABS":
        for part in w.current_partitions(_given(args[7])):
            lo, hi = bounds(part, None)
            selected.update(part[lo : hi + 1])
    else:
        for row in w.current:
            part, pos = w.position(row)
            lo, hi = bounds(part, pos)
            selected.update(part[lo : hi + 1])
    return w.table(selected)


def _rank(ev: Evaluator, args: list[Any], ctx: Context) -> Value:
    args = _pad(args, 6)
    ties = _keyword(args[0], "SKIP")
    w = _window(ev, "RANK", args[1:6], ctx, unique=False)
    if len(w.current) != 1:
        return None
    part, _ = w.position(w.current[0])
    key = w.keys[w.current[0]]
    before = [w.keys[r] for r in part if w.compare(w.keys[r], key) < 0]
    if ties == "SKIP":
        return len(before) + 1
    distinct: list[list[Value]] = []
    for k in before:
        if not any(w.compare(k, d) == 0 for d in distinct):
            distinct.append(k)
    return len(distinct) + 1


def _rownumber(ev: Evaluator, args: list[Any], ctx: Context) -> Value:
    args = _pad(args, 5)
    w = _window(ev, "ROWNUMBER", args[0:5], ctx, unique=True)
    if len(w.current) != 1:
        return None
    _, pos = w.position(w.current[0])
    return pos + 1


SCALAR = {"RANK": _rank, "ROWNUMBER": _rownumber}
TABLES = {"OFFSET": _offset, "INDEX": _index, "WINDOW": _window_table}
