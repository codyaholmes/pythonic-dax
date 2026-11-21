"""SUMMARIZECOLUMNS with every group computed at once (Phase 9).

Power BI's storage engine answers a visual with a few grouped scans, not one scan per
cell. This module does the same. An expression is evaluated for a list of filter
contexts at once (one per group of the visual), and each aggregation in it is computed
in one of two ways:

- **Group-by.** In the visual's own group contexts, it takes the table's rows visible
  under the query's filters without the group-by columns, labels each row with the
  group-by values it reaches through relationships, and aggregates with one Polars
  ``group_by``.
- **Data cache.** In contexts CALCULATE made (one per group, for example the same
  dates last year), it aggregates the table once by the rows of the tables that filter
  it, and then adds up the parts each context can see.

Both are exact when filters only flow from the one side of relationships to the many
side: a table's visible rows are then its own filters ANDed with lookups of the visible
rows of the tables on the one side, so a context's rows are the rows whose lookups it
allows. When a relationship filters both ways, or an aggregation isn't one of the
shapes in ``_vector.py``, the expression is evaluated per group by the formula engine,
exactly as before. Scalar steps (arithmetic, DIVIDE, IF, ...) run once per group on the
results. If anything raises, the whole query is evaluated the old way, so this module
never changes a result or an error.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import polars as pl

from . import _scalar as sc
from . import _vector
from ._evaluate import Context, Precomputed, _Lazy
from ._vector import (
    COLUMN_AGGREGATIONS,
    ITERATORS,
    Aggregation,
    NoVector,
    RowExpression,
    column_aggregation,
    iterator_aggregation,
)
from .expr import (
    BinaryOp,
    Call,
    Literal_,
    UnaryOp,
    VarBlock,
    VarRef,
    _Omitted,
    is_column,
    is_measure,
)
from .types import DaxType

if TYPE_CHECKING:
    from ._evaluate import Evaluator, Value
    from ._filters import FilterContext
    from .objects import Column, Relationship, Table

# Scalar functions whose handlers only evaluate their arguments with ``ev.scalar`` and
# combine the values, so they can run on each group's precomputed arguments.
_PURE = frozenset({"DIVIDE", "IF", "COALESCE", "AND", "OR", "NOT", "ISBLANK", "ABS"})
_CONSTANT = frozenset({"BLANK", "TRUE", "FALSE"})


STRICT = False
"""Tests set this to see errors instead of falling back to the per-group path."""

used = 0
"""How many groupings were computed here (tests check the fast path is taken)."""


class _PerGroup(Exception):
    """This expression is evaluated per group by the formula engine."""


class _Grouping:
    """The group contexts of a SUMMARIZECOLUMNS: the query's filters without the
    group-by columns (``base``), plus one combination of group-by values each."""

    def __init__(
        self,
        tables: list[list[Column]],
        group: list[Column],
        combos: pl.DataFrame,
        ctx: Context,
    ) -> None:
        self.tables = tables
        self.group = group
        self.combos = combos
        self.base = ctx.filters.remove({id(c) for c in group})


class _Contexts:
    """The filter contexts an expression is evaluated in, one per group."""

    def __init__(
        self,
        size: int,
        make: Callable[[], list[Context]],
        grouping: _Grouping | None = None,
    ) -> None:
        self.size = size
        self._make = make
        self._contexts: list[Context] | None = None
        self.grouping = grouping

    @property
    def contexts(self) -> list[Context]:
        if self._contexts is None:
            self._contexts = self._make()
        return self._contexts

    def bound(self, env: dict[int, list[Value]]) -> list[Context]:
        """The contexts with the batch's variables bound to each group's value."""
        if not env:
            return self.contexts
        result = []
        for i, context in enumerate(self.contexts):
            bound = dict(context.env)
            for key, values in env.items():
                bound[key] = _known(values[i])
            result.append(replace(context, env=bound))
        return result


def _known(value: Value) -> _Lazy:
    lazy = _Lazy(lambda: None)
    lazy.done, lazy.value = True, value
    return lazy


def batch_group_rows(
    ev: Evaluator,
    tables: list[list[Column]],
    combos: pl.DataFrame,
    named: list[tuple[str, Any]],
    ctx: Context,
) -> list[tuple[tuple[Value, ...], list[Value]]] | None:
    """The rows ``Evaluator._group_rows`` returns, or None to use its per-group path."""
    if not _vector.ENABLED or not _one_way(ev, ctx):
        return None
    group = [c for cols in tables for c in cols]
    for col in group:
        if col.data_type is DaxType.DECIMAL and combos[f"{id(col)}"].is_nan().any():
            return None
    grouping = _Grouping(tables, group, combos, ctx)
    rows = combos.rows() if combos.width else [()]

    def make() -> list[Context]:
        return [ev.group_context(tables, group, combo, ctx) for combo in rows]

    groups = _Contexts(len(rows), make, grouping)
    batch = _Batch(ev)
    try:
        batch.prefetch([expr for _, expr in named], groups)
        columns = [batch.values(expr, groups, {}) for _, expr in named]
    except Exception:
        # Errors, and anything this module can't do exactly: the per-group path gives
        # Power BI's answer or error.
        if STRICT:
            raise
        return None
    if not batch.used_storage:
        return None  # nothing was faster than the per-group path; keep its order of work
    global used
    used += 1
    result = []
    for i, combo in enumerate(rows):
        values = [column[i] for column in columns]
        if any(v is not None for v in values):
            result.append((combo, values))
    return result


def _one_way(ev: Evaluator, ctx: Context) -> bool:
    """Whether every active relationship filters only from its one side."""
    for rel in ev.model.relationships:
        state = ctx.filters.state(rel)
        if state.active and state.many_to_one:
            return False
    return True


class _Batch:
    def __init__(self, ev: Evaluator) -> None:
        self.ev = ev
        self.used_storage = False
        self._memo: dict[tuple[int, int], list[Value]] = {}
        self._code_cache: dict[tuple[Any, ...], tuple[pl.Series, pl.DataFrame]] = {}
        self._key_cache: dict[tuple[int, int], Any] = {}
        self._prefetched: dict[tuple[int, int], list[Value]] = {}
        self._keep: list[Any] = []

    # --- expressions --------------------------------------------------------------

    def values(self, node: Any, cs: _Contexts, env: dict[int, list[Value]]) -> list[Value]:
        """The node's value in each context."""
        key = (id(node), id(cs))
        if not env and key in self._memo:
            return self._memo[key]
        try:
            result = self._batched(node, cs, env)
        except (_PerGroup, NoVector):
            result = [self.ev.scalar(node, c) for c in cs.bound(env)]
        if not env:
            self._memo[key] = result
            self._keep.append((node, cs))
        return result

    def _batched(self, node: Any, cs: _Contexts, env: dict[int, list[Value]]) -> list[Value]:
        if isinstance(node, Literal_):
            return [self.ev.scalar(node, Context())] * cs.size
        if is_measure(node):
            return self.values(node.expression, cs, {})  # a measure can't see outer VARs
        if isinstance(node, VarRef):
            values = env.get(id(node.value))
            if values is None:
                raise _PerGroup  # a variable from outside the visual's query
            return values
        if isinstance(node, VarBlock):
            inner = dict(env)
            for _, value in node.variables:
                if value.kind == "table":
                    raise _PerGroup
                inner[id(value)] = self.values(value, cs, inner)
            return self.values(node.body, cs, inner)
        if isinstance(node, UnaryOp):
            operand = self.values(node.operand, cs, env)
            if node.op == "-":
                return [sc.negate(v) for v in operand]
            return operand
        if isinstance(node, BinaryOp):
            return self._binary(node, cs, env)
        if isinstance(node, Call):
            return self._call(node, cs, env)
        raise _PerGroup

    def _binary(self, node: BinaryOp, cs: _Contexts, env: dict[int, list[Value]]) -> list[Value]:
        op = node.op
        if op == "IN":
            raise _PerGroup
        left = self.values(node.left, cs, env)
        right = self.values(node.right, cs, env)
        pairs = zip(left, right, strict=True)
        if op == "&&":
            return [sc.to_bool(a) and sc.to_bool(b) for a, b in pairs]
        if op == "||":
            return [sc.to_bool(a) or sc.to_bool(b) for a, b in pairs]
        if op in ("+", "-", "*", "/", "^"):
            return [sc.arithmetic(op, a, b) for a, b in pairs]
        if op == "&":
            return [sc.concatenate(a, b) for a, b in pairs]
        return [sc.compare(op, a, b) for a, b in pairs]

    def _call(self, node: Call, cs: _Contexts, env: dict[int, list[Value]]) -> list[Value]:
        name, args = node.name, node.args
        if name in _CONSTANT:
            return [self.ev.scalar(node, Context())] * cs.size
        if name == "CALCULATE":
            return self._calculate(args[0], args[1:], cs, env)
        as_calculate = getattr(self.ev.scalar_handler(name), "as_calculate", None)
        if as_calculate is not None:  # TOTALYTD and the like
            expression, filters = as_calculate(args)
            return self._calculate(expression, filters, cs, env)
        spec = self._spec(node)
        if spec is not None:
            return self._aggregate(spec, cs)
        if name in _PURE or (name in ("MIN", "MAX") and len(args) == 2):
            evaluated = [a if isinstance(a, _Omitted) else self.values(a, cs, env) for a in args]
            handler = self.ev.scalar_handler(name)
            result = []
            for i in range(cs.size):
                values = [
                    a if isinstance(a, _Omitted) else Precomputed(a_node, a[i])
                    for a, a_node in zip(evaluated, args, strict=True)
                ]
                result.append(handler(self.ev, values, Context()))
            return result
        raise _PerGroup

    def _calculate(
        self, expression: Any, filters: list[Any], cs: _Contexts, env: dict[int, list[Value]]
    ) -> list[Value]:
        """CALCULATE: the formula engine works out each group's new filter context
        (cheap: filter arguments mostly touch small tables), then the expression is
        evaluated in all of them at once."""
        bound = cs.bound(env)

        def make() -> list[Context]:
            return [self.ev.calculate_context(filters, c) for c in bound]

        new = _Contexts(cs.size, make)
        if not all(_one_way(self.ev, c) for c in new.contexts):
            raise _PerGroup
        return self.values(expression, new, env)

    # --- aggregations -------------------------------------------------------------

    def _spec(self, node: Call) -> _Spec | None:
        """The aggregation a call makes, if it is one of the shapes computed here."""
        name = node.name
        args: list[Any] = node.args
        if name in COLUMN_AGGREGATIONS and len(args) == 1 and is_column(args[0]):
            column = args[0]
            if column.name not in self.ev.storage.ext(column.table).columns:
                raise _PerGroup  # a calculated column being computed
            return _Spec(node, column.table, column, name)
        if name == "COUNTROWS" and args and getattr(args[0], "is_table_ref", False):
            return _Spec(node, args[0], None, name)
        if (
            name in ITERATORS
            and len(args) == 2
            and getattr(args[0], "is_table_ref", False)
            and not isinstance(args[1], _Omitted)
        ):
            return _Spec(node, args[0], RowExpression(self.ev, args[0], args[1]), name)
        return None

    def _aggregate(self, spec: _Spec, cs: _Contexts) -> list[Value]:
        """An aggregation over its table's visible rows, in each context."""
        if cs.grouping is not None:
            hit = self._prefetched.get((id(spec.node), id(cs)))
            if hit is not None:
                return hit
            return self._grouped([spec], cs.grouping)[0]
        return self._data_cache(spec, cs)

    def prefetch(self, nodes: list[Any], cs: _Contexts) -> None:
        """Compute the aggregations the nodes make directly in the visual's groups
        (not inside CALCULATE) together: one ``group_by`` per table, like a storage
        engine query that fetches several measures at once."""
        assert cs.grouping is not None
        found: dict[int, list[_Spec]] = {}
        seen: set[int] = set()

        def walk(node: Any) -> None:
            if id(node) in seen:
                return
            seen.add(id(node))
            if is_measure(node):
                walk(node.expression)
            elif isinstance(node, VarBlock):
                for _, value in node.variables:
                    walk(value)
                walk(node.body)
            elif isinstance(node, UnaryOp):
                walk(node.operand)
            elif isinstance(node, BinaryOp):
                if node.op != "IN":
                    walk(node.left)
                    walk(node.right)
            elif isinstance(node, Call):
                if node.name in _PURE or (node.name in ("MIN", "MAX") and len(node.args) == 2):
                    for arg in node.args:
                        if not isinstance(arg, _Omitted):
                            walk(arg)
                    return
                try:
                    spec = self._spec(node)
                except (_PerGroup, NoVector):
                    return
                if spec is not None:
                    found.setdefault(id(spec.table), []).append(spec)

        for node in nodes:
            walk(node)
        for specs in found.values():
            if len(specs) < 2:
                continue
            try:
                results = self._grouped(specs, cs.grouping)
            except (_PerGroup, NoVector):
                continue
            for spec, values in zip(specs, results, strict=True):
                self._prefetched[(id(spec.node), id(cs))] = values

    def _real_mask(self, table: Table, mask: pl.Series | None) -> pl.Series:
        real = self.ev.storage.real_rows(table)
        return real if mask is None else mask & real

    def _values(self, spec: _Spec, mask: pl.Series) -> tuple[pl.Series, Aggregation]:
        """The aggregated values for the rows in ``mask``, and the aggregation."""
        from ._evaluate import infer_type

        source = spec.source
        if isinstance(source, RowExpression):
            values, kind = source.values(mask)
            agg = iterator_aggregation(spec.name, kind, values, infer_type(spec.node), source.bound)
            return values.alias("v"), agg
        if source is None:
            values = pl.repeat(0, int(mask.sum()), dtype=pl.Int8, eager=True).alias("v")
            return values, Aggregation(lambda v: [v.len()], lambda r: int(r[0]) or None, ("sum",))
        values = self.ev.storage.ext(spec.table)[source.name].filter(mask).alias("v")
        return values, column_aggregation(self.ev, spec.name, source, values)

    def _grouped(self, specs: list[_Spec], grouping: _Grouping) -> list[list[Value]]:
        """Aggregations over one table for every group, with one ``group_by``."""
        table = specs[0].table
        visible, mask, gid, labels = self._group_keys(table, grouping)
        columns = []
        aggs = []
        for j, spec in enumerate(specs):
            values, agg = self._values(spec, visible)
            if mask is not None:
                values = values.filter(mask)  # rows that reach no group
            columns.append(values.alias(f"v{j}"))
            aggs.append(agg)
        self.used_storage = True
        size = grouping.combos.height if grouping.combos.width else 1
        frame = pl.DataFrame(columns)
        parts = [agg.on(f"v{j}", f"a{j}_") for j, agg in enumerate(aggs)]
        if gid is None:
            row = frame.select([e for p in parts for e in p]).row(0, named=True)
            return [[agg.finish(_take(row, j, agg))] * size for j, agg in enumerate(aggs)]
        empty_row = frame.clear().select([e for p in parts for e in p]).row(0, named=True)
        grouped = (
            frame.with_columns(gid)
            .group_by("__gid")
            .agg([*(e for p in parts for e in p), pl.lit(True).alias("__hit")])
            .join(labels, on="__gid", how="left")
        )
        names = [n for n in labels.columns if n != "__gid"]
        joined = grouping.combos.select(names).join(
            grouped, on=names, how="left", nulls_equal=True, maintain_order="left"
        )
        hit = joined["__hit"].to_list()
        result = []
        for j, agg in enumerate(aggs):
            empty = agg.finish(_take(empty_row, j, agg))
            cols = [joined[f"a{j}_{i}"].to_list() for i in range(len(agg.exprs))]
            rows = zip(*cols, strict=True)
            result.append([agg.finish(r) if h else empty for r, h in zip(rows, hit, strict=True)])
        return result

    def _group_keys(
        self, table: Table, grouping: _Grouping
    ) -> tuple[pl.Series, pl.Series | None, pl.Series | None, pl.DataFrame]:
        """For aggregations over ``table`` in the visual's groups: its visible rows
        (a mask over ``ext``), which of those reach some group (a mask over the
        visible rows, or None for all), each such row's group number, and the group-by
        values of each group number. The group number is None when no group-by column
        filters the table. Shared by every aggregation over the table."""
        key = (id(table), id(grouping))
        hit = self._key_cache.get(key)
        if hit is not None:
            return hit  # type: ignore[no-any-return]
        storage = self.ev.storage
        visible = self._real_mask(table, storage.visible(table, grouping.base))
        reach = visible
        codes: list[pl.Series] = []  # a code per row of ``table``, one per key
        labels: list[pl.DataFrame] = []  # the group-by values of each code
        for cols in grouping.tables:
            # Label the group table's rows with a code per combination of their
            # group-by values, then look up each row's code: cheaper than looking up
            # the values themselves (often text) for every row.
            if cols[0].table is table:
                code, label = self._codes(cols)
            else:
                lookup = self._group_lookup(cols[0].table, table, grouping.base)
                if lookup is None:
                    continue  # these group-by columns don't filter this table
                code, label = self._codes(cols)
                reach = reach & lookup.is_not_null()
                code = code.gather(lookup)
            codes.append(code)
            labels.append(label)
        if not codes:
            result: Any = (visible, None, None, pl.DataFrame())
            self._key_cache[key] = result
            return result  # type: ignore[no-any-return]
        partial = None if reach.equals(visible) else reach.filter(visible)
        # The group number combines the codes like digits, each in base the number of
        # codes of its table; the labels are every combination of the tables' labels.
        sizes = [label.height for label in labels]
        if math.prod(sizes) >= 2**62:
            raise _PerGroup
        gid_expr: pl.Expr = pl.lit(0, dtype=pl.UInt64)
        label_frame = pl.DataFrame({"__gid": pl.Series([0], dtype=pl.UInt64)})
        for i, label in enumerate(labels):
            code_name = label.columns[0]
            gid_expr = gid_expr * sizes[i] + pl.col(f"c{i}").cast(pl.UInt64)
            label_frame = label_frame.join(label, how="cross").select(
                (pl.col("__gid") * sizes[i] + pl.col(code_name).cast(pl.UInt64)).alias("__gid"),
                pl.exclude("__gid", code_name),
            )
        frame = pl.DataFrame([c.alias(f"c{i}") for i, c in enumerate(codes)]).filter(reach)
        gid = frame.select(gid_expr.alias("__gid"))["__gid"]
        result = (visible, partial, gid, label_frame)
        self._key_cache[key] = result
        return result  # type: ignore[no-any-return]

    def _codes(self, cols: list[Column]) -> tuple[pl.Series, pl.DataFrame]:
        """A code for each row of the columns' table (in ``ext`` order) naming its
        combination of their values, and the values of each code."""
        key = tuple(id(c) for c in cols)
        hit = self._code_cache.get(key)
        if hit is not None:
            return hit
        name = f"__code{len(self._code_cache)}"
        values = self.ev.storage.ext(cols[0].table).select(
            [pl.col(c.name).alias(f"{id(c)}") for c in cols]
        )
        label = values.unique(maintain_order=True).with_row_index(name)
        codes = values.join(
            label, on=values.columns, how="left", nulls_equal=True, maintain_order="left"
        )[name]
        self._code_cache[key] = (codes, label)
        return codes, label

    def _group_lookup(self, source: Table, table: Table, base: FilterContext) -> pl.Series | None:
        """How rows of ``table`` find their row of ``source`` (a group-by table): the
        row index along the one filter path (null for rows with none), or None when
        ``source`` doesn't filter ``table``."""
        paths = self._paths(table, source, base, frozenset({id(table)}))
        if not paths:
            return None
        if len(paths) > 1:
            raise _PerGroup  # several filter paths: leave it to the formula engine
        index: pl.Series | None = None
        for rel in paths[0]:
            step = self.ev.storage.many_to_one(rel)
            index = step if index is None else step.gather(index)
        return index

    def _paths(
        self, table: Table, target: Table, base: FilterContext, seen: frozenset[int]
    ) -> list[list[Relationship]]:
        """The active relationship chains along which ``target`` filters ``table``."""
        found: list[list[Relationship]] = []
        for rel in self.ev.model.relationships:
            state = base.state(rel)
            if not state.active or not state.one_to_many or rel.from_table is not table:
                continue
            other = rel.to_table
            if id(other) in seen:
                continue
            if other is target:
                found.append([rel])
            else:
                for rest in self._paths(other, target, base, seen | {id(other)}):
                    found.append([rel, *rest])
        return found

    def _data_cache(self, spec: _Spec, cs: _Contexts) -> list[Value]:
        """The aggregation in arbitrary contexts.

        With filters flowing one way, a row of the table is visible in a context when
        the table's own filters allow it and, for each relationship to a table on the
        one side, that table's row it looks up is visible. When the table's own filters
        are the same in every context, the rows are aggregated once by their lookups,
        and each context adds up the parts whose lookups it can see.
        """
        storage = self.ev.storage
        table = spec.table
        contexts = [c.filters for c in cs.contexts]
        own = {tuple(f.signature for f in fc.filters if f.table is table) for fc in contexts}
        if len(own) > 1:
            return self._each(spec, contexts)
        common = self._real_mask(table, storage.own_mask(table, contexts[0]))
        varying: list[tuple[str, Relationship, list[pl.Series | None]]] = []
        for rel in self.ev.model.relationships:
            if rel.from_table is not table:
                continue
            per: list[pl.Series | None] = []
            for fc in contexts:
                state = fc.state(rel)
                if state.active and state.one_to_many:
                    per.append(storage.visible(rel.to_table, fc))
                else:
                    per.append(None)
            if all(p is None for p in per):
                continue
            first = per[0]
            if first is not None and all(p is not None and p.equals(first) for p in per):
                common = common & storage._restrict(table, rel, first)
                continue
            varying.append((f"r{len(varying)}", rel, per))
        values, agg = self._values(spec, common)
        self.used_storage = True
        if not varying:
            value = agg.finish(values.to_frame().select(agg.named("a")).row(0))
            return [value] * cs.size
        if agg.combine is None:
            return self._each(spec, contexts)
        keys = [storage.many_to_one(rel).filter(common).alias(k) for k, rel, _ in varying]
        cache = (
            pl.DataFrame([values, *keys]).group_by([k for k, _, _ in varying]).agg(agg.named("a"))
        )
        result = []
        for i in range(cs.size):
            conditions = [
                pl.col(k).is_in(per[i].arg_true().implode())  # type: ignore[union-attr]
                for k, _, per in varying
                if per[i] is not None
            ]
            part = cache.filter(conditions) if conditions else cache
            result.append(agg.finish(part.select(agg.combined("a")).row(0)))
        return result

    def _each(self, spec: _Spec, contexts: list[FilterContext]) -> list[Value]:
        """The aggregation in each context, one Polars pass each."""
        storage = self.ev.storage
        result = []
        for fc in contexts:
            mask = self._real_mask(spec.table, storage.visible(spec.table, fc))
            values, agg = self._values(spec, mask)
            result.append(agg.finish(values.to_frame().select(agg.named("a")).row(0)))
        self.used_storage = True
        return result


@dataclass
class _Spec:
    """An aggregation call: of a column, a row expression (iterators), or None to
    count rows (COUNTROWS), over a table."""

    node: Call
    table: Table
    source: Column | RowExpression | None
    name: str


def _take(row: dict[str, Any], j: int, agg: Aggregation) -> tuple[Any, ...]:
    return tuple(row[f"a{j}_{i}"] for i in range(len(agg.exprs)))
