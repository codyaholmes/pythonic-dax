"""Filter context, and how filters flow across relationships.

A filter context is a list of column filters. Each filter names one or more columns
of a table and the combinations of values it allows. A table's visible rows are the
rows that pass its own filters and the filters that reach it through active
relationships: from the one side to the many side always, and back from the many side
when the relationship filters both ways.

Rows on the many side whose key has no match on the one side belong to the one side's
hidden blank row, so they are visible exactly when that blank row is.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import polars as pl

from .errors import NotSupportedError
from .objects import Cardinality, CrossFilter
from .types import DaxType

if TYPE_CHECKING:
    from .model import Model
    from .objects import Column, Relationship, Table


_CONTENT_KEY_ROWS = 256


class _Identity:
    """Hashes and compares an object by identity, keeping it alive."""

    __slots__ = ("obj",)

    def __init__(self, obj: object) -> None:
        self.obj = obj

    def __hash__(self) -> int:
        return id(self.obj)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _Identity) and other.obj is self.obj


class ColumnFilter:
    """Allowed value combinations for some columns of one table.

    ``grouping`` marks the filters SUMMARIZECOLUMNS adds for its group-by columns,
    which is what ISINSCOPE looks for.
    """

    __slots__ = ("_signature", "columns", "grouping", "table", "values")

    def __init__(
        self,
        table: Table,
        columns: Sequence[Column],
        values: pl.DataFrame,
        *,
        grouping: bool = False,
    ) -> None:
        self.table = table
        self.columns = tuple(columns)
        self.values = values  # one column per filter column, named like the column
        self.grouping = grouping
        self._signature: tuple[Any, ...] | None = None

    @property
    def signature(self) -> tuple[Any, ...]:
        """A hashable key that is equal for filters allowing the same values.

        Small value lists are compared by content; large ones by identity, holding a
        reference so the identity stays unique while the key is in use.
        """
        if self._signature is None:
            values: Any
            if self.values.height <= _CONTENT_KEY_ROWS:
                values = tuple(self.values.rows())
            else:
                values = _Identity(self.values)
            self._signature = (
                id(self.table),
                tuple(id(c) for c in self.columns),
                self.grouping,
                values,
            )
        return self._signature

    def without(self, ids: set[int]) -> ColumnFilter | None:
        """This filter with some columns taken out, or None if none are left."""
        rest = [c for c in self.columns if id(c) not in ids]
        if len(rest) == len(self.columns):
            return self
        if not rest:
            return None
        values = self.values.select([c.name for c in rest]).unique()
        return ColumnFilter(self.table, rest, values, grouping=self.grouping)

    def __repr__(self) -> str:
        cols = ", ".join(c.ref for c in self.columns)
        return f"ColumnFilter({cols}: {self.values.height} values)"


@dataclass(frozen=True)
class RelationshipState:
    """How a relationship behaves in a filter context (USERELATIONSHIP, CROSSFILTER)."""

    active: bool
    one_to_many: bool  # the one side filters the many side
    many_to_one: bool  # the many side filters the one side


def default_state(rel: Relationship) -> RelationshipState:
    return RelationshipState(rel.is_active, True, rel.cross_filter is CrossFilter.BOTH)


class FilterContext:
    """An immutable set of column filters, plus relationship changes made by
    USERELATIONSHIP and CROSSFILTER."""

    __slots__ = ("_signature", "filters", "overrides")

    def __init__(
        self,
        filters: tuple[ColumnFilter, ...] = (),
        overrides: tuple[tuple[Relationship, RelationshipState], ...] = (),
    ) -> None:
        self.filters = filters
        self.overrides = overrides
        self._signature: tuple[Any, ...] | None = None

    @property
    def signature(self) -> tuple[Any, ...]:
        """A hashable key that is equal for filter contexts with the same filters and
        relationship changes."""
        if self._signature is None:
            self._signature = (
                tuple(f.signature for f in self.filters),
                tuple((id(r), s) for r, s in self.overrides),
            )
        return self._signature

    def apply(
        self, new: Iterable[ColumnFilter], keep: Iterable[ColumnFilter] = ()
    ) -> FilterContext:
        """Add filters the way CALCULATE does.

        Each filter in ``new`` replaces existing filters on its columns (a filter over
        several columns keeps its other columns). Filters in ``keep`` (KEEPFILTERS)
        are added without replacing anything. New filters never replace each other.
        """
        new = list(new)
        ids = {id(c) for f in new for c in f.columns}
        return FilterContext((*self.remove(ids).filters, *new, *keep), self.overrides)

    def remove(self, ids: set[int]) -> FilterContext:
        """Take the columns with these ids out of every filter."""
        kept = [g for f in self.filters if (g := f.without(ids)) is not None]
        return FilterContext(tuple(kept), self.overrides)

    def cleared(self) -> FilterContext:
        """No filters at all (``ALL()``), keeping relationship changes."""
        return FilterContext((), self.overrides)

    def state(self, rel: Relationship) -> RelationshipState:
        for r, state in reversed(self.overrides):
            if r is rel:
                return state
        return default_state(rel)

    def override(self, rel: Relationship, state: RelationshipState) -> FilterContext:
        return FilterContext(self.filters, (*self.overrides, (rel, state)))

    def direct(self, column: Column) -> list[ColumnFilter]:
        """The filters that name this column."""
        return [f for f in self.filters if any(c is column for c in f.columns)]

    def __repr__(self) -> str:
        return f"FilterContext({list(self.filters)})"


def normalized(column: Column, name: str | None = None) -> pl.Expr:
    """The value DAX compares on. Text compares without case."""
    expr = pl.col(name or column.name)
    if column.data_type is DaxType.STRING:
        return expr.str.to_lowercase()
    return expr


def bool_series(height: int, value: bool) -> pl.Series:
    """A True/False series of one value, built without a Python list."""
    if height == 0:
        return pl.Series([], dtype=pl.Boolean)
    return pl.Series([value], dtype=pl.Boolean).new_from_index(0, height)  # fastest


class _ModelCache:
    """Work that depends only on a model's data, kept between evaluations.

    Entries remember the data frames they were built from (frames are immutable, so a
    reload or a new calculated column replaces them) and are rebuilt when those change.
    """

    def __init__(self) -> None:
        self.ext: dict[int, tuple[pl.DataFrame, bool, pl.DataFrame, int | None]] = {}
        self.maps: dict[int, tuple[Any, pl.Series]] = {}
        self.text: dict[int, pl.DataFrame] = {}
        self.keys: dict[int, tuple[Any, ...]] = {}
        self.rows: OrderedDict[tuple[int, int], tuple[Any, ...]] = OrderedDict()


def model_cache(model: Model) -> _ModelCache:
    cache: _ModelCache | None = getattr(model, "_engine_cache", None)
    if cache is None:
        cache = _ModelCache()
        model._engine_cache = cache  # type: ignore[attr-defined]
    return cache


class Storage:
    """The rows of each table, with the hidden blank row, plus cached visibility.

    One Storage serves one evaluation. Visibility is cached by the content of the
    filter context, so equal contexts built in different places share the work.
    Work that depends only on the data (the blank row, relationship lookups, checks)
    is cached on the model and reused by later evaluations while the data is unchanged.
    """

    def __init__(self, model: Model) -> None:
        self.model = model
        self.cache = model_cache(model)
        self._visible: dict[Any, pl.Series | None] = {}
        self._masks: dict[Any, pl.Series] = {}

    # --- rows ---------------------------------------------------------------------

    def ext(self, table: Table) -> pl.DataFrame:
        """The table's rows, followed by the blank row if Power BI would add one."""
        return self._ext_entry(table)[0]

    def _ext_entry(self, table: Table) -> tuple[pl.DataFrame, int | None]:
        key = id(table)
        data = table.data
        has_blank = table.has_blank_row
        hit = self.cache.ext.get(key)
        if hit is not None and hit[2] is data and hit[1] == has_blank:
            return hit[0], hit[3]
        if has_blank:
            blank = pl.DataFrame([pl.Series(c, [None], dtype=data.schema[c]) for c in data.columns])
            ext = pl.concat([data, blank], rechunk=False)
            index: int | None = data.height
        else:
            ext, index = data, None
        self.cache.ext[key] = (ext, has_blank, data, index)
        return ext, index

    def blank_index(self, table: Table) -> int | None:
        return self._ext_entry(table)[1]

    def real_rows(self, table: Table) -> pl.Series:
        """True for every row except the blank row."""
        ext, index = self._ext_entry(table)
        mask = bool_series(ext.height, True)
        if index is not None:
            mask = mask.scatter(index, False)
        return mask

    def check_text(self, column: Column) -> None:
        """Refuse text columns whose values differ only by case or trailing spaces.

        Power BI stores one spelling for such values, and which one it keeps depends on
        load order, so results that show these values could differ.
        """
        if column.data_type is not DaxType.STRING:
            return
        data = column.table.data
        if self.cache.text.get(id(column)) is data:
            return
        values = data[column.name].drop_nulls().unique()
        folded = values.str.to_lowercase().str.strip_chars_end(" ").n_unique()
        if folded != values.len():
            raise NotSupportedError(
                f"text values in {column.ref} that differ only by case or trailing spaces"
            )
        self.cache.text[id(column)] = data

    # --- visibility -----------------------------------------------------------------

    def visible(self, table: Table, filters: FilterContext) -> pl.Series | None:
        """A mask over :meth:`ext` rows, or None when nothing filters the table."""
        key = (id(table), filters.signature)
        if key not in self._visible:
            self._visible[key] = self._visible_from(table, filters, frozenset(), None)
        return self._visible[key]

    def visible_rows(self, table: Table, filters: FilterContext) -> pl.DataFrame:
        mask = self.visible(table, filters)
        ext = self.ext(table)
        return ext if mask is None else ext.filter(mask)

    def own_rows(self, table: Table, filters: FilterContext) -> pl.DataFrame:
        """Rows allowed by the filters on the table's own columns, ignoring filters that
        would reach it through relationships (SUMMARIZECOLUMNS group-by values)."""
        mask = self.own_mask(table, filters)
        ext = self.ext(table)
        return ext if mask is None else ext.filter(mask)

    def own_mask(self, table: Table, filters: FilterContext) -> pl.Series | None:
        mask: pl.Series | None = None
        for f in filters.filters:
            if f.table is table:
                m = self._filter_mask(f)
                mask = m if mask is None else mask & m
        return mask

    def blank_row_visible(self, table: Table, filters: FilterContext) -> bool:
        index = self.blank_index(table)
        if index is None:
            return False
        mask = self.visible(table, filters)
        return mask is None or bool(mask[index])

    def _visible_from(
        self,
        table: Table,
        filters: FilterContext,
        path: frozenset[int],
        via: Relationship | None,
    ) -> pl.Series | None:
        mask = self.own_mask(table, filters)
        for rel in self.model.relationships:
            state = filters.state(rel)
            if not state.active or rel is via:
                continue
            if rel.cardinality is Cardinality.MANY_TO_MANY:
                raise NotSupportedError("many-to-many relationships")
            if rel.from_table is rel.to_table:
                raise NotSupportedError("a relationship from a table to itself")
            if rel.from_table is table and state.one_to_many:
                other = rel.to_table
            elif rel.to_table is table and state.many_to_one:
                other = rel.from_table
            else:
                continue
            if id(other) in path:
                continue
            other_mask = self._visible_from(other, filters, path | {id(table)}, rel)
            if other_mask is None:
                continue
            restriction = self._restrict(table, rel, other_mask)
            mask = restriction if mask is None else mask & restriction
        return mask

    def direct_mask(self, column: Column, filters: FilterContext) -> pl.Series | None:
        """Rows allowed by the filters that name ``column`` (ignoring all others)."""
        mask: pl.Series | None = None
        for f in filters.direct(column):
            m = self._filter_mask(f)
            mask = m if mask is None else mask & m
        return mask

    def _filter_mask(self, f: ColumnFilter) -> pl.Series:
        key = f.signature
        hit = self._masks.get(key)
        if hit is not None:
            return hit
        ext = self.ext(f.table)
        keys = [f"k{i}" for i in range(len(f.columns))]
        right = f.values.select(
            [normalized(c).alias(k) for c, k in zip(f.columns, keys, strict=True)]
        )
        if len(keys) == 1:
            mask = ext.select(
                normalized(f.columns[0]).is_in(right["k0"].implode(), nulls_equal=True)
            ).to_series()
        else:
            left = ext.select(
                [normalized(c).alias(k) for c, k in zip(f.columns, keys, strict=True)]
            ).with_row_index("__i")
            hits = left.join(right, on=keys, how="semi", nulls_equal=True)["__i"]
            mask = bool_series(ext.height, False).scatter(hits, True)
        self._masks[key] = mask
        return mask

    def many_to_one(self, rel: Relationship) -> pl.Series:
        """For each row of the many side (in :meth:`ext` order), the index of its row on
        the one side: the blank row for keys with no match, or null when the one side
        has no blank row."""
        self._check_keys(rel)
        many, one = rel.from_column, rel.to_column
        many_data, one_data = many.table.data, one.table.data
        state = (many_data, one_data, many.table.has_blank_row, one.table.has_blank_row)
        hit = self.cache.maps.get(id(rel))
        if hit is not None and all(a is b for a, b in zip(hit[0], state, strict=True)):
            return hit[1]
        one_ext = self.ext(one.table)
        many_ext = self.ext(many.table)
        one_keys = (
            one_ext.select(rel.key(one).alias("k"))
            .with_row_index("__j")
            .filter(self.real_rows(one.table))
            .drop_nulls("k")
        )
        if one_keys["k"].n_unique() != one_keys.height:
            raise NotSupportedError(f"duplicate keys on the one side of {rel!r}")
        joined = many_ext.select(rel.key(many).alias("k")).join(
            one_keys, on="k", how="left", maintain_order="left"
        )
        index = joined["__j"]
        blank = self.blank_index(one.table)
        if blank is not None:
            index = index.fill_null(blank)
        self.cache.maps[id(rel)] = (state, index)
        return index

    def _restrict(self, table: Table, rel: Relationship, other_mask: pl.Series) -> pl.Series:
        """Rows of ``table`` allowed by the visible rows of the relationship's other side."""
        index = self.many_to_one(rel)
        if table is rel.from_column.table:
            return other_mask.gather(index).fill_null(False)
        hits = index.filter(other_mask).drop_nulls().unique()
        return bool_series(self.ext(table).height, False).scatter(hits, True)

    def _check_keys(self, rel: Relationship) -> None:
        state = (rel.from_column.table.data, rel.to_column.table.data, rel.cardinality)
        hit = self.cache.keys.get(id(rel))
        if hit is not None and hit[0] is state[0] and hit[1] is state[1] and hit[2] is state[2]:
            return
        for col in (rel.from_column, rel.to_column):
            if col.data_type is DaxType.STRING:
                values = col.table.data[col.name].drop_nulls()
                if (values.str.strip_chars_end(" ") != values).any():
                    raise NotSupportedError(
                        f"relationship keys with trailing spaces ({col.ref}); Power BI "
                        "may match them to keys without the spaces"
                    )
        if rel.text_to_number:
            ends = (rel.from_column, rel.to_column)
            text = next(c for c in ends if c.data_type is DaxType.STRING)
            values = text.table.data[text.name].drop_nulls()
            if (~values.str.contains(r"^-?(?:0|[1-9][0-9]*)$")).any():
                raise NotSupportedError(
                    f"a Text key that isn't a plain whole number ({text.ref}) in a relationship "
                    "to a Whole Number column"
                )
        if rel.cardinality is Cardinality.ONE_TO_ONE:
            one = rel.to_column.table.data.select(rel.key(rel.to_column).alias("k"))["k"]
            many = rel.from_column.table.data.select(rel.key(rel.from_column).alias("k"))["k"]
            if (~one.drop_nulls().is_in(many.drop_nulls().implode())).any():
                raise NotSupportedError(
                    "a one-to-one relationship where both sides have keys the other lacks"
                )
        self.cache.keys[id(rel)] = state

    # --- filters from rows -------------------------------------------------------------

    def row_filter(self, table: Table, index: int, columns: Sequence[Column]) -> ColumnFilter:
        """A filter matching one row's values in ``columns`` (context transition)."""
        names = [c.name for c in columns]
        values = self.ext(table).select(names).slice(index, 1)
        return ColumnFilter(table, columns, values)

    def present_columns(self, table: Table) -> list[Column]:
        """The table's columns that have values (a calculated column being computed
        doesn't yet)."""
        names = set(self.ext(table).columns)
        return [c for c in table.columns if c.name in names]
