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

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any

import polars as pl

from .errors import NotSupportedError
from .objects import Cardinality, CrossFilter
from .types import DaxType

if TYPE_CHECKING:
    from .model import Model
    from .objects import Column, Relationship, Table


class ColumnFilter:
    """Allowed value combinations for some columns of one table."""

    __slots__ = ("columns", "table", "values")

    def __init__(self, table: Table, columns: Sequence[Column], values: pl.DataFrame) -> None:
        self.table = table
        self.columns = tuple(columns)
        self.values = values  # one column per filter column, named like the column

    def __repr__(self) -> str:
        cols = ", ".join(c.ref for c in self.columns)
        return f"ColumnFilter({cols}: {self.values.height} values)"


class FilterContext:
    """An immutable set of column filters."""

    __slots__ = ("filters",)

    def __init__(self, filters: tuple[ColumnFilter, ...] = ()) -> None:
        self.filters = filters

    def apply(self, new: Iterable[ColumnFilter]) -> FilterContext:
        """Add filters the way CALCULATE does: each one replaces existing filters on
        its columns. A filter over several columns keeps its other columns."""
        result = list(self.filters)
        for f in new:
            ids = {id(c) for c in f.columns}
            kept: list[ColumnFilter] = []
            for g in result:
                rest = [c for c in g.columns if id(c) not in ids]
                if len(rest) == len(g.columns):
                    kept.append(g)
                elif rest:
                    values = g.values.select([c.name for c in rest]).unique()
                    kept.append(ColumnFilter(g.table, rest, values))
            result = [*kept, f]
        return FilterContext(tuple(result))

    def __repr__(self) -> str:
        return f"FilterContext({list(self.filters)})"


def normalized(column: Column, name: str | None = None) -> pl.Expr:
    """The value DAX compares on. Text compares without case."""
    expr = pl.col(name or column.name)
    if column.data_type is DaxType.STRING:
        return expr.str.to_lowercase()
    return expr


class Storage:
    """The rows of each table, with the hidden blank row, plus cached visibility.

    One Storage serves one evaluation, so cached results never go stale.
    """

    def __init__(self, model: Model) -> None:
        self.model = model
        self._ext: dict[int, pl.DataFrame] = {}
        self._blank: dict[int, int | None] = {}
        self._visible: dict[tuple[int, int], pl.Series | None] = {}
        self._keep: list[Any] = []  # keeps cached keys' objects alive
        self._checked: set[int] = set()

    # --- rows ---------------------------------------------------------------------

    def ext(self, table: Table) -> pl.DataFrame:
        """The table's rows, followed by the blank row if Power BI would add one."""
        key = id(table)
        if key not in self._ext:
            data = table.data
            if table.has_blank_row:
                blank = pl.DataFrame(
                    [pl.Series(c, [None], dtype=data.schema[c]) for c in data.columns]
                )
                self._blank[key] = data.height
                data = pl.concat([data, blank])
            else:
                self._blank[key] = None
            self._ext[key] = data
        return self._ext[key]

    def blank_index(self, table: Table) -> int | None:
        self.ext(table)
        return self._blank[id(table)]

    def real_rows(self, table: Table) -> pl.Series:
        """True for every row except the blank row."""
        ext = self.ext(table)
        mask = pl.Series([True] * ext.height, dtype=pl.Boolean)
        index = self.blank_index(table)
        if index is not None:
            mask = mask.scatter(index, False)
        return mask

    def check_text(self, column: Column) -> None:
        """Refuse text columns whose values differ only by case or trailing spaces.

        Power BI stores one spelling for such values, and which one it keeps depends on
        load order, so results that show these values could differ.
        """
        if column.data_type is not DaxType.STRING or id(column) in self._checked:
            return
        values = column.table.data[column.name].drop_nulls().unique()
        folded = values.str.to_lowercase().str.strip_chars_end(" ").n_unique()
        if folded != values.len():
            raise NotSupportedError(
                f"text values in {column.ref} that differ only by case or trailing spaces"
            )
        self._checked.add(id(column))

    # --- visibility -----------------------------------------------------------------

    def visible(self, table: Table, filters: FilterContext) -> pl.Series | None:
        """A mask over :meth:`ext` rows, or None when nothing filters the table."""
        key = (id(table), id(filters))
        if key not in self._visible:
            self._keep.append(filters)
            self._visible[key] = self._visible_from(table, filters, frozenset(), None)
        return self._visible[key]

    def visible_rows(self, table: Table, filters: FilterContext) -> pl.DataFrame:
        mask = self.visible(table, filters)
        ext = self.ext(table)
        return ext if mask is None else ext.filter(mask)

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
        mask: pl.Series | None = None
        for f in filters.filters:
            if f.table is table:
                m = self._filter_mask(f)
                mask = m if mask is None else mask & m
        for rel in self.model.relationships:
            if not rel.is_active or rel is via:
                continue
            if rel.cardinality is Cardinality.MANY_TO_MANY:
                raise NotSupportedError("many-to-many relationships")
            if rel.from_table is rel.to_table:
                raise NotSupportedError("a relationship from a table to itself")
            if rel.from_table is table:
                other = rel.to_table
            elif rel.to_table is table and rel.cross_filter is CrossFilter.BOTH:
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

    def _filter_mask(self, f: ColumnFilter) -> pl.Series:
        ext = self.ext(f.table)
        keys = [f"k{i}" for i in range(len(f.columns))]
        left = ext.select(
            [normalized(c).alias(k) for c, k in zip(f.columns, keys, strict=True)]
        ).with_row_index("__i")
        right = f.values.select(
            [normalized(c).alias(k) for c, k in zip(f.columns, keys, strict=True)]
        )
        hits = left.join(right, on=keys, how="semi", nulls_equal=True)["__i"]
        return pl.Series([False] * ext.height, dtype=pl.Boolean).scatter(hits, True)

    def _restrict(self, table: Table, rel: Relationship, other_mask: pl.Series) -> pl.Series:
        """Rows of ``table`` allowed by the visible rows of the relationship's other side."""
        self._check_keys(rel)
        many, one = rel.from_column, rel.to_column
        one_ext = self.ext(one.table)
        many_ext = self.ext(many.table)
        one_real = self.real_rows(one.table)
        one_keys = one_ext.select(normalized(one).alias("k"))["k"]
        many_keys = many_ext.select(normalized(many).alias("k"))["k"]
        all_one = one_keys.filter(one_real).drop_nulls()
        unmatched = ~many_keys.is_in(all_one.implode()).fill_null(False)

        if table is many.table:
            visible_one = one_keys.filter(other_mask & one_real).drop_nulls()
            mask = many_keys.is_in(visible_one.implode()).fill_null(False)
            blank = self.blank_index(one.table)
            if blank is not None and bool(other_mask[blank]):
                mask = mask | unmatched
            return mask

        visible_many = many_keys.filter(other_mask)
        mask = one_keys.is_in(visible_many.drop_nulls().implode()).fill_null(False) & one_real
        blank = self.blank_index(one.table)
        if blank is not None and bool((other_mask & unmatched).any()):
            mask = mask.scatter(blank, True)
        return mask

    def _check_keys(self, rel: Relationship) -> None:
        if id(rel) in self._checked:
            return
        for col in (rel.from_column, rel.to_column):
            if col.data_type is DaxType.STRING:
                values = col.table.data[col.name].drop_nulls()
                if (values.str.strip_chars_end(" ") != values).any():
                    raise NotSupportedError(
                        f"relationship keys with trailing spaces ({col.ref}); Power BI "
                        "may match them to keys without the spaces"
                    )
        if rel.cardinality is Cardinality.ONE_TO_ONE:
            one = rel.to_column.table.data.select(normalized(rel.to_column).alias("k"))["k"]
            many = rel.from_column.table.data.select(normalized(rel.from_column).alias("k"))["k"]
            if (~one.drop_nulls().is_in(many.drop_nulls().implode())).any():
                raise NotSupportedError(
                    "a one-to-one relationship where both sides have keys the other lacks"
                )
        self._checked.add(id(rel))

    # --- filters from rows -------------------------------------------------------------

    def row_filter(self, table: Table, index: int, columns: Sequence[Column]) -> ColumnFilter:
        """A filter matching one row's values in ``columns`` (context transition)."""
        names = [c.name for c in columns]
        values = self.ext(table).select(names).slice(index, 1)
        return ColumnFilter(table, columns, values)
