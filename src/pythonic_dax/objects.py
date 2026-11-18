"""Semantic model objects: tables, columns, measures and relationships.

Each object carries the metadata Power BI's semantic model stores for it (description,
format string, display folder, hidden flag and so on). All metadata is optional.
``display_name`` (and a description on relationships) are this library's own
additions: Power BI has no single model-wide property for either.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Any, ClassVar

import polars as pl

from ._names import NameMap
from .errors import ModelError
from .expr import PREC_ATOM, Expr, TableExpr, quote_name
from .sources import Source
from .types import DaxType

if TYPE_CHECKING:
    from .model import Model


class SummarizeBy(StrEnum):
    """A column's default aggregation in visuals (Power BI's "Summarization")."""

    DEFAULT = "default"
    NONE = "none"
    SUM = "sum"
    MIN = "min"
    MAX = "max"
    COUNT = "count"
    AVERAGE = "average"
    DISTINCT_COUNT = "distinct_count"


DATA_CATEGORIES = frozenset(
    {
        "Uncategorized", "Address", "City", "Continent", "Country", "County",
        "Latitude", "Longitude", "Place", "PostalCode", "StateOrProvince",
        "WebUrl", "ImageUrl", "Barcode",
    }
)  # fmt: skip
"""Data categories Power BI offers for columns and measures."""


class _Metadata:
    """Shared metadata behavior: ``set()`` and the ``display_name`` fallback."""

    _settable: ClassVar[frozenset[str]] = frozenset()

    name: str
    description: str | None
    _display_name: str | None
    annotations: dict[str, str]

    @property
    def display_name(self) -> str:
        """The name reports show. Falls back to the model name."""
        return self._display_name or self.name

    @display_name.setter
    def display_name(self, value: str | None) -> None:
        self._display_name = value or None

    def _apply(self, props: dict[str, Any]) -> None:
        unknown = set(props) - self._settable
        if unknown:
            allowed = ", ".join(sorted(self._settable))
            raise TypeError(
                f"{type(self).__name__} has no property {sorted(unknown)[0]!r}. "
                f"Settable properties: {allowed}"
            )
        for key, value in props.items():
            setattr(self, key, value)


class Column(_Metadata, Expr):
    """A column of a model table, and a column reference in DAX expressions."""

    is_column_ref = True

    _settable = frozenset(
        {
            "description", "display_name", "format_string", "display_folder",
            "is_hidden", "data_category", "summarize_by", "sort_by_column", "is_key",
            "is_nullable", "annotations",
        }
    )  # fmt: skip

    def __init__(self, table: Table, name: str, data_type: DaxType) -> None:
        self.table = table
        self.name = name
        self.data_type = data_type
        self.description: str | None = None
        self._display_name: str | None = None
        self.format_string: str | None = None
        self.display_folder: str | None = None
        self.is_hidden = False
        self._data_category: str | None = None
        self._summarize_by = SummarizeBy.SUM if data_type.is_numeric else SummarizeBy.NONE
        self._sort_by_column: Column | None = None
        self.is_key = False
        self.is_nullable = True
        self.annotations: dict[str, str] = {}

    def set(self, **props: Any) -> Column:
        """Set metadata, for example ``col.set(description="A flag for voids.")``."""
        self._apply(props)
        return self

    @property
    def data_category(self) -> str | None:
        return self._data_category

    @data_category.setter
    def data_category(self, value: str | None) -> None:
        self._data_category = _check_data_category(value)

    @property
    def summarize_by(self) -> SummarizeBy:
        return self._summarize_by

    @summarize_by.setter
    def summarize_by(self, value: SummarizeBy | str) -> None:
        try:
            self._summarize_by = SummarizeBy(value)
        except ValueError:
            choices = ", ".join(s.value for s in SummarizeBy)
            raise ModelError(f"summarize_by must be one of: {choices}") from None

    @property
    def sort_by_column(self) -> Column | None:
        return self._sort_by_column

    @sort_by_column.setter
    def sort_by_column(self, value: Column | str | None) -> None:
        if value is None:
            self._sort_by_column = None
            return
        other = self.table[value] if isinstance(value, str) else value
        if other.table is not self.table:
            raise ModelError("a column can only be sorted by a column in the same table")
        if other is self:
            raise ModelError(f"column {self.ref} can't be sorted by itself")
        _check_sort_by(self, other)
        self._sort_by_column = other

    @property
    def values(self) -> pl.Series:
        """The column's stored values."""
        return self.table.data[self.name]

    @property
    def ref(self) -> str:
        """The column as DAX writes it, for example ``Sales[Amount]``."""
        return f"{self.table.ref}[{self.name.replace(']', ']]')}]"

    @property
    def dax_type(self) -> DaxType:
        return self.data_type

    def _dax(self) -> tuple[str, int]:
        return self.ref, PREC_ATOM

    def __repr__(self) -> str:
        return f"Column({self.ref}, {self.data_type.value})"


class Table(_Metadata, TableExpr):
    """A model table: its data, its columns and their metadata. Also a table in DAX."""

    is_table_ref = True

    _settable = frozenset({"description", "display_name", "is_hidden", "annotations"})

    def __init__(self, model: Model, name: str, source: Source) -> None:
        self.model = model
        self.name = name
        self.source = source
        self.description: str | None = None
        self._display_name: str | None = None
        self.is_hidden = False
        self.annotations: dict[str, str] = {}
        self.date_column: Column | None = None
        self._columns: NameMap[Column] = NameMap("column")
        self._data = pl.DataFrame()

    # --- data ---------------------------------------------------------------

    @property
    def data(self) -> pl.DataFrame:
        """The table's rows, stored with DAX data types."""
        return self._data

    @property
    def columns(self) -> list[Column]:
        return list(self._columns)

    def __len__(self) -> int:
        return self._data.height

    @property
    def has_blank_row(self) -> bool:
        """Whether Power BI would add its hidden blank row to this table.

        That happens when this table is the one side of a relationship and some rows
        on the many side have a key with no match here.
        """
        return any(r.to_table is self and r.unmatched_rows > 0 for r in self.model.relationships)

    # --- metadata -----------------------------------------------------------

    def set(self, **props: Any) -> Table:
        """Set metadata, for example ``table.set(description="One row per exam.")``."""
        self._apply(props)
        return self

    def mark_as_date_table(self, column: Column | str) -> Table:
        """Mark this as a date table, like Power BI's "Mark as date table".

        The date column must be Date/Time, with no blanks, no duplicates, no time of
        day, and no gaps between the first and last date.
        """
        col = self[column] if isinstance(column, str) else column
        if col.table is not self:
            raise ModelError(f"{col.ref} is not a column of {self.ref}")
        _check_date_column(col)
        self.date_column = col
        return self

    @property
    def is_date_table(self) -> bool:
        return self.date_column is not None

    # --- column access ------------------------------------------------------

    def __getitem__(self, name: str) -> Column:
        """Look up a column by name, ignoring case like DAX: ``Sales["Amount"]``."""
        try:
            return self._columns[name]
        except KeyError:
            raise KeyError(f"table {self.name!r} has no column {name!r}") from None

    def __getattr__(self, name: str) -> Column:
        # Called only when normal attribute lookup fails: lets ``Sales.Amount`` work.
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self._columns[name]
        except KeyError:
            raise AttributeError(f"table {self.name!r} has no column {name!r}") from None

    def __contains__(self, name: object) -> bool:
        return name in self._columns

    @property
    def ref(self) -> str:
        """The table as DAX writes it: ``Sales``, or ``'Sales Order'`` when quotes are needed."""
        return quote_name(self.name)

    def _dax(self) -> tuple[str, int]:
        return self.ref, PREC_ATOM

    def __repr__(self) -> str:
        return f"Table({self.ref}, {len(self)} rows, {len(self._columns)} columns)"

    # --- loading (used by Model) ---------------------------------------------

    def _load(self, column_types: dict[str, DaxType], first_load: bool) -> None:
        from .types import infer_dax_type, to_dax_series

        raw = self.source.load()
        names = NameMap[str]("column")
        for col_name in raw.columns:
            try:
                names.add(col_name, col_name)
            except KeyError:
                raise ModelError(
                    f"table {self.name!r}: column names must be unique ignoring case, "
                    f"but {col_name!r} appears twice"
                ) from None

        if not first_load:
            missing = [c.name for c in self._columns if c.name not in names]
            if missing:
                raise ModelError(f"table {self.name!r}: the source no longer has {missing}")

        series: list[pl.Series] = []
        for col_name in raw.columns:
            existing = self._columns.get(col_name)
            if existing is not None:
                dax_type = existing.data_type
            else:
                dax_type = column_types.get(col_name) or infer_dax_type(raw.schema[col_name])
            series.append(to_dax_series(raw[col_name], dax_type))
            if existing is None:
                self._columns.add(col_name, Column(self, col_name, dax_type))
        self._data = pl.DataFrame(series)


class Measure(_Metadata, Expr):
    """A named DAX expression evaluated in filter context, and a reference to it."""

    is_measure_ref = True

    _settable = frozenset(
        {
            "description", "display_name", "format_string", "display_folder",
            "is_hidden", "data_category", "annotations",
        }
    )  # fmt: skip

    def __init__(self, model: Model, name: str, expression: Any, table: Table) -> None:
        self.model = model
        self.name = name
        self.expression = expression
        self.table = table
        self.description: str | None = None
        self._display_name: str | None = None
        self.format_string: str | None = None
        self.display_folder: str | None = None
        self.is_hidden = False
        self._data_category: str | None = None
        self.annotations: dict[str, str] = {}

    def set(self, **props: Any) -> Measure:
        """Set metadata, for example ``measure.set(format_string="$#,0.00")``."""
        self._apply(props)
        return self

    @property
    def data_category(self) -> str | None:
        return self._data_category

    @data_category.setter
    def data_category(self, value: str | None) -> None:
        self._data_category = _check_data_category(value)

    @property
    def ref(self) -> str:
        """The measure as DAX writes it, for example ``[Total Sales]``."""
        return "[" + self.name.replace("]", "]]") + "]"

    @property
    def dax_type(self) -> DaxType | None:
        return getattr(self.expression, "dax_type", None)

    def _dax(self) -> tuple[str, int]:
        return self.ref, PREC_ATOM

    def definition(self) -> str:
        """The measure as a DAX definition: ``MEASURE Sales[Total] = SUM(Sales[Amount])``."""
        return f"MEASURE {self.table.ref}{self.ref} = {self.expression.to_dax()}"

    def __repr__(self) -> str:
        return f"Measure({self.ref}, home table {self.table.ref})"


class Cardinality(StrEnum):
    MANY_TO_ONE = "many_to_one"
    ONE_TO_ONE = "one_to_one"
    MANY_TO_MANY = "many_to_many"


class CrossFilter(StrEnum):
    """Which way filters flow across a relationship."""

    SINGLE = "single"
    """From the one side to the many side only."""
    BOTH = "both"
    """Both ways."""


class Relationship(_Metadata):
    """A relationship from a many-side column to a one-side column."""

    _settable = frozenset({"description", "display_name", "annotations"})

    def __init__(
        self,
        model: Model,
        from_column: Column,
        to_column: Column,
        cardinality: Cardinality,
        cross_filter: CrossFilter,
        is_active: bool,
        name: str,
        rely_on_referential_integrity: bool,
    ) -> None:
        self.model = model
        self.from_column = from_column
        self.to_column = to_column
        self.cardinality = cardinality
        self.cross_filter = cross_filter
        self.is_active = is_active
        self.name = name
        self.rely_on_referential_integrity = rely_on_referential_integrity
        self.description: str | None = None
        self._display_name: str | None = None
        self.annotations: dict[str, str] = {}
        self._unmatched: int | None = None

    @property
    def from_table(self) -> Table:
        """The many side (for one-to-one, the side listed first)."""
        return self.from_column.table

    @property
    def to_table(self) -> Table:
        """The one side."""
        return self.to_column.table

    def set(self, **props: Any) -> Relationship:
        self._apply(props)
        return self

    @property
    def unmatched_rows(self) -> int:
        """Rows on the many side whose key has no match on the one side.

        Blank keys count too. These rows belong to the one side's hidden blank row.
        """
        if self._unmatched is None:
            self._unmatched = _count_unmatched(self.from_column, self.to_column)
        return self._unmatched

    def _invalidate(self) -> None:
        self._unmatched = None

    def __repr__(self) -> str:
        arrow = "<->" if self.cross_filter is CrossFilter.BOTH else "<-"
        state = "" if self.is_active else ", inactive"
        return (
            f"Relationship({self.from_column.ref} {arrow} {self.to_column.ref}, "
            f"{self.cardinality.value}{state})"
        )


# --- validation helpers -------------------------------------------------------


def _check_data_category(value: str | None) -> str | None:
    if value is None or value in DATA_CATEGORIES:
        return value
    choices = ", ".join(sorted(DATA_CATEGORIES))
    raise ModelError(f"unknown data category {value!r}. Power BI's categories: {choices}")


def match_key(column: Column) -> pl.Expr:
    """An expression giving the value DAX compares on: text is compared ignoring case."""
    expr = pl.col(column.name)
    if column.data_type is DaxType.STRING:
        return expr.str.to_lowercase()
    return expr


def duplicate_count(column: Column) -> int:
    """How many values appear more than once, comparing text without case."""
    keys = column.table.data.select(match_key(column).alias("k"))["k"]
    return int(keys.len() - keys.n_unique())


def _count_unmatched(many: Column, one: Column) -> int:
    many_keys = many.table.data.select(match_key(many).alias("k"))
    one_keys = one.table.data.select(match_key(one).alias("k")).drop_nulls().unique()
    unmatched = many_keys.join(one_keys, on="k", how="anti", nulls_equal=False)
    return unmatched.height


def _check_sort_by(column: Column, sort_by: Column) -> None:
    pairs = column.table.data.select(
        match_key(column).alias("v"), match_key(sort_by).alias("s")
    ).unique()
    if pairs["v"].n_unique() != pairs.height:
        raise ModelError(
            f"can't sort {column.ref} by {sort_by.ref}: there can't be more than one value "
            f"in {sort_by.ref} for the same value in {column.ref}"
        )


def _check_date_column(col: Column) -> None:
    if col.data_type is not DaxType.DATETIME:
        raise ModelError(f"the date column {col.ref} must be of type Date/Time")
    values = col.values
    if values.null_count():
        raise ModelError(f"the date column {col.ref} can't contain blanks")
    if values.n_unique() != values.len():
        raise ModelError(f"the date column {col.ref} can't contain duplicate dates")
    if values.len() == 0:
        raise ModelError(f"the date column {col.ref} has no dates")
    if (values.dt.truncate("1d") != values).any():
        raise ModelError(f"the date column {col.ref} can't contain times of day")
    days = values.dt.date()
    span_days = (days.max() - days.min()).days + 1  # type: ignore[operator,union-attr]
    if span_days != values.len():
        raise ModelError(f"the date column {col.ref} must have no gaps between dates")
