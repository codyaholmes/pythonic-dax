"""A report layer that works with any front end: slicers, visuals, labels and formatting.

This module turns report concepts into DAX queries the way Power BI's report canvas
does, and returns plain Python and Polars results. It doesn't import any UI framework:
``pythonic_dax.streamlit`` draws these results with Streamlit, and the same objects
can feed Dash, FastAPI, a notebook or anything else.

    report = Report(model)
    report.select(Product["Category"], ["Bikes"])          # a slicer selection
    report.select_range(Date["Date"], start, end)           # a date range slicer
    card = report.card(total_sales)                         # Card: label, value, text
    visual = report.table([Product["Color"]], [total_sales])  # rows with a total row

How the queries are built follows Power BI:

* A slicer selection is a filter table on its column, ``TREATAS({values}, <column>)``,
  or ``FILTER(ALL(<column>), <column> >= start && <column> <= end)`` for a range.
* A visual is ``SUMMARIZECOLUMNS(<row columns>, <filters>, <measures>)``. Its total row
  comes from ``ROLLUPADDISSUBTOTAL``, so a total is the measure evaluated over all the
  rows (as in Power BI), not a sum of the rows shown.
* A selection made in one visual (cross-filtering) filters every other visual, but not
  the visual it came from.

Values are formatted with each field's format string, through the same code as DAX
``FORMAT``. Labels use display names, and tooltips use descriptions.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import polars as pl

from . import _format
from ._special import ROLLUPADDISSUBTOTAL, SUMMARIZECOLUMNS
from .errors import NotSupportedError
from .expr import TableExpr
from .model import Model
from .objects import Column, Measure

__all__ = ["Card", "Field", "Report", "Selection", "Visual", "format_value"]

Value = Any
ValueSpec = Measure | tuple[str, Any]  # a measure, or (name, expression)


# --- fields and formatting --------------------------------------------------------------


@dataclass(frozen=True)
class Field:
    """A column or measure as a report shows it.

    ``key`` is the column name in a visual's data (``Product[Color]`` or ``[Total]``),
    ``label`` the display name, and ``tooltip`` the description (or None).
    """

    key: str
    label: str
    tooltip: str | None
    format_string: str | None
    kind: str  # "column" or "measure"
    folder: str | None = None
    numeric: bool = False


def _column_field(column: Column) -> Field:
    return Field(
        key=f"{column.table.name}[{column.name}]",
        label=column.display_name,
        tooltip=column.description,
        format_string=column.format_string,
        kind="column",
        folder=column.display_folder,
        numeric=column.data_type.is_numeric,
    )


def _value_field(spec: ValueSpec) -> tuple[str, Any, Field]:
    if isinstance(spec, Measure):
        f = Field(
            key=f"[{spec.name}]",
            label=spec.display_name,
            tooltip=spec.description,
            format_string=spec.format_string,
            kind="measure",
            folder=spec.display_folder,
            numeric=True,
        )
        return spec.name, spec, f
    name, expression = spec
    return name, expression, Field(f"[{name}]", name, None, None, "measure", numeric=True)


def format_value(value: Value, format_string: str | None, culture: str = "en-US") -> str:
    """A value as a report shows it: with its format string when it has one, else as
    DAX converts it to text. BLANK shows as empty text."""
    if value is None:
        return ""
    if format_string:
        return _format.format_value(value, format_string, culture)
    if culture.lower() != "en-us":
        raise NotSupportedError(f"formatting values for the culture {culture!r} (only en-US)")
    return _format.to_text(value)


# --- selections -------------------------------------------------------------------------


@dataclass(frozen=True)
class Selection:
    """A filter on one column: a list of values, or a range from ``start`` to ``end``
    (either may be None for an open end). ``source`` names the visual it came from, for
    cross-filtering; slicers have no source."""

    column: Column
    values: tuple[Value, ...] | None = None
    start: Value = None
    end: Value = None
    source: str | None = None

    def filter_table(self) -> TableExpr:
        from ._functions import ALL, FILTER, TREATAS

        if self.values is not None:
            return TREATAS(list(self.values), self.column)
        col = self.column
        if self.start is not None and self.end is not None:
            condition = (col >= self.start) & (col <= self.end)
        elif self.start is not None:
            condition = col >= self.start
        else:
            condition = col <= self.end
        return FILTER(ALL(col), condition)


# --- visual results ---------------------------------------------------------------------


@dataclass
class Card:
    """One value, as a card visual shows it."""

    field: Field
    value: Value
    text: str

    @property
    def label(self) -> str:
        return self.field.label

    @property
    def tooltip(self) -> str | None:
        return self.field.tooltip


@dataclass
class Visual:
    """The rows of a table, matrix or chart visual.

    ``data`` holds the raw values, one column per field (named by ``Field.key``), with
    the total row last when there is one. ``text`` holds the same values formatted for
    display, with columns named by label. ``total`` is the index of the total row, or
    None.
    """

    fields: list[Field]
    data: pl.DataFrame
    text: pl.DataFrame
    total: int | None = None
    rows: list[Field] = field(default_factory=list)
    values: list[Field] = field(default_factory=list)

    def body(self) -> pl.DataFrame:
        """The data without the total row."""
        return self.data if self.total is None else self.data.head(self.total)

    def row_values(self, index: int) -> dict[str, Value]:
        """The row columns' values of one data row, by field key."""
        return {f.key: self.data[f.key][index] for f in self.rows}


# --- the report -------------------------------------------------------------------------


class Report:
    """The filter state of a report page and the visuals evaluated in it."""

    def __init__(self, model: Model) -> None:
        self.model = model
        self.selections: list[Selection] = []

    # --- filters --------------------------------------------------------------------

    def select(
        self, column: Column, values: Iterable[Value] | None, *, source: str | None = None
    ) -> None:
        """Filter a column to some values, like a slicer or a click on a visual.
        ``None`` (or an empty list, as in a slicer with nothing picked) clears it."""
        self.clear(column, source=source)
        picked = tuple(values) if values is not None else ()
        if picked:
            self.selections.append(Selection(column, picked, source=source))

    def select_range(
        self, column: Column, start: Value, end: Value, *, source: str | None = None
    ) -> None:
        """Filter a column to a range, like a between slicer. ``None`` leaves an end open;
        both None clears the filter. Dates are taken as midnight."""
        self.clear(column, source=source)
        start, end = _as_datetime(start), _as_datetime(end)
        if start is not None or end is not None:
            self.selections.append(Selection(column, None, start, end, source))

    def clear(self, column: Column | None = None, *, source: str | None = None) -> None:
        """Remove the selections on a column (all of them when ``column`` is None) made
        by ``source`` (a visual's name; None means slicers)."""
        self.selections = [
            s
            for s in self.selections
            if not ((column is None or s.column is column) and s.source == source)
        ]

    def filter_tables(self, exclude: str | None = None) -> list[TableExpr]:
        """The filter tables for a visual: every selection except those made by the
        visual named ``exclude``."""
        return [
            s.filter_table() for s in self.selections if s.source is None or s.source != exclude
        ]

    # --- visuals --------------------------------------------------------------------

    def card(self, value: ValueSpec, *, name: str | None = None) -> Card:
        """A card: one measure evaluated under the report's filters."""
        key, expression, f = _value_field(value)
        query = SUMMARIZECOLUMNS(*self.filter_tables(name), {key: expression})
        frame = self.model.evaluate(query)
        result = frame[f.key][0] if frame.height else None
        return Card(f, result, format_value(result, f.format_string, self.model.culture))

    def table(
        self,
        rows: Sequence[Column],
        values: Sequence[ValueSpec],
        *,
        totals: bool = True,
        name: str | None = None,
        sort_by: Column | ValueSpec | None = None,
        descending: bool = False,
    ) -> Visual:
        """A table or chart: one row per combination of the row columns that has a
        value, the measures as columns, and a total row at the end.

        Rows are sorted by ``sort_by`` (a row column or a value), else by the row
        columns, using each column's sort-by column as Power BI does.
        """
        if not values:
            raise TypeError("a visual needs at least one value (measure)")
        row_fields = [_column_field(c) for c in rows]
        specs = [_value_field(v) for v in values]
        named = {key: expression for key, expression, _ in specs}
        value_fields = [f for _, _, f in specs]
        filters = self.filter_tables(name)

        flags: list[str] = []
        if rows and totals:
            pairs: list[Any] = []
            for i, c in enumerate(rows):
                flags.append(f"__total{i}")
                pairs += [c, flags[-1]]
            group: list[Any] = [ROLLUPADDISSUBTOTAL(*pairs)]
        else:
            group = list(rows)
        frame = self.model.evaluate(SUMMARIZECOLUMNS(*group, *filters, named))

        total_row: pl.DataFrame | None = None
        if flags:
            is_grand = pl.all_horizontal([pl.col(f"[{f}]") for f in flags])
            is_detail = ~pl.any_horizontal([pl.col(f"[{f}]") for f in flags])
            total_row = frame.filter(is_grand).drop([f"[{f}]" for f in flags])
            frame = frame.filter(is_detail).drop([f"[{f}]" for f in flags])
        keys = [f.key for f in row_fields] + [f.key for f in value_fields]
        frame = self._sorted(frame.select(keys), rows, row_fields, sort_by, descending)
        total: int | None = None
        if total_row is not None:
            total = frame.height
            if total_row.height == 0:
                total_row = pl.DataFrame([pl.Series(k, [None], frame.schema[k]) for k in keys])
            frame = pl.concat([frame, total_row.select(keys).cast(frame.schema)])

        fields = row_fields + value_fields
        text = self._text(frame, fields, total)
        return Visual(fields, frame, text, total, row_fields, value_fields)

    def choices(self, column: Column) -> list[Value]:
        """The values a slicer on this column offers: all its values (BLANK last), in
        the order of its sort-by column."""
        values = self.model.table(column.table.name).data[column.name].unique().to_list()
        order = _sort_keys(column)
        present = [v for v in values if v is not None]
        present.sort(key=lambda v: order.get(v, v) if order else v)
        return present + ([None] if None in values else [])

    def fields(self) -> list[Field]:
        """The fields a field picker offers: visible columns and measures, by table,
        with their display folders."""
        out: list[Field] = []
        for table in self.model.tables:
            if table.is_hidden:
                continue
            out += [_column_field(c) for c in table.columns if not c.is_hidden]
        out += [_value_field(m)[2] for m in self.model.measures if not m.is_hidden]
        return out

    # --- helpers --------------------------------------------------------------------

    def _sorted(
        self,
        frame: pl.DataFrame,
        rows: Sequence[Column],
        row_fields: list[Field],
        sort_by: Column | ValueSpec | None,
        descending: bool,
    ) -> pl.DataFrame:
        if frame.height == 0:
            return frame
        if sort_by is None:
            by = list(zip(rows, row_fields, strict=True))
        elif isinstance(sort_by, Column):
            by = [(c, f) for c, f in zip(rows, row_fields, strict=True) if c is sort_by]
            if not by:
                raise ValueError(f"{sort_by.ref} isn't one of the visual's row columns")
        else:
            key = _value_field(sort_by)[2].key
            return frame.sort(key, descending=descending, nulls_last=True)
        exprs = []
        for column, f in by:
            order = _sort_keys(column)
            if order:
                exprs.append(pl.col(f.key).replace_strict(order, default=None))
            else:
                exprs.append(pl.col(f.key))
        return frame.sort(exprs, descending=descending, nulls_last=True)

    def _text(self, frame: pl.DataFrame, fields: list[Field], total: int | None) -> pl.DataFrame:
        culture = self.model.culture
        columns: dict[str, list[str]] = {}
        for i, f in enumerate(fields):
            texts = [format_value(v, f.format_string, culture) for v in frame[f.key].to_list()]
            if total is not None and i == 0 and not texts[total]:
                texts[total] = "Total"
            columns[_unique(f.label, columns)] = texts
        return pl.DataFrame(columns, schema=dict.fromkeys(columns, pl.String))


def _unique(label: str, taken: Mapping[str, Any]) -> str:
    name, n = label, 2
    while name in taken:
        name, n = f"{label} ({n})", n + 1
    return name


def _sort_keys(column: Column) -> dict[Value, Value]:
    """Each value of a column mapped to its sort-by column's value (empty when the
    column has no sort-by column)."""
    by = column.sort_by_column
    if by is None:
        return {}
    pairs = column.table.data.select(column.name, by.name).unique(subset=column.name)
    return dict(zip(pairs[column.name].to_list(), pairs[by.name].to_list(), strict=True))


def _as_datetime(value: Value) -> Value:
    if isinstance(value, dt.date) and not isinstance(value, dt.datetime):
        return dt.datetime(value.year, value.month, value.day)
    return value
