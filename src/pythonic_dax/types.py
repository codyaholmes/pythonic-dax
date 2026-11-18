"""DAX data types and how source data is converted to them.

Power BI stores every column as one of six data types. When data is loaded, each
source type is converted the way Power BI's import does it. Types that Power BI can't
hold in a model column (lists, structs, binary) are rejected with a clear error.
"""

from __future__ import annotations

import datetime as dt
from enum import StrEnum

import polars as pl

from .errors import ModelError


class DaxType(StrEnum):
    """The data types a Power BI model column can have."""

    INTEGER = "integer"
    """Whole Number: a 64-bit integer."""
    DECIMAL = "decimal"
    """Decimal Number: a 64-bit double."""
    CURRENCY = "currency"
    """Fixed Decimal Number: exactly 4 decimal places."""
    DATETIME = "datetime"
    """Date/Time. A Date is a Date/Time at midnight."""
    STRING = "string"
    """Text. Compared without regard to case."""
    BOOLEAN = "boolean"
    """True/False."""

    @property
    def polars_dtype(self) -> pl.DataType:
        """The Polars dtype used to store values of this type."""
        return _STORAGE[self]

    @property
    def is_numeric(self) -> bool:
        return self in (DaxType.INTEGER, DaxType.DECIMAL, DaxType.CURRENCY)


_STORAGE: dict[DaxType, pl.DataType] = {
    DaxType.INTEGER: pl.Int64(),
    DaxType.DECIMAL: pl.Float64(),
    DaxType.CURRENCY: pl.Decimal(precision=19, scale=4),
    DaxType.DATETIME: pl.Datetime("us"),
    DaxType.STRING: pl.String(),
    DaxType.BOOLEAN: pl.Boolean(),
}

# Power BI keeps times of day as Date/Time values on its base date.
DAX_BASE_DATE = dt.date(1899, 12, 30)

_MICROSECONDS_PER_DAY = 86_400_000_000


def infer_dax_type(dtype: pl.DataType) -> DaxType:
    """The DAX type Power BI gives a column of this source type by default.

    Source decimals become Decimal Number (a double), as in Power BI. Ask for
    ``DaxType.CURRENCY`` explicitly to get Fixed Decimal Number.
    """
    if dtype == pl.Boolean:
        return DaxType.BOOLEAN
    if dtype.is_integer():
        return DaxType.INTEGER
    if dtype.is_float() or isinstance(dtype, (pl.Decimal, pl.Duration)):
        return DaxType.DECIMAL
    if isinstance(dtype, pl.Datetime) or dtype in (pl.Date, pl.Time):
        return DaxType.DATETIME
    if dtype in (pl.String, pl.Categorical) or isinstance(dtype, pl.Enum):
        return DaxType.STRING
    if dtype == pl.Null:
        # A column with no values at all. Power BI types it as Text.
        return DaxType.STRING
    raise ModelError(f"source type {dtype} can't be stored in a DAX column")


def to_dax_series(series: pl.Series, dax_type: DaxType) -> pl.Series:
    """Convert a source column to the storage dtype of ``dax_type``.

    Raises :class:`ModelError` when a value can't be represented, rather than silently
    changing it (for example, an unsigned integer too big for a Whole Number).
    """
    name = series.name
    source = series.dtype
    target = dax_type.polars_dtype

    if source == target:
        return series

    if dax_type is DaxType.DATETIME:
        if source == pl.Date:
            return series.cast(target)
        if source == pl.Time:
            base = pl.lit(dt.datetime.combine(DAX_BASE_DATE, dt.time()))
            frame = pl.DataFrame({name: series})
            return frame.select((base + pl.col(name).cast(pl.Duration("us"))).alias(name))[name]
        if isinstance(source, pl.Datetime):
            # Power BI has no time zones: keep the wall-clock time as loaded.
            if source.time_zone is not None:
                series = series.dt.replace_time_zone(None)
            return series.cast(target)

    if dax_type is DaxType.DECIMAL and isinstance(source, pl.Duration):
        # Power BI loads a duration as a number of days.
        return (series.dt.total_microseconds() / _MICROSECONDS_PER_DAY).rename(name)

    if dax_type is DaxType.STRING and (source == pl.Categorical or isinstance(source, pl.Enum)):
        return series.cast(pl.String)

    if dax_type is DaxType.CURRENCY and (source.is_numeric() or isinstance(source, pl.Decimal)):
        # Fixed Decimal Number rounds to 4 decimal places.
        if source.is_float():
            series = series.round(4)
        return _strict_cast(series, target, dax_type)

    if source == pl.Null:
        return series.cast(target)

    if dax_type.is_numeric and (source.is_numeric() or isinstance(source, pl.Decimal)):
        return _strict_cast(series, target, dax_type)

    if dax_type is DaxType.INTEGER and source == pl.Boolean:
        return series.cast(target)

    raise ModelError(f"column {name!r}: can't convert {source} to DAX type {dax_type.value}")


def _strict_cast(series: pl.Series, target: pl.DataType, dax_type: DaxType) -> pl.Series:
    try:
        return series.cast(target, strict=True)
    except pl.exceptions.InvalidOperationError as exc:
        raise ModelError(
            f"column {series.name!r}: a value doesn't fit DAX type {dax_type.value}"
        ) from exc
