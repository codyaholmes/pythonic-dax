from __future__ import annotations

import datetime as dt
from decimal import Decimal

import polars as pl
import pytest

from pythonic_dax import DaxType, ModelError
from pythonic_dax.types import infer_dax_type, to_dax_series


@pytest.mark.parametrize(
    ("dtype", "expected"),
    [
        (pl.Int8(), DaxType.INTEGER),
        (pl.UInt32(), DaxType.INTEGER),
        (pl.Int64(), DaxType.INTEGER),
        (pl.Float32(), DaxType.DECIMAL),
        (pl.Decimal(10, 2), DaxType.DECIMAL),  # Power BI's default for source decimals
        (pl.Duration("us"), DaxType.DECIMAL),
        (pl.Date(), DaxType.DATETIME),
        (pl.Time(), DaxType.DATETIME),
        (pl.Datetime("ns", "UTC"), DaxType.DATETIME),
        (pl.String(), DaxType.STRING),
        (pl.Categorical(), DaxType.STRING),
        (pl.Boolean(), DaxType.BOOLEAN),
        (pl.Null(), DaxType.STRING),
    ],
)
def test_infer_dax_type(dtype: pl.DataType, expected: DaxType) -> None:
    assert infer_dax_type(dtype) is expected


@pytest.mark.parametrize("dtype", [pl.List(pl.Int64), pl.Struct({"a": pl.Int64}), pl.Binary()])
def test_unsupported_types_are_rejected(dtype: pl.DataType) -> None:
    with pytest.raises(ModelError):
        infer_dax_type(dtype)


def test_date_becomes_midnight_datetime() -> None:
    s = to_dax_series(pl.Series("d", [dt.date(2026, 1, 31), None]), DaxType.DATETIME)
    assert s.dtype == pl.Datetime("us")
    assert s.to_list() == [dt.datetime(2026, 1, 31), None]


def test_time_lands_on_power_bi_base_date() -> None:
    s = to_dax_series(pl.Series("t", [dt.time(13, 30)]), DaxType.DATETIME)
    assert s.to_list() == [dt.datetime(1899, 12, 30, 13, 30)]


def test_time_zone_keeps_wall_clock_time() -> None:
    src = pl.Series("t", [dt.datetime(2026, 1, 1, 9, 0)]).dt.replace_time_zone("America/Chicago")
    assert to_dax_series(src, DaxType.DATETIME).to_list() == [dt.datetime(2026, 1, 1, 9, 0)]


def test_duration_becomes_days() -> None:
    s = to_dax_series(pl.Series("d", [dt.timedelta(hours=36)]), DaxType.DECIMAL)
    assert s.to_list() == [1.5]


def test_currency_rounds_to_four_places() -> None:
    s = to_dax_series(pl.Series("p", [1.23456, 2.0]), DaxType.CURRENCY)
    assert s.dtype == pl.Decimal(19, 4)
    assert s.to_list() == [Decimal("1.2346"), Decimal("2.0000")]


def test_unsigned_overflow_is_an_error_not_a_wraparound() -> None:
    with pytest.raises(ModelError, match="doesn't fit"):
        to_dax_series(pl.Series("k", [2**64 - 1], dtype=pl.UInt64), DaxType.INTEGER)


def test_text_to_number_is_rejected() -> None:
    with pytest.raises(ModelError, match="can't convert"):
        to_dax_series(pl.Series("x", ["1"]), DaxType.INTEGER)
