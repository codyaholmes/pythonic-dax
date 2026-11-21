"""Phase 9: the fast paths (``_vector.py``, ``_batch.py``) give exactly what the
row-by-row, group-by-group engine gives.

The golden cases check both paths against Power BI (see test_golden.py). These tests
cover shapes the golden models don't: single-direction models (most golden models
filter both ways, which uses the per-group path), snowflakes, blank rows, BLANK
values, currency, RELATED, USERELATIONSHIP and time intelligence.
"""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Callable, Iterator
from decimal import Decimal
from typing import Any

import polars as pl
import pytest

import pythonic_dax as pdax
from pythonic_dax import (
    ALL,
    AVERAGE,
    AVERAGEX,
    CALCULATE,
    COUNT,
    COUNTBLANK,
    COUNTROWS,
    DISTINCTCOUNT,
    DIVIDE,
    IF,
    ISBLANK,
    MAX,
    MAXX,
    MIN,
    MINX,
    RELATED,
    ROLLUPADDISSUBTOTAL,
    SAMEPERIODLASTYEAR,
    SUM,
    SUMMARIZECOLUMNS,
    SUMX,
    TOTALYTD,
    TREATAS,
    USERELATIONSHIP,
    VAR,
    _batch,
    _vector,
)


@pytest.fixture
def fast_and_slow() -> Iterator[Callable[[pdax.Model, Any], pl.DataFrame]]:
    """Evaluate a query both ways, check they agree, and return the fast result."""

    def run(model: pdax.Model, query: Any) -> pl.DataFrame:
        _vector.ENABLED, _batch.STRICT = False, False
        slow = model.evaluate(query)
        _vector.ENABLED, _batch.STRICT = True, True
        fast = model.evaluate(query)
        assert_same(fast, slow)
        return fast

    yield run
    _vector.ENABLED, _batch.STRICT = True, False


def _key(row: tuple[Any, ...]) -> tuple[Any, ...]:
    return tuple((v is None, str(type(v)), str(v)) for v in row)


def assert_same(fast: pl.DataFrame, slow: pl.DataFrame) -> None:
    assert fast.columns == slow.columns
    assert fast.schema == slow.schema
    a, b = sorted(fast.rows(), key=_key), sorted(slow.rows(), key=_key)
    assert len(a) == len(b), (a, b)
    for x, y in zip(a, b, strict=True):
        for p, q in zip(x, y, strict=True):
            if isinstance(p, float) and isinstance(q, float):
                assert (math.isnan(p) and math.isnan(q)) or p == pytest.approx(q, rel=1e-12)
            else:
                assert p == q and type(p) is type(q), (x, y)


@pytest.fixture
def model() -> pdax.Model:
    """A snowflake with unmatched keys (blank rows), BLANK values and currency."""
    m = pdax.Model()
    category = m.add_table(
        "Category",
        pl.DataFrame({"CategoryKey": [1, 2, 3], "Category": ["Bikes", "Parts", "Empty"]}),
    )
    product = m.add_table(
        "Product",
        pl.DataFrame(
            {
                "ProductKey": [1, 2, 3, 4, 5],
                "Product": ["Road", "Trail", "Chain", "Bell", "Orphan"],
                "CategoryKey": [1, 1, 2, 2, 9],
                "Price": [Decimal("999.99"), Decimal("1500"), Decimal("19.5"), None, Decimal("5")],
            }
        ),
        columns={"Price": {"data_type": "currency"}},
    )
    customer = m.add_table(
        "Customer",
        pl.DataFrame({"CustomerKey": [1, 2, 3], "Region": ["North", "South", "North"]}),
    )
    dates = m.add_table(
        "Date",
        pl.DataFrame(
            {"Date": pl.date_range(dt.date(2023, 1, 1), dt.date(2024, 12, 31), eager=True)}
        )
        .with_columns(Year=pl.col("Date").dt.year().cast(pl.Int64))
        .with_columns(Month=pl.col("Date").dt.strftime("%Y-%m")),
    )
    dates.mark_as_date_table("Date")
    rows = 400
    sales = m.add_table(
        "Sales",
        pl.DataFrame(
            {
                "OrderID": [i // 3 for i in range(rows)],
                "OrderDate": [
                    dt.date(2023, 1, 1) + dt.timedelta(days=(i * 37) % 730) for i in range(rows)
                ],
                "ShipDate": [
                    dt.date(2023, 1, 5) + dt.timedelta(days=(i * 37) % 720) for i in range(rows)
                ],
                "ProductKey": [(i % 6) + 1 if i % 17 else None for i in range(rows)],
                "CustomerKey": [(i % 4) + 1 for i in range(rows)],
                "Qty": [(i % 5) + 1 if i % 11 else None for i in range(rows)],
                "Discount": [((i % 7) / 10) for i in range(rows)],
                "UnitPrice": [Decimal(f"{(i % 50) + 0.25:.2f}") for i in range(rows)],
            }
        ),
        columns={"UnitPrice": {"data_type": "currency"}},
    )
    m.add_relationship(sales["ProductKey"], product["ProductKey"])
    m.add_relationship(product["CategoryKey"], category["CategoryKey"])
    m.add_relationship(sales["CustomerKey"], customer["CustomerKey"])
    m.add_relationship(sales["OrderDate"], dates["Date"])
    m.add_relationship(sales["ShipDate"], dates["Date"], active=False)
    return m


def measures(m: pdax.Model) -> dict[str, Any]:
    s, p, d = m.table("Sales"), m.table("Product"), m.table("Date")
    revenue = SUMX(s, s["Qty"] * s["UnitPrice"])
    return {
        "Revenue": revenue,
        "List": SUMX(s, s["Qty"] * RELATED(p["Price"])),
        "Discounted": SUMX(s, s["Qty"] * s["UnitPrice"] * (1 - s["Discount"])),
        "Qty": SUM(s["Qty"]),
        "AvgQty": AVERAGE(s["Qty"]),
        "AvgLine": AVERAGEX(s, s["Qty"] * 2),
        "MinPrice": MIN(s["UnitPrice"]),
        "MaxDate": MAX(s["OrderDate"]),
        "MinX": MINX(s, s["Qty"] - 1),
        "MaxX": MAXX(s, s["UnitPrice"] + s["UnitPrice"]),
        "Lines": COUNTROWS(s),
        "QtyCount": COUNT(s["Qty"]),
        "NoQty": COUNTBLANK(s["Qty"]),
        "Orders": DISTINCTCOUNT(s["OrderID"]),
        "Products": COUNTROWS(p),
        "Share": DIVIDE(revenue, CALCULATE(revenue, ALL(p))),
        "PY": CALCULATE(revenue, SAMEPERIODLASTYEAR(d["Date"])),
        "YTD": TOTALYTD(SUM(s["Qty"]), d["Date"]),
        "Shipped": CALCULATE(SUM(s["Qty"]), USERELATIONSHIP(s["ShipDate"], d["Date"])),
        "Big": IF(SUM(s["Qty"]) > 100, 1, 0),
        "Var": VAR(q=SUM(s["Qty"])).RETURN(lambda v: IF(ISBLANK(v.q), -1, v.q * 2)),
    }


GROUPINGS: dict[str, Callable[[pdax.Model], list[Any]]] = {
    "none": lambda m: [],
    "category": lambda m: [m.table("Category")["Category"]],
    "product": lambda m: [m.table("Product")["Product"]],
    "region x year": lambda m: [m.table("Customer")["Region"], m.table("Date")["Year"]],
    "fact column": lambda m: [m.table("Sales")["CustomerKey"]],
    "month rollup": lambda m: [
        ROLLUPADDISSUBTOTAL(m.table("Date")["Year"], "Y", m.table("Date")["Month"], "M")
    ],
}


@pytest.mark.parametrize("grouping", list(GROUPINGS))
def test_groupings_match(model: pdax.Model, fast_and_slow: Any, grouping: str) -> None:
    before = _batch.used
    group = GROUPINGS[grouping](model)
    fast_and_slow(model, SUMMARIZECOLUMNS(*group, measures(model)))
    assert _batch.used > before  # the fast path was taken


def test_filter_tables_match(model: pdax.Model, fast_and_slow: Any) -> None:
    c, d = model.table("Category"), model.table("Date")
    query = SUMMARIZECOLUMNS(
        model.table("Product")["Product"],
        TREATAS(["Bikes"], c["Category"]),
        TREATAS([2024], d["Year"]),
        measures(model),
    )
    fast_and_slow(model, query)


def test_group_by_columns_of_interleaved_tables(model: pdax.Model, fast_and_slow: Any) -> None:
    """Group-by columns from different tables, in any order (this used to fail)."""
    p, d = model.table("Product"), model.table("Date")
    query = SUMMARIZECOLUMNS(
        p["Product"], d["Year"], p["CategoryKey"], {"Q": SUM(model.table("Sales")["Qty"])}
    )
    result = fast_and_slow(model, query)
    assert result.columns == ["Product[Product]", "Date[Year]", "Product[CategoryKey]", "[Q]"]


def test_two_way_relationship_uses_per_group_path(model: pdax.Model, fast_and_slow: Any) -> None:
    s, c = model.table("Sales"), model.table("Customer")
    rel = next(r for r in model.relationships if r.to_table is c)
    rel.cross_filter = pdax.CrossFilter.BOTH
    before = _batch.used
    fast_and_slow(model, SUMMARIZECOLUMNS(model.table("Product")["Product"], {"N": COUNTROWS(c)}))
    assert _batch.used == before
    fast_and_slow(model, SUMMARIZECOLUMNS(c["Region"], {"R": SUMX(s, s["Qty"] * 3)}))


def test_iterators_outside_summarizecolumns(model: pdax.Model, fast_and_slow: Any) -> None:
    s = model.table("Sales")
    query = pdax.ROW(
        "Sum",
        SUMX(s, s["Qty"] * s["UnitPrice"]),
        "Filtered",
        SUMX(pdax.FILTER(s, s["Qty"] > 2), -s["Qty"] + 1.5),
        "Related",
        SUMX(s, RELATED(model.table("Product")["Price"]) * 2),
        "Avg",
        AVERAGEX(s, s["Discount"] * s["Qty"]),
    )
    fast_and_slow(model, query)


def test_cache_follows_model_changes(model: pdax.Model, fast_and_slow: Any) -> None:
    """Cached lookups and row values are rebuilt when the data or relationships change."""
    s, p = model.table("Sales"), model.table("Product")
    query = SUMMARIZECOLUMNS(p["Product"], {"L": SUMX(s, s["Qty"] * RELATED(p["Price"]))})
    first = fast_and_slow(model, query)
    model.add_column(s, "Double", s["Qty"] * 2)
    model.add_relationship(model.add_table("Extra", pl.DataFrame({"K": [1]}))["K"], p["ProductKey"])
    assert_same(fast_and_slow(model, query), first)
