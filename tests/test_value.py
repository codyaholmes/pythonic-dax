"""Unit tests for Model.value(), the one-value helper.

What the values are is pinned by golden cases in ``tests/golden/cases/value``; these
tests cover how arguments are read and the errors.
"""

from __future__ import annotations

import polars as pl
import pytest

import pythonic_dax as pdax
from pythonic_dax import ALL, CALCULATE, FILTER, KEEPFILTERS, ROW, SUM, VALUES


@pytest.fixture
def model() -> pdax.Model:
    m = pdax.Model()
    m.add_table(
        "Customer", pl.DataFrame({"CustomerKey": [1, 2, 3], "Tier": ["Gold", "Silver", "Gold"]})
    )
    m.add_table(
        "Sales",
        pl.DataFrame({"CustomerKey": [1, 2, 3, 9], "Amount": [10.0, 5.0, 2.5, 100.0]}),
    )
    m.add_relationship(m.table("Sales")["CustomerKey"], m.table("Customer")["CustomerKey"])
    m.add_measure("Total Sales", SUM(m.table("Sales")["Amount"]), table="Sales")
    return m


def test_matches_evaluate_row_calculate(model: pdax.Model) -> None:
    total, tier = model["Total Sales"], model.table("Customer")["Tier"]
    want = model.evaluate(ROW("v", CALCULATE(total, tier == "Gold"))).item()
    assert model.value(total, tier == "Gold") == want == 12.5


def test_measure_object_name_and_dax_text_agree(model: pdax.Model) -> None:
    tier = model.table("Customer")["Tier"]
    by_object = model.value(model["Total Sales"], tier == "Gold")
    assert model.value("Total Sales", tier == "Gold") == by_object
    assert model.value("total sales", tier == "Gold") == by_object  # names ignore case
    assert model.value("[Total Sales]", 'Customer[Tier] = "Gold"') == by_object
    assert model.value("SUM(Sales[Amount])", 'Customer[Tier] = "Gold"') == by_object


def test_no_filters_is_the_whole_model(model: pdax.Model) -> None:
    assert model.value("Total Sales") == 117.5


def test_modifiers_and_filter_tables(model: pdax.Model) -> None:
    customer = model.table("Customer")
    tier = customer["Tier"]
    # As in CALCULATE, ALL is applied before the other filters, so Gold still counts.
    assert model.value("Total Sales", tier == "Gold", ALL(customer)) == 12.5
    assert model.value("Total Sales", ALL(customer)) == 117.5
    assert model.value("Total Sales", KEEPFILTERS(tier == "Gold"), tier != "Gold") is None
    key = customer["CustomerKey"]
    assert model.value("Total Sales", FILTER(ALL(key), key >= 2)) == 7.5
    assert model.value("Total Sales", VALUES(key)) == 117.5  # VALUES keeps the blank row


def test_runs_through_evaluate(model: pdax.Model) -> None:
    """value() is EVALUATE ROW("Value", CALCULATE(...)), so to_dax_query shows it."""
    seen: list[str] = []
    real = model.evaluate

    def recording(query: object) -> pl.DataFrame:
        seen.append(model.to_dax_query(query))
        return real(query)

    model.evaluate = recording  # type: ignore[method-assign]
    model.value("Total Sales", 'Customer[Tier] = "Gold"')
    assert seen[-1].endswith('ROW("Value", CALCULATE([Total Sales], Customer[Tier] = "Gold"))')
    model.value("Total Sales")
    assert seen[-1].endswith('ROW("Value", [Total Sales])')


def test_errors(model: pdax.Model) -> None:
    with pytest.raises(TypeError, match="not a table"):
        model.value(ALL(model.table("Customer")))
    with pytest.raises(pdax.DaxError, match="single value for column"):
        model.value(model.table("Sales")["Amount"])
    with pytest.raises(pdax.DaxError, match="Nope"):
        model.value("Nope")
    other = pdax.Model()
    t = other.add_table("T", pl.DataFrame({"x": [1]}))
    with pytest.raises(pdax.ModelError, match="different model"):
        model.value(other.add_measure("M", SUM(t["x"])))
