"""Phase 6: the scalar function library and the INFO.VIEW functions.

Values checked against Power BI are in tests/golden/cases/library. These tests cover
the refusals (NotSupportedError where Power BI's answer wasn't checked), errors, and
behavior that depends on the model or the clock.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import polars as pl
import pytest

import pythonic_dax as pdax
from pythonic_dax import (
    BLANK,
    CONTAINS,
    CONTAINSROW,
    CONVERT,
    DATE,
    DATEVALUE,
    DIVIDE,
    FILTER,
    FORMAT,
    IFERROR,
    INFO,
    ISERROR,
    MOD,
    MROUND,
    NOW,
    ROUND,
    ROW,
    SEARCH,
    SQRT,
    SUBSTITUTE,
    SUM,
    TODAY,
    VALUE,
    WEEKDAY,
)
from pythonic_dax.errors import CalculationError
from pythonic_dax.expr import coerce


@pytest.fixture
def model() -> pdax.Model:
    m = pdax.Model()
    product = m.add_table(
        "Product",
        pl.DataFrame({"ProductKey": [1, 2], "Color": ["Red", "Blue"]}),
        description="Things we sell.",
    )
    sales = m.add_table(
        "Sales",
        pl.DataFrame({"ProductKey": [1, 1, 2], "Amount": [10.0, 5.0, 2.5]}),
    )
    m.add_relationship(sales["ProductKey"], product["ProductKey"])
    m.add_measure("Total", SUM(sales["Amount"]), table="Sales", format_string="0.00")
    return m


def value(model: pdax.Model, expr: Any) -> Any:
    return model.evaluate(ROW("v", expr))["[v]"][0]


# --- errors and IFERROR ---------------------------------------------------------------


def test_calculation_errors_surface_outside_iferror(model: pdax.Model) -> None:
    with pytest.raises(CalculationError):
        value(model, SQRT(-1))
    with pytest.raises(pdax.DaxError, match="Cannot convert value 'x' of type Text"):
        value(model, VALUE("x"))


def test_iferror_refuses_errors_power_bi_may_not_catch(model: pdax.Model) -> None:
    with pytest.raises(pdax.NotSupportedError, match="may not catch"):
        value(model, ISERROR(DATE(2024, 0, 1)))


def test_date_with_month_zero_is_an_error(model: pdax.Model) -> None:
    with pytest.raises(pdax.DaxError, match="An argument of function 'DATE'"):
        value(model, DATE(2024, 0, 1))


def test_iferror_of_a_measure(model: pdax.Model) -> None:
    total = model["Total"]
    assert value(model, IFERROR(total / 0, -1)) == -1.0
    assert value(model, DIVIDE(total, 0)) is None


# --- refusals -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "expr",
    [
        FORMAT(1234.5, "#,##0.00", "de-DE"),
        FORMAT(1, "Bogus Name 0x"),
        coerce("1") + 1,
        ROUND(1.5, 0.5),
        SQRT(BLANK()),
        MROUND(10, -3),
        CONVERT(BLANK(), pdax.STRING),
        SUBSTITUTE("abc", "", "x"),
        WEEKDAY(DATE(2024, 1, 1), 11),
        DATEVALUE("5 March 2024"),
        VALUE("12,34"),
        coerce(1e-7).concat(""),
        coerce(1.5e20).concat(""),
        DATE(1800, 1, 1),
    ],
    ids=lambda e: e.to_dax(),
)
def test_unverified_behavior_is_refused(model: pdax.Model, expr: Any) -> None:
    with pytest.raises(pdax.NotSupportedError):
        value(model, expr)


def test_mod_by_zero_is_an_error(model: pdax.Model) -> None:
    with pytest.raises(CalculationError):
        value(model, MOD(1, 0))


# --- other behavior -------------------------------------------------------------------


def test_search_wildcards_can_be_escaped(model: pdax.Model) -> None:
    assert value(model, SEARCH("~*", "ab*c")) == 3
    assert value(model, SEARCH("b?d", "abcd")) == 2


def test_substitute_instance_past_the_last_leaves_text(model: pdax.Model) -> None:
    assert value(model, SUBSTITUTE("a-b", "-", "+", 2)) == "a-b"


def test_contains_and_containsrow(model: pdax.Model) -> None:
    product = model.table("Product")
    assert value(model, CONTAINS(product, product["Color"], "red")) is True
    assert value(model, CONTAINS(product, product["Color"], "Green")) is False
    blue = FILTER(product, product["Color"] == "Blue")
    assert value(model, CONTAINSROW(blue, 2, "BLUE")) is True
    assert value(model, CONTAINSROW(blue, 1, "Blue")) is False


def test_today_and_now(model: pdax.Model) -> None:
    today = value(model, TODAY())
    assert today == dt.datetime.combine(dt.date.today(), dt.time())
    assert abs((value(model, NOW()) - dt.datetime.now()).total_seconds()) < 5


# --- INFO.VIEW ------------------------------------------------------------------------


def test_info_view_tables_and_measures(model: pdax.Model) -> None:
    tables = model.evaluate(INFO.VIEW.TABLES())
    assert tables["[Name]"].to_list() == ["Product", "Sales"]
    assert tables["[Description]"].to_list() == ["Things we sell.", None]
    assert tables["[StorageMode]"].to_list() == ["Import", "Import"]

    measures = model.evaluate(INFO.VIEW.MEASURES())
    assert measures.row(0, named=True) | {"[ID]": 1} == {
        "[ID]": 1,
        "[Name]": "Total",
        "[Table]": "Sales",
        "[Description]": None,
        "[DataType]": "Number",
        "[Expression]": "SUM(Sales[Amount])",
        "[FormatString]": "0.00",
        "[IsHidden]": False,
        "[State]": "Valid",
        "[KPIID]": None,
        "[IsSimpleMeasure]": False,
        "[DisplayFolder]": None,
        "[DetailRowsDefinition]": None,
        "[DataCategory]": None,
        "[FormatStringDefinition]": None,
        "[LineageTag]": None,
    }


def test_info_view_columns(model: pdax.Model) -> None:
    columns = model.evaluate(INFO.VIEW.COLUMNS())
    assert columns.columns[:4] == ["[ID]", "[Name]", "[Table]", "[DataType]"]
    rows = {(r["[Table]"], r["[Name]"]): r for r in columns.iter_rows(named=True)}
    assert rows[("Sales", "Amount")]["[DataType]"] == "Number"
    assert rows[("Product", "Color")]["[DataType]"] == "Text"
    assert rows[("Product", "Color")]["[SummarizeBy]"] == "Default"
    assert rows[("Product", "Color")]["[Type]"] == "Data"


def test_info_view_relationships(model: pdax.Model) -> None:
    rels = model.evaluate(INFO.VIEW.RELATIONSHIPS())
    row = rels.row(0, named=True)
    assert row["[Relationship]"] == "'Sales'[ProductKey] *[<-]1 'Product'[ProductKey]"
    assert row["[CrossFilteringBehavior]"] == "OneDirection"
    assert (row["[FromCardinality]"], row["[ToCardinality]"]) == ("Many", "One")
    assert row["[State]"] == "Ready"


def test_info_view_prints_as_dax(model: pdax.Model) -> None:
    assert "INFO.VIEW.MEASURES()" in model.to_dax_query(INFO.VIEW.MEASURES())
