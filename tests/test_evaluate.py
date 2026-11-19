"""Unit tests for the Phase 3 evaluation engine.

Behavior that Power BI defines is pinned by golden cases; these tests cover the engine's
own plumbing and the places where it refuses to guess.
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal

import polars as pl
import pytest

import pythonic_dax as pdax
from pythonic_dax import (
    BLANK,
    COUNTROWS,
    IF,
    RELATED,
    RELATEDTABLE,
    ROW,
    SUM,
    SUMMARIZECOLUMNS,
    VAR,
)
from pythonic_dax import _scalar as sc
from pythonic_dax._filters import ColumnFilter, FilterContext


@pytest.fixture
def model() -> pdax.Model:
    m = pdax.Model()
    m.add_table("Product", pl.DataFrame({"ProductKey": [1, 2, 3], "Color": ["Red", "Red", "Blue"]}))
    m.add_table(
        "Sales",
        pl.DataFrame({"ProductKey": [1, 1, 2, 9], "Amount": [10.0, 5.0, 2.5, 100.0]}),
    )
    m.add_relationship(m.table("Sales")["ProductKey"], m.table("Product")["ProductKey"])
    return m


def value(model: pdax.Model, expr: object) -> object:
    return model.evaluate(ROW("v", expr))["[v]"][0]


# --- results --------------------------------------------------------------------------


def test_milestone_first_measure(model: pdax.Model) -> None:
    sales, product = model.table("Sales"), model.table("Product")
    total = model.add_measure("Total", SUM(sales["Amount"]), table=sales)
    result = model.evaluate(SUMMARIZECOLUMNS(product["Color"], {"Total": total}))
    assert result.columns == ["Product[Color]", "[Total]"]
    # Blue has no sales, so its row is dropped. Key 9 has no product: the blank row.
    assert set(result.rows()) == {("Red", 17.5), (None, 100.0)}


def test_measure_inside_measure(model: pdax.Model) -> None:
    sales = model.table("Sales")
    total = model.add_measure("Total", SUM(sales["Amount"]), table=sales)
    double = model.add_measure("Double", total * 2, table=sales)
    assert value(model, double) == 235.0


def test_evaluate_needs_a_table(model: pdax.Model) -> None:
    with pytest.raises(TypeError, match="EVALUATE needs a table"):
        model.evaluate(SUM(model.table("Sales")["Amount"]))


def test_column_without_row_context_is_an_error(model: pdax.Model) -> None:
    with pytest.raises(pdax.DaxError, match="A single value for column 'Amount'"):
        value(model, model.table("Sales")["Amount"])


def test_unbuilt_functions_raise_not_supported(model: pdax.Model) -> None:
    with pytest.raises(pdax.NotSupportedError, match="Phase 6"):
        value(model, pdax.ABS(-1))


# --- variables --------------------------------------------------------------------------


def test_var_is_evaluated_once_and_lazily(model: pdax.Model) -> None:
    sales = model.table("Sales")
    expr = VAR(Total=SUM(sales["Amount"]), Bad=BLANK() ** BLANK()).RETURN(lambda v: v.Total + 1)
    assert value(model, expr) == 118.5  # Bad is never used, so its error never happens


def test_var_error_surfaces_when_used(model: pdax.Model) -> None:
    expr = VAR(Bad=BLANK() ** BLANK()).RETURN(lambda v: v.Bad)
    with pytest.raises(pdax.DaxError, match="POWER"):
        value(model, expr)


def test_if_only_evaluates_the_chosen_branch(model: pdax.Model) -> None:
    assert value(model, IF(True, 1, BLANK() ** BLANK())) == 1


# --- calculated columns ----------------------------------------------------------------


def test_calculated_column_with_related(model: pdax.Model) -> None:
    sales, product = model.table("Sales"), model.table("Product")
    color = model.add_column(sales, "Color", RELATED(product["Color"]), description="From Product")
    assert color.is_calculated
    assert color.description == "From Product"
    assert sales.data["Color"].to_list() == ["Red", "Red", "Red", None]
    assert "COLUMN Sales[Color] = RELATED(Product[Color])" in model.to_dax_query(
        SUMMARIZECOLUMNS(sales["Color"], {"N": COUNTROWS(sales)})
    )


def test_calculated_column_relatedtable(model: pdax.Model) -> None:
    sales, product = model.table("Sales"), model.table("Product")
    model.add_column(product, "Sales Rows", COUNTROWS(RELATEDTABLE(sales)))
    assert product.data["Sales Rows"].to_list() == [2, 1, None]


def test_refresh_recomputes_calculated_columns() -> None:
    frames = [pl.DataFrame({"x": [1, 2]})]
    m = pdax.Model()
    t = m.add_table("T", pdax.sources.DataFrameSource(frames[0]))
    m.add_column(t, "Double", t["x"] * 2)
    t.source = pdax.sources.DataFrameSource(pl.DataFrame({"x": [5]}))
    m.refresh()
    assert t.data["Double"].to_list() == [10]


def test_calculated_column_name_rules(model: pdax.Model) -> None:
    sales = model.table("Sales")
    with pytest.raises(pdax.ModelError, match="already has a column"):
        model.add_column(sales, "amount", sales["Amount"] * 2)
    model.add_measure("Total", SUM(sales["Amount"]), table=sales)
    with pytest.raises(pdax.ModelError, match="already has a measure"):
        model.add_column(sales, "Total", sales["Amount"] * 2)


def test_key_column_allows_two_transitions(model: pdax.Model) -> None:
    m = pdax.Model()
    t = m.add_table("T", pl.DataFrame({"id": [1, 2], "x": [3.0, 4.0]}))
    total = m.add_measure("Total", SUM(t["x"]), table=t)
    m.add_column(t, "A", total)
    with pytest.raises(pdax.DaxError, match="circular dependency"):
        m.add_column(t, "B", total + 1)
    t["id"].set(is_key=True)
    m.add_column(t, "B", total + 1)
    assert t.data["B"].to_list() == [4.0, 5.0]


# --- filter context ----------------------------------------------------------------------


def test_new_filter_replaces_filters_on_its_columns(model: pdax.Model) -> None:
    product = model.table("Product")
    key, color = product["ProductKey"], product["Color"]
    both = ColumnFilter(product, [key, color], pl.DataFrame({"ProductKey": [1], "Color": ["Red"]}))
    only_color = ColumnFilter(product, [color], pl.DataFrame({"Color": ["Blue"]}))
    ctx = FilterContext((both,)).apply([only_color])
    assert [f.columns for f in ctx.filters] == [(key,), (color,)]
    assert ctx.filters[0].values["ProductKey"].to_list() == [1]


# --- refusing to guess -----------------------------------------------------------------


def test_text_values_differing_by_case_are_refused() -> None:
    m = pdax.Model()
    t = m.add_table("T", pl.DataFrame({"c": ["a", "A"], "x": [1, 2]}))
    total = m.add_measure("Total", SUM(t["x"]), table=t)
    with pytest.raises(pdax.NotSupportedError, match="differ only by case"):
        m.evaluate(SUMMARIZECOLUMNS(t["c"], {"Total": total}))


def test_trailing_space_keys_are_refused() -> None:
    m = pdax.Model()
    one = m.add_table("One", pl.DataFrame({"k": ["a", "b"]}))
    many = m.add_table("Many", pl.DataFrame({"k": ["a ", "b"], "x": [1, 2]}))
    m.add_relationship(many["k"], one["k"])
    total = m.add_measure("Total", SUM(many["x"]), table=many)
    with pytest.raises(pdax.NotSupportedError, match="trailing spaces"):
        m.evaluate(SUMMARIZECOLUMNS(one["k"], {"Total": total}))


def test_currency_rounding_ties_are_refused() -> None:
    with pytest.raises(pdax.NotSupportedError, match="tie"):
        sc.to_currency(0.00005)
    assert sc.to_currency(2 / 3) == Decimal("0.6667")


# --- scalar rules --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("op", "a", "b", "expected"),
    [
        ("+", None, 1, 1),
        ("*", None, 1, None),
        ("/", 1, None, float("inf")),
        ("/", None, 1, None),
        ("+", Decimal("1.25"), Decimal("2"), Decimal("3.25")),
        ("*", Decimal("1.5"), Decimal("2"), 3.0),
        ("+", Decimal("1"), 1.5, 2.5),
        ("+", dt.datetime(2020, 1, 1), 1, dt.datetime(2020, 1, 2)),
        ("-", dt.datetime(2020, 1, 3), dt.datetime(2020, 1, 1), dt.datetime(1900, 1, 1)),
        ("*", dt.datetime(2020, 1, 1), 2, 87662.0),
    ],
)
def test_arithmetic(op: str, a: object, b: object, expected: object) -> None:
    result = sc.arithmetic(op, a, b)
    assert result == expected
    assert type(result) is type(expected)


def test_comparing_text_with_numbers_is_an_error() -> None:
    with pytest.raises(pdax.DaxError, match="comparing values of type Text with values of type"):
        sc.compare("=", "a", 1)


def test_zero_divided_by_blank_is_nan() -> None:
    result = sc.arithmetic("/", 0, None)
    assert result != result  # NaN


# --- CALCULATE (Phase 4) -------------------------------------------------------------


def test_new_filters_in_one_calculate_intersect(model: pdax.Model) -> None:
    product = model.table("Product")
    color = product["Color"]
    red = ColumnFilter(product, [color], pl.DataFrame({"Color": ["Red"]}))
    blue = ColumnFilter(product, [color], pl.DataFrame({"Color": ["Blue"]}))
    ctx = FilterContext().apply([red, blue])
    assert len(ctx.filters) == 2


def test_boolean_filter_fast_path_matches_row_by_row(model: pdax.Model) -> None:
    sales = model.table("Sales")
    total = model.add_measure("Total", SUM(sales["Amount"]), table=sales)
    fast = pdax.CALCULATE(total, sales["Amount"] > 4)
    slow = pdax.CALCULATE(total, IF(sales["Amount"] > 4, True, False))  # not vectorized
    assert value(model, fast) == value(model, slow) == 115.0


def test_allselected_needs_an_iteration(model: pdax.Model) -> None:
    sales = model.table("Sales")
    expr = pdax.CALCULATE(SUM(sales["Amount"]), pdax.ALLSELECTED())
    with pytest.raises(pdax.NotSupportedError, match="ALLSELECTED"):
        value(model, expr)


def test_keepfilters_of_all_is_refused(model: pdax.Model) -> None:
    sales, product = model.table("Sales"), model.table("Product")
    expr = pdax.CALCULATE(SUM(sales["Amount"]), pdax.KEEPFILTERS(pdax.ALL(product["Color"])))
    with pytest.raises(pdax.NotSupportedError, match="KEEPFILTERS"):
        value(model, expr)


def test_expansion_through_an_inactive_relationship_is_refused() -> None:
    m = pdax.Model()
    region = m.add_table("Region", pl.DataFrame({"R": [1, 2]}))
    store = m.add_table("Store", pl.DataFrame({"S": [1], "R": [1], "R2": [2]}))
    m.add_relationship(store["R2"], region["R"], active=False)
    expr = pdax.CALCULATE(COUNTROWS(region), pdax.REMOVEFILTERS(store))
    with pytest.raises(pdax.NotSupportedError, match="inactive relationship"):
        m.evaluate(ROW("v", expr))


def test_crossfilter_on_an_inactive_relationship_is_refused() -> None:
    m = pdax.Model()
    one = m.add_table("One", pl.DataFrame({"k": [1, 2]}))
    many = m.add_table("Many", pl.DataFrame({"k": [1], "j": [2]}))
    m.add_relationship(many["k"], one["k"])
    m.add_relationship(many["j"], one["k"], active=False)
    expr = pdax.CALCULATE(COUNTROWS(one), pdax.CROSSFILTER(many["j"], one["k"], "BOTH"))
    with pytest.raises(pdax.NotSupportedError, match="inactive"):
        m.evaluate(ROW("v", expr))


def test_lookupvalue_blank_search_is_refused(model: pdax.Model) -> None:
    product = model.table("Product")
    expr = pdax.LOOKUPVALUE(product["Color"], product["ProductKey"], BLANK())
    with pytest.raises(pdax.NotSupportedError, match="BLANK"):
        value(model, expr)


def test_table_constructor_as_a_table(model: pdax.Model) -> None:
    result = model.evaluate(pdax.TREATAS([1, 2], model.table("Product")["ProductKey"]))
    assert result.columns == ["Product[ProductKey]"]
    assert result["Product[ProductKey]"].to_list() == [1, 2]
