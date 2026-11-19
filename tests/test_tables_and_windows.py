"""Unit tests for Phase 5: iterators, table functions, subtotals and window functions.

Results Power BI defines are pinned by golden cases; these tests cover the places where
the engine refuses to guess and the errors it raises.
"""

from __future__ import annotations

import polars as pl
import pytest

import pythonic_dax as pdax
from pythonic_dax import (
    ALL,
    ASC,
    CALCULATE,
    CONCATENATEX,
    COUNTROWS,
    DATATABLE,
    DESC,
    OFFSET,
    ORDERBY,
    RANK,
    ROLLUPADDISSUBTOTAL,
    ROW,
    ROWNUMBER,
    SUM,
    SUMMARIZECOLUMNS,
    TOPN,
)
from pythonic_dax.errors import DaxError, NotSupportedError


@pytest.fixture
def model() -> pdax.Model:
    m = pdax.Model()
    m.add_table(
        "Product",
        pl.DataFrame(
            {"ProductKey": [1, 2, 3], "Color": ["Red", "Red", "Blue"], "Size": [-1, None, 2]}
        ),
    )
    m.add_table("Sales", pl.DataFrame({"ProductKey": [1, 1, 2], "Amount": [10.0, 5.0, 2.5]}))
    m.add_relationship(m.table("Sales")["ProductKey"], m.table("Product")["ProductKey"])
    return m


def value(model: pdax.Model, expr: object) -> object:
    return model.evaluate(ROW("v", expr))["[v]"][0]


def test_concatenatex_without_an_order_is_refused(model: pdax.Model) -> None:
    p = model.table("Product")
    with pytest.raises(NotSupportedError, match="without an order"):
        value(model, CONCATENATEX(p, p["Color"]))


def test_topn_without_an_order_is_refused_when_it_cuts(model: pdax.Model) -> None:
    p = model.table("Product")
    with pytest.raises(NotSupportedError):
        value(model, COUNTROWS(TOPN(1, p)))


def test_window_function_needs_a_relation(model: pdax.Model) -> None:
    p = model.table("Product")
    with pytest.raises(NotSupportedError, match="without a relation"):
        value(model, COUNTROWS(OFFSET(1, orderBy=ORDERBY(p["Color"]))))


def test_window_order_must_identify_each_row(model: pdax.Model) -> None:
    p = model.table("Product")
    with pytest.raises(NotSupportedError, match="identify each row"):
        value(model, COUNTROWS(OFFSET(1, ALL(p), ORDERBY(p["Color"]))))


def test_rank_allows_ties(model: pdax.Model) -> None:
    p = model.table("Product")
    result = model.evaluate(
        SUMMARIZECOLUMNS(
            p["ProductKey"], {"R": RANK(pdax.DENSE, ALL(p), ORDERBY(p["Color"], DESC))}
        )
    )
    assert result["[R]"].to_list() == [1, 1, 2]


def test_default_blanks_next_to_negative_numbers_are_refused(model: pdax.Model) -> None:
    p = model.table("Product")
    with pytest.raises(NotSupportedError, match="negative numbers"):
        model.evaluate(
            SUMMARIZECOLUMNS(p["ProductKey"], {"N": ROWNUMBER(ALL(p), ORDERBY(p["Size"], ASC))})
        )


def test_blanks_last_next_to_negative_numbers_works(model: pdax.Model) -> None:
    p = model.table("Product")
    result = model.evaluate(
        SUMMARIZECOLUMNS(p["ProductKey"], {"N": ROWNUMBER(ALL(p), ORDERBY(p["Size"]), pdax.LAST)})
    )
    assert result["[N]"].to_list() == [1, 3, 2]


def test_relation_with_only_computed_columns_is_an_error(model: pdax.Model) -> None:
    t = DATATABLE({"K": "INTEGER"}, [[1], [2]])
    with pytest.raises(DaxError, match="only contains columns added by DAX table functions"):
        value(model, ROWNUMBER(t, ORDERBY(t["K"])))


def test_rollup_only_works_in_summarizecolumns(model: pdax.Model) -> None:
    p = model.table("Product")
    with pytest.raises(DaxError, match="group-by columns of SUMMARIZECOLUMNS"):
        CALCULATE(SUM(model.table("Sales")["Amount"]), ROLLUPADDISSUBTOTAL(p["Color"], "T"))


def test_two_rollups_are_refused(model: pdax.Model) -> None:
    p = model.table("Product")
    with pytest.raises(NotSupportedError, match="more than one"):
        SUMMARIZECOLUMNS(
            ROLLUPADDISSUBTOTAL(p["Color"], "A"),
            ROLLUPADDISSUBTOTAL(p["Size"], "B"),
            {"N": COUNTROWS(p)},
        )


def test_rollup_needs_column_and_name_pairs(model: pdax.Model) -> None:
    p = model.table("Product")
    with pytest.raises(TypeError):
        ROLLUPADDISSUBTOTAL(p["Color"])
