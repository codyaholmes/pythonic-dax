"""Phase 7: time intelligence.

Values checked against Power BI are in tests/golden/cases/time-intelligence. These tests
cover the refusals (NotSupportedError where Power BI's answer wasn't checked), errors,
and the date-filter rule in corner cases.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import polars as pl
import pytest

import pythonic_dax as pdax
from pythonic_dax import (
    CALCULATE,
    CALENDAR,
    CLOSINGBALANCEYEAR,
    COUNTROWS,
    DATE,
    DATEADD,
    DATESBETWEEN,
    DATESINPERIOD,
    DATESYTD,
    ENDOFYEAR,
    FILTER,
    LASTDATE,
    OPENINGBALANCEYEAR,
    PREVIOUSYEAR,
    ROW,
    STARTOFYEAR,
    SUM,
    TOTALYTD,
    VALUES,
)


@pytest.fixture
def model() -> pdax.Model:
    m = pdax.Model()
    days = [dt.datetime(2024, 1, 1) + dt.timedelta(days=i) for i in range(366)]
    m.add_table(
        "Dates",
        pl.DataFrame({"Date": days, "Month": [d.strftime("%Y-%m") for d in days]}),
    )
    sales = m.add_table(
        "Sales",
        pl.DataFrame(
            {
                "Date": [dt.datetime(2024, 1, 5), dt.datetime(2024, 2, 5)],
                "Stamp": [dt.datetime(2024, 1, 5, 9), dt.datetime(2024, 2, 5, 10)],
                "Amount": [10.0, 20.0],
            }
        ),
    )
    m.add_relationship(sales["Date"], m.table("Dates")["Date"])
    return m


def value(model: pdax.Model, expr: Any) -> Any:
    return model.evaluate(ROW("v", expr))["[v]"][0]


def dates(model: pdax.Model) -> pdax.Column:
    return model.table("Dates")["Date"]


@pytest.mark.parametrize(
    "make",
    [
        lambda d, s: DATESYTD(d, "2/29"),
        lambda d, s: DATESYTD(d, "6/15"),
        lambda d, s: DATESYTD(d, "June 30"),
        lambda d, s: STARTOFYEAR(d, "6/30"),
        lambda d, s: PREVIOUSYEAR(d, "6/30"),
        lambda d, s: DATESINPERIOD(d, DATE(2024, 3, 1), 0, "MONTH"),
        lambda d, s: DATESINPERIOD(d, pdax.BLANK(), 1, "MONTH"),
        lambda d, s: DATESBETWEEN(d, DATE(2024, 3, 1) + pdax.TIME(6, 0, 0), pdax.BLANK()),
        lambda d, s: DATESYTD(VALUES(d)),
        lambda d, s: LASTDATE(s.table["Stamp"]),
        lambda d, s: CALENDAR(DATE(2024, 3, 2), DATE(2024, 3, 1)),
        lambda d, s: CALENDAR(pdax.BLANK(), DATE(2024, 3, 1)),
    ],
)
def test_unverified_behavior_is_refused(model: pdax.Model, make: Any) -> None:
    sales_date = model.table("Sales")["Date"]
    with pytest.raises(pdax.NotSupportedError):
        value(model, COUNTROWS(make(dates(model), sales_date)))


def test_dateadd_of_a_set_with_gaps_and_neighbours_is_refused(model: pdax.Model) -> None:
    d = dates(model)
    picked = d.IN([DATE(2024, 3, 1), DATE(2024, 3, 2), DATE(2024, 3, 10)])
    with pytest.raises(pdax.NotSupportedError, match="gaps"):
        value(model, CALCULATE(COUNTROWS(DATEADD(d, -1, "MONTH")), picked))
    lone = d.IN([DATE(2024, 3, 1), DATE(2024, 3, 10)])
    assert value(model, CALCULATE(COUNTROWS(DATEADD(d, -1, "MONTH")), lone)) == 2


def test_opening_balance_with_a_year_end_is_refused(model: pdax.Model) -> None:
    total = SUM(model.table("Sales")["Amount"])
    with pytest.raises(pdax.NotSupportedError):
        value(model, OPENINGBALANCEYEAR(total, dates(model), year_end_date="6/30"))
    closing = CLOSINGBALANCEYEAR(total, dates(model), year_end_date="6/30")
    assert 'CLOSINGBALANCEYEAR(SUM(Sales[Amount]), Dates[Date], "6/30")' in closing.to_dax()


def test_text_third_argument_is_the_year_end(model: pdax.Model) -> None:
    total = SUM(model.table("Sales")["Amount"])
    d = dates(model)
    expr = TOTALYTD(total, d, year_end_date="1/31")
    assert expr.to_dax() == 'TOTALYTD(SUM(Sales[Amount]), Dates[Date], "1/31")'
    feb = model.table("Dates")["Month"] == "2024-02"
    assert value(model, CALCULATE(expr, feb)) == 20.0


def test_date_column_must_be_a_date(model: pdax.Model) -> None:
    with pytest.raises(pdax.DaxError, match="Date/Time"):
        value(model, COUNTROWS(DATESYTD(model.table("Dates")["Month"])))


def test_filter_on_one_side_of_inactive_relationship_only_is_refused() -> None:
    m = pdax.Model()
    days = [dt.datetime(2024, 1, 1) + dt.timedelta(days=i) for i in range(3)]
    cal = m.add_table("Cal", pl.DataFrame({"Date": days}))
    t = m.add_table("T", pl.DataFrame({"D": days[:1]}))
    m.add_relationship(t["D"], cal["Date"], active=False)
    with pytest.raises(pdax.NotSupportedError, match="inactive"):
        value(m, CALCULATE(COUNTROWS(cal), cal["Date"] == DATE(2024, 1, 1)))


def test_marked_date_table_filter_clears_the_table() -> None:
    m = pdax.Model()
    days = [dt.datetime(2024, 1, 1) + dt.timedelta(days=i) for i in range(60)]
    cal = m.add_table("Cal", pl.DataFrame({"Date": days, "M": [d.month for d in days]}))
    inner = CALCULATE(COUNTROWS(cal), cal["Date"] <= DATE(2024, 1, 3))
    assert value(m, CALCULATE(inner, cal["M"] == 2)) is None
    cal.mark_as_date_table("Date")
    assert value(m, CALCULATE(inner, cal["M"] == 2)) == 3


def test_endofyear_on_a_sparse_column(model: pdax.Model) -> None:
    sales_date = model.table("Sales")["Date"]
    got = value(model, pdax.MAXX(ENDOFYEAR(sales_date), sales_date))
    assert got == dt.datetime(2024, 2, 5)


def test_calendar_rows(model: pdax.Model) -> None:
    frame = model.evaluate(CALENDAR(DATE(2024, 2, 28), DATE(2024, 3, 1)))
    assert frame["[Date]"].to_list() == [
        dt.datetime(2024, 2, 28),
        dt.datetime(2024, 2, 29),
        dt.datetime(2024, 3, 1),
    ]


def test_filter_of_time_intelligence_rows(model: pdax.Model) -> None:
    d = dates(model)
    ytd = FILTER(DATESYTD(d), d >= DATE(2024, 1, 30))
    feb = model.table("Dates")["Month"] == "2024-02"
    assert value(model, CALCULATE(COUNTROWS(ytd), feb)) == 31
