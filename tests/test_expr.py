"""The Python expression API: building expressions, DAX checks and DAX printing."""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

import pythonic_dax as pdax
from pythonic_dax import (
    ADDCOLUMNS,
    ALL,
    ALLEXCEPT,
    BLANK,
    CALCULATE,
    CALCULATETABLE,
    CONCATENATEX,
    COUNTROWS,
    CROSSFILTER,
    DATATABLE,
    DATEADD,
    DESC,
    DIVIDE,
    FILTER,
    IF,
    KEEPFILTERS,
    LOOKUPVALUE,
    MAX,
    MIN,
    QUARTER,
    RANKX,
    REMOVEFILTERS,
    ROW,
    SELECTCOLUMNS,
    SUM,
    SUMMARIZECOLUMNS,
    SUMX,
    SWITCH,
    TOPN,
    TREATAS,
    TRUE,
    USERELATIONSHIP,
    VALUES,
    VAR,
    YEAR,
    DaxError,
)


@pytest.fixture
def m() -> pdax.Model:
    model = pdax.Model()
    model.add_table(
        "Product", pl.DataFrame({"ProductKey": [1, 2], "Color": ["Red", "Blue"], "Price": [1.5, 2]})
    )
    model.add_table(
        "Sales",
        pl.DataFrame(
            {
                "ProductKey": [1, 2],
                "Amount": [10.0, 20.0],
                "Qty": [1, 2],
                "OrderDate": [dt.date(2026, 1, 1), dt.date(2026, 1, 2)],
                "ShipDate": [dt.date(2026, 1, 2), dt.date(2026, 1, 3)],
            }
        ),
    )
    model.add_table(
        "Date",
        pl.DataFrame({"Date": pl.date_range(dt.date(2026, 1, 1), dt.date(2026, 1, 3), eager=True)}),
    )
    model.add_table("Sales Order", pl.DataFrame({"Order No": [1]}))
    s, p, d = model.table("Sales"), model.table("Product"), model.table("Date")
    model.add_relationship(s["ProductKey"], p["ProductKey"])
    model.add_relationship(s["OrderDate"], d["Date"])
    model.add_relationship(s["ShipDate"], d["Date"], active=False)
    return model


def T(m: pdax.Model, name: str) -> pdax.Table:
    return m.table(name)


# --- references and literals ----------------------------------------------------------


def test_references_print_as_dax(m: pdax.Model) -> None:
    assert T(m, "Sales")["Amount"].to_dax() == "Sales[Amount]"
    assert T(m, "Sales Order")["Order No"].to_dax() == "'Sales Order'[Order No]"
    assert T(m, "Date").to_dax() == "'Date'"  # DATE is a DAX function, so it's quoted
    total = m.add_measure("Total Sales", SUM(T(m, "Sales")["Amount"]))
    assert total.to_dax() == "[Total Sales]"


@pytest.mark.parametrize(
    ("value", "dax"),
    [
        (1, "1"),
        (1.5, "1.5"),
        ('say "hi"', '"say ""hi"""'),
        (True, "TRUE()"),
        (None, "BLANK()"),
        (dt.date(2026, 1, 31), "DATE(2026, 1, 31)"),
        (dt.datetime(2026, 1, 31, 8, 30), "DATE(2026, 1, 31) + TIME(8, 30, 0)"),
    ],
)
def test_literals(value: object, dax: str) -> None:
    assert IF(TRUE(), value).to_dax() == f"IF(TRUE(), {dax})"


# --- operators ------------------------------------------------------------------------


def test_operators_and_precedence(m: pdax.Model) -> None:
    amt, qty = T(m, "Sales")["Amount"], T(m, "Sales")["Qty"]
    assert ((amt + 1) * qty).to_dax() == "(Sales[Amount] + 1) * Sales[Qty]"
    assert (amt + 1 * qty).to_dax() == "Sales[Amount] + 1 * Sales[Qty]"
    assert (amt - (qty - 1)).to_dax() == "Sales[Amount] - (Sales[Qty] - 1)"
    assert (2 - amt).to_dax() == "2 - Sales[Amount]"
    assert (-(amt + 1)).to_dax() == "-(Sales[Amount] + 1)"
    assert (amt**2).to_dax() == "Sales[Amount] ^ 2"
    assert ((amt > 1) & (qty == 2) | ~(qty != 3)).to_dax() == (
        "Sales[Amount] > 1 && Sales[Qty] = 2 || NOT(Sales[Qty] <> 3)"
    )
    assert ((amt > 1) & ((qty == 2) | (qty == 3))).to_dax() == (
        "Sales[Amount] > 1 && (Sales[Qty] = 2 || Sales[Qty] = 3)"
    )
    assert T(m, "Product")["Color"].concat("!").to_dax() == 'Product[Color] & "!"'
    assert (amt.strict_eq(None)).to_dax() == "Sales[Amount] == BLANK()"


def test_in_with_a_list(m: pdax.Model) -> None:
    color = T(m, "Product")["Color"]
    assert color.IN(["Red", "Blue"]).to_dax() == 'Product[Color] IN {"Red", "Blue"}'
    assert pdax.IN(color, VALUES(color)).to_dax() == "Product[Color] IN VALUES(Product[Color])"


def test_python_logic_on_expressions_points_to_dax(m: pdax.Model) -> None:
    amt = T(m, "Sales")["Amount"]
    with pytest.raises(TypeError, match="IF"):
        if amt > 1:
            pass
    with pytest.raises(TypeError, match="parentheses"):
        _ = amt > 1 and amt < 5
    with pytest.raises(TypeError, match="MOD"):
        _ = amt % 2


def test_result_types(m: pdax.Model) -> None:
    s = T(m, "Sales")
    assert (s["Qty"] + 1).dax_type is pdax.DaxType.INTEGER
    assert (s["Qty"] / 2).dax_type is pdax.DaxType.DECIMAL
    assert (s["Amount"] > 1).dax_type is pdax.DaxType.BOOLEAN
    assert SUM(s["Qty"]).dax_type is pdax.DaxType.INTEGER
    assert SUM(s["Amount"]).dax_type is pdax.DaxType.DECIMAL


# --- VAR ------------------------------------------------------------------------------


def test_var_return(m: pdax.Model) -> None:
    s = T(m, "Sales")
    expr = VAR(
        Total=SUM(s["Amount"]),
        Avg=lambda v: DIVIDE(v.Total, COUNTROWS(s)),
    ).RETURN(lambda v: IF(v.Total > 100, v.Avg))
    assert expr.to_dax() == (
        "VAR Total = SUM(Sales[Amount]) VAR Avg = DIVIDE(Total, COUNTROWS(Sales)) "
        "RETURN IF(Total > 100, Avg)"
    )


def test_table_var(m: pdax.Model) -> None:
    s = T(m, "Sales")
    expr = VAR(Big=FILTER(s, s["Amount"] > 10)).RETURN(lambda v: COUNTROWS(v.Big))
    assert expr.to_dax() == "VAR Big = FILTER(Sales, Sales[Amount] > 10) RETURN COUNTROWS(Big)"


def test_var_inside_an_operator_is_parenthesized(m: pdax.Model) -> None:
    block = VAR(x=1).RETURN(lambda v: v.x)
    assert (block + 1).to_dax() == "(VAR x = 1 RETURN x) + 1"


def test_var_names_are_checked() -> None:
    with pytest.raises(ValueError, match="variable name"):
        VAR(SUM=1)
    with pytest.raises(AttributeError, match="no variable named 'b'"):
        VAR(a=1, c=lambda v: v.b)


# --- argument checks (DAX errors raised when the formula is written) --------------------


def test_column_only_arguments(m: pdax.Model) -> None:
    with pytest.raises(DaxError, match="only accepts a column reference"):
        SUM(T(m, "Sales")["Amount"] * 2)


def test_sum_refuses_text_and_boolean(m: pdax.Model) -> None:
    with pytest.raises(DaxError, match="cannot work with values of type String"):
        SUM(T(m, "Product")["Color"])


def test_missing_and_extra_arguments(m: pdax.Model) -> None:
    with pytest.raises(TypeError):
        DIVIDE(1)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        SUM(T(m, "Sales")["Amount"], T(m, "Sales")["Qty"])  # type: ignore[call-arg]


def test_optional_arguments_in_the_middle_are_left_empty(m: pdax.Model) -> None:
    s = T(m, "Sales")
    assert RANKX(ALL(s), SUM(s["Amount"]), order=DESC).to_dax() == (
        "RANKX(ALL(Sales), SUM(Sales[Amount]), , DESC)"
    )


def test_boolean_filters_use_one_table(m: pdax.Model) -> None:
    s, p = T(m, "Sales"), T(m, "Product")
    total = SUM(s["Amount"])
    CALCULATE(total, (p["Color"] == "Red") | (p["Price"] > 1))  # one table: fine
    CALCULATE(total, p["Color"] == MAX(p["Color"]))  # aggregations don't count
    with pytest.raises(DaxError, match="columns from multiple tables"):
        CALCULATE(total, (p["Color"] == "Red") & (s["Qty"] > 1))
    with pytest.raises(DaxError, match="must refer to a column"):
        CALCULATE(total, TRUE())


def test_modifiers_only_inside_calculate(m: pdax.Model) -> None:
    s, d = T(m, "Sales"), T(m, "Date")
    rel = USERELATIONSHIP(s["ShipDate"], d["Date"])
    assert CALCULATE(SUM(s["Amount"]), rel).to_dax() == (
        "CALCULATE(SUM(Sales[Amount]), USERELATIONSHIP(Sales[ShipDate], 'Date'[Date]))"
    )
    with pytest.raises(DaxError, match="only be used as a filter argument"):
        IF(TRUE(), rel)
    with pytest.raises(TypeError):
        SUMX(rel, 1)
    with pytest.raises(DaxError, match="only be used as a filter argument"):
        SUMX(s, REMOVEFILTERS(s))


def test_userelationship_needs_a_relationship(m: pdax.Model) -> None:
    s, p = T(m, "Sales"), T(m, "Product")
    with pytest.raises(DaxError, match="participating in relationship"):
        USERELATIONSHIP(s["Amount"], p["Price"])
    # Either order of the two columns works, as in DAX.
    USERELATIONSHIP(T(m, "Date")["Date"], s["ShipDate"])
    CROSSFILTER(s["ProductKey"], p["ProductKey"], pdax.BOTH)


def test_all_columns_from_one_table(m: pdax.Model) -> None:
    s, p = T(m, "Sales"), T(m, "Product")
    ALL(p["Color"], p["Price"])
    with pytest.raises(DaxError, match="same table"):
        ALL(p["Color"], s["Qty"])
    with pytest.raises(DaxError, match="not both"):
        ALL(p, p["Color"])
    with pytest.raises(DaxError, match="table name"):
        ALL(FILTER(p, p["Price"] > 1))
    assert ALL().kind == "modifier"  # ALL() with no arguments is a CALCULATE modifier
    ALLEXCEPT(p, p["Color"])


def test_keywords_accept_constants_text_and_function_names(m: pdax.Model) -> None:
    d = T(m, "Date")["Date"]
    assert DATEADD(d, -1, YEAR).to_dax() == "DATEADD('Date'[Date], -1, YEAR)"
    assert DATEADD(d, -1, "quarter").to_dax() == "DATEADD('Date'[Date], -1, QUARTER)"
    assert DATEADD(d, 1, QUARTER).to_dax().endswith("QUARTER)")
    with pytest.raises(ValueError, match="must be one of"):
        DATEADD(d, -1, "fortnight")


def test_keepfilters_and_table_filters(m: pdax.Model) -> None:
    p = T(m, "Product")
    total = SUM(T(m, "Sales")["Amount"])
    expr = CALCULATE(total, KEEPFILTERS(p["Color"] == "Red"), FILTER(ALL(p), p["Price"] > 1))
    assert expr.to_dax() == (
        'CALCULATE(SUM(Sales[Amount]), KEEPFILTERS(Product[Color] = "Red"), '
        "FILTER(ALL(Product), Product[Price] > 1))"
    )
    assert CALCULATETABLE(p, p["Color"] == "Red").kind == "table"


def test_treatas_width_must_match(m: pdax.Model) -> None:
    p = T(m, "Product")
    TREATAS(["Red", "Blue"], p["Color"])
    with pytest.raises(DaxError, match="number of columns"):
        TREATAS([("Red", 1)], p["Color"])


# --- functions with name/value pairs ----------------------------------------------------


def test_summarizecolumns_forms(m: pdax.Model) -> None:
    p, s = T(m, "Product"), T(m, "Sales")
    total = m.add_measure("Total", SUM(s["Amount"]), table="Sales")
    a = SUMMARIZECOLUMNS(p["Color"], {"Total": total})
    b = SUMMARIZECOLUMNS(p["Color"], "Total", total)
    assert a.to_dax() == b.to_dax() == 'SUMMARIZECOLUMNS(Product[Color], "Total", [Total])'
    filtered = SUMMARIZECOLUMNS(p["Color"], TREATAS(["Red"], p["Color"]), {"Total": total})
    assert "TREATAS" in filtered.to_dax()
    with pytest.raises(DaxError, match="more than once"):
        SUMMARIZECOLUMNS(p["Color"], {"Total": total}, "total", total)
    with pytest.raises(DaxError, match="must come before"):
        SUMMARIZECOLUMNS({"Total": total}, p["Color"])


def test_addcolumns_selectcolumns_row(m: pdax.Model) -> None:
    p = T(m, "Product")
    assert ADDCOLUMNS(p, {"Double": p["Price"] * 2}).to_dax() == (
        'ADDCOLUMNS(Product, "Double", Product[Price] * 2)'
    )
    assert SELECTCOLUMNS(p, p["Color"], "Twice", p["Price"] * 2).to_dax() == (
        'SELECTCOLUMNS(Product, Product[Color], "Twice", Product[Price] * 2)'
    )
    assert ROW("Plus", BLANK() + 1).to_dax() == 'ROW("Plus", BLANK() + 1)'


def test_switch_topn_concatenatex_lookupvalue(m: pdax.Model) -> None:
    p, s = T(m, "Product"), T(m, "Sales")
    assert SWITCH(p["Color"], "Red", 1, "Blue", 2, 0).to_dax() == (
        'SWITCH(Product[Color], "Red", 1, "Blue", 2, 0)'
    )
    assert TOPN(1, p, SUM(s["Amount"]), DESC).to_dax() == (
        "TOPN(1, Product, SUM(Sales[Amount]), DESC)"
    )
    assert CONCATENATEX(p, p["Color"], ", ", p["Color"], "asc").to_dax() == (
        'CONCATENATEX(Product, Product[Color], ", ", Product[Color], ASC)'
    )
    assert LOOKUPVALUE(p["Color"], p["ProductKey"], 1, "none").to_dax() == (
        'LOOKUPVALUE(Product[Color], Product[ProductKey], 1, "none")'
    )


def test_min_max_two_forms(m: pdax.Model) -> None:
    s = T(m, "Sales")
    assert MIN(s["Amount"]).to_dax() == "MIN(Sales[Amount])"
    assert MAX(1, s["Qty"] * 2).to_dax() == "MAX(1, Sales[Qty] * 2)"
    with pytest.raises(DaxError, match="column reference"):
        MIN(s["Qty"] * 2)


def test_datatable() -> None:
    table = DATATABLE({"Color": "STRING", "Qty": pdax.INTEGER}, [["Red", 1], ["Blue", 2]])
    assert table.to_dax() == 'DATATABLE("Color", STRING, "Qty", INTEGER, {{"Red", 1}, {"Blue", 2}})'
    with pytest.raises(DaxError, match="constants"):
        DATATABLE({"x": "INTEGER"}, [[BLANK() + 1]])


# --- measures and queries -----------------------------------------------------------------


def test_measures_hold_expressions(m: pdax.Model) -> None:
    s, p = T(m, "Sales"), T(m, "Product")
    total = m.add_measure("Total Sales", SUM(s["Amount"]), table="Sales")
    share = m.add_measure(
        "Share", DIVIDE(total, CALCULATE(total, ALL(p))), table="Sales", format_string="0.0%"
    )
    assert share.expression.to_dax() == (
        "DIVIDE([Total Sales], CALCULATE([Total Sales], ALL(Product)))"
    )
    assert share.definition() == (
        "MEASURE Sales[Share] = DIVIDE([Total Sales], CALCULATE([Total Sales], ALL(Product)))"
    )
    assert share.dax_type is pdax.DaxType.DECIMAL


def test_measure_columns_must_come_from_the_model(m: pdax.Model) -> None:
    other = pdax.Model()
    t = other.add_table("X", pl.DataFrame({"a": [1]}))
    with pytest.raises(pdax.ModelError, match="different model"):
        m.add_measure("Bad", SUM(t["a"]))


def test_to_dax_query(m: pdax.Model) -> None:
    s, p = T(m, "Sales"), T(m, "Product")
    total = m.add_measure("Total Sales", SUM(s["Amount"]), table="Sales")
    share = m.add_measure("Share", DIVIDE(total, CALCULATE(total, ALL(p))), table="Sales")
    query = m.to_dax_query(
        SUMMARIZECOLUMNS(p["Color"], {"Share": share}), order_by=[p["Color"], DESC]
    )
    assert query == "\n".join(
        [
            "DEFINE",
            "    MEASURE Sales[Total Sales] = SUM(Sales[Amount])",
            "    MEASURE Sales[Share] = "
            "DIVIDE([Total Sales], CALCULATE([Total Sales], ALL(Product)))",
            "EVALUATE",
            '    SUMMARIZECOLUMNS(Product[Color], "Share", [Share])',
            "ORDER BY",
            "    Product[Color] DESC",
        ]
    )
    with pytest.raises(TypeError, match="table expression"):
        m.to_dax_query(total)
