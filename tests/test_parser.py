"""The DAX text parser: syntax, name resolution, and that it builds what Python builds."""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

import pythonic_dax as pdax
from pythonic_dax import (
    ALL,
    CALCULATE,
    DIVIDE,
    FILTER,
    SUM,
    SUMX,
    VAR,
    DaxError,
    NotSupportedError,
)
from pythonic_dax._parser import parse_ast, split_query, tokenize


@pytest.fixture
def model() -> pdax.Model:
    m = pdax.Model()
    m.add_table("Product", pl.DataFrame({"ProductKey": [1, 2, 3], "Color": ["Red", "Blue", "Red"]}))
    m.add_table(
        "Sales Order",
        pl.DataFrame({"ProductKey": [1, 1, 2, 3], "Amount": [10.0, 5.0, 2.5, 1.0]}),
    )
    m.add_relationship(m.table("Sales Order")["ProductKey"], m.table("Product")["ProductKey"])
    return m


def value(model: pdax.Model, text: str) -> object:
    return model.evaluate_dax(f'EVALUATE ROW("v", {text})').item()


# --- tokens and syntax ---------------------------------------------------------------


def test_tokens_skip_comments_and_read_quoted_names() -> None:
    tokens = tokenize("'Sales ''Q1'''[Amount]] x] // note\n-- more\n/* block */ 1.5e2")
    assert [(t.kind, t.value) for t in tokens[:-1]] == [
        ("QUOTED", "Sales 'Q1'"),
        ("BRACKET", "Amount] x"),
        ("NUM", 150.0),
    ]


def test_syntax_errors_say_where() -> None:
    with pytest.raises(DaxError, match=r"line 2, column 5"):
        parse_ast("1 +\n2 + )")
    with pytest.raises(DaxError, match="never closed"):
        parse_ast('"open')


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("1 + 2 * 3", 7),
        ("(1 + 2) * 3", 9),
        ("2 ^ 3 ^ 2", 512.0),  # right to left, as in Power BI
        ("(2 ^ 3) ^ 2", 64.0),
        ("-2 ^ 2", -4.0),  # ^ binds tighter than the sign
        ("2 ^ -1", 0.5),
        ('"a" & 1 + 2', "a3"),  # & is below + and -
        ("1 < 2 && 3 < 4 || FALSE", True),
        ("NOT 1 = 2 && TRUE", True),  # NOT binds below comparisons
        ("NOT(TRUE) || TRUE", True),
        ("3 IN {1, 2, 3}", True),
        ('(2, "b") IN {(1, "a"), (2, "b")}', None),
        ("1 == BLANK()", False),
        ("BLANK() = 0", True),
        ("TRUE", True),
        ('dt"2024-03-05"', dt.datetime(2024, 3, 5)),
        ("VAR a = 2 VAR b = a * 3 RETURN a + b", 8),
        ("VAR a = VAR b = 4 RETURN b * 2 RETURN a + 1", 9),
        ('IF(1 > 2, "x", "y")', "y"),
        ('SWITCH(TRUE(), 1 > 2, "a", 2 > 1, "b", "c")', "b"),
        ("DIVIDE(1, , )", None),
    ],
)
def test_operators_and_constructs(model: pdax.Model, text: str, expected: object) -> None:
    if text.startswith("(2,"):
        with pytest.raises(NotSupportedError):
            value(model, text)
        return
    if text == "DIVIDE(1, , )":
        with pytest.raises(DaxError):
            value(model, text)
        return
    assert value(model, text) == expected


# --- building the same expressions as Python ---------------------------------------


def test_parse_builds_what_python_builds(model: pdax.Model) -> None:
    sales = model.table("Sales Order")
    product = model.table("Product")
    total = model.add_measure("Total", SUM(sales["Amount"]), table=sales)
    cases = {
        "SUM('Sales Order'[Amount])": SUM(sales["Amount"]),
        "calculate([Total], all(Product))": CALCULATE(total, ALL(product)),
        "Sales Order[Total]".replace("Sales Order", "'Sales Order'"): total,
        'SUMX(FILTER(Product, Product[Color] = "Red"), [Total])': SUMX(
            FILTER(product, product["Color"] == "Red"), total
        ),
        "VAR t = [Total] RETURN DIVIDE(t, 2)": VAR(t=total).RETURN(lambda v: DIVIDE(v.t, 2)),
    }
    for text, built in cases.items():
        assert model.parse_dax(text).to_dax() == built.to_dax(), text


def test_names_are_case_insensitive(model: pdax.Model) -> None:
    model.add_measure("Total", SUM(model.table("Sales Order")["Amount"]))
    assert value(model, "CALCULATE([total], 'sales order'[AMOUNT] > 2)") == 17.5
    assert value(model, "var X = 1 return x + 1") == 2


def test_unknown_names(model: pdax.Model) -> None:
    with pytest.raises(DaxError, match="Cannot find table 'Nope'"):
        model.parse_dax("SUM(Nope[x])")
    with pytest.raises(DaxError, match="Column 'Price' in table 'Product' cannot be found"):
        model.parse_dax("SUM(Product[Price])")
    with pytest.raises(DaxError, match="Failed to resolve name 'Nope'"):
        model.parse_dax("COUNTROWS(Nope)")
    with pytest.raises(NotSupportedError, match=r"PERCENTILE\.INC"):
        model.parse_dax("PERCENTILE.INC(Product[ProductKey], 0.5)")


def test_python_checks_apply_to_text(model: pdax.Model) -> None:
    with pytest.raises(DaxError):
        model.parse_dax("SUM(Product[Color])")  # SUM of text, refused as Power BI does


def test_calculated_column_names_its_own_columns(model: pdax.Model) -> None:
    sales = model.table("Sales Order")
    col = model.add_column(sales, "Double", model.parse_dax("[Amount] * 2", table=sales))
    assert col.expression.to_dax() == "'Sales Order'[Amount] * 2"
    assert col.values.to_list() == [20.0, 10.0, 5.0, 2.0]


def test_unqualified_column_reads_the_row_context(model: pdax.Model) -> None:
    # Checked in Power BI Desktop: SUMX(Sales, [Qty]) sums Sales[Qty].
    assert value(model, "SUMX('Sales Order', [Amount] * 2)") == 37.0
    with pytest.raises(DaxError, match="cannot be found"):
        value(model, "[Amount] + 1")  # no row context


def test_names_of_table_expression_columns(model: pdax.Model) -> None:
    text = 'SUMX(ADDCOLUMNS(Product, "Twice", Product[ProductKey] * 2), [Twice])'
    assert value(model, text) == 12


def test_column_variations_are_refused(model: pdax.Model) -> None:
    with pytest.raises(NotSupportedError, match="variations"):
        model.parse_dax("MAX(Product[ProductKey].[Year])")


def test_datatable_and_info(model: pdax.Model) -> None:
    frame = model.evaluate_dax(
        'EVALUATE DATATABLE("Name", STRING, "N", INTEGER, {{"a", 1}, {"b", BLANK()}})'
    )
    assert frame.rows() == [("a", 1), ("b", None)]
    assert model.evaluate_dax("EVALUATE INFO.VIEW.TABLES()").height == 2


def test_year_end_text_in_totalytd(model: pdax.Model) -> None:
    dates = model.add_calculated_table(
        "Dates", model.parse_dax("CALENDAR(DATE(2024, 1, 1), DATE(2024, 12, 31))")
    )
    node = model.parse_dax('TOTALYTD(COUNTROWS(Dates), Dates[Date], "6/30")')
    assert node.to_dax() == 'TOTALYTD(COUNTROWS(Dates), Dates[Date], "6/30")'
    assert len(dates) == 366


# --- queries ---------------------------------------------------------------------


def test_query_definitions_in_any_order_last_only_for_the_query(model: pdax.Model) -> None:
    result = model.evaluate_dax(
        """
        DEFINE
            MEASURE Product[Share] = DIVIDE([Total], CALCULATE([Total], ALL(Product)))
            MEASURE 'Sales Order'[Total] = SUM('Sales Order'[Amount])
            COLUMN Product[Big] = [Total] > 5
            VAR Colors = {"Red"}
        EVALUATE
            SUMMARIZECOLUMNS(
                Product[Color], Product[Big],
                FILTER(ALL(Product[Color]), Product[Color] IN Colors),
                "Share", [Share]
            )
        """
    )
    assert sorted(result.rows()) == [("Red", False, 1 / 18.5), ("Red", True, 15 / 18.5)]
    assert model.measures == []
    assert "Big" not in model.table("Product")


def test_query_measure_replaces_a_model_measure(model: pdax.Model) -> None:
    total = model.add_measure(
        "Total", SUM(model.table("Sales Order")["Amount"]), table="Sales Order"
    )
    double = model.add_measure("Double", total * 2)
    query = "DEFINE MEASURE 'Sales Order'[Total] = 1 EVALUATE ROW(\"d\", [Double])"
    assert model.evaluate_dax(query).item() == 2
    assert model.evaluate(pdax.ROW("d", double)).item() == 37.0  # restored afterwards


def test_query_variable_inside_a_measure_is_refused(model: pdax.Model) -> None:
    with pytest.raises(NotSupportedError, match="query variable"):
        model.evaluate_dax('DEFINE VAR x = 1 MEASURE Product[M] = x EVALUATE ROW("m", [M])')


def test_split_query_keeps_vars_inside_definitions() -> None:
    query = split_query(
        'DEFINE MEASURE T[A] = VAR x = 1 VAR y = 2 RETURN x + y VAR q = 3 EVALUATE ROW("a", 1)'
    )
    assert [(d.kind, d.name) for d in query.definitions] == [("MEASURE", "A"), ("VAR", "q")]
    assert query.definitions[0].text.strip() == "VAR x = 1 VAR y = 2 RETURN x + y"


def test_queries_need_one_evaluate_and_no_order_by(model: pdax.Model) -> None:
    with pytest.raises(NotSupportedError, match="EVALUATE"):
        model.evaluate_dax('EVALUATE ROW("a", 1) EVALUATE ROW("b", 2)')
    with pytest.raises(NotSupportedError, match="ORDER BY"):
        model.evaluate_dax("EVALUATE Product ORDER BY Product[Color]")
