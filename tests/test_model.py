from __future__ import annotations

import datetime as dt
import sqlite3
from pathlib import Path

import pandas as pd
import polars as pl
import pytest

import pythonic_dax as pdax
from pythonic_dax import CrossFilter, DaxType, ModelError, SummarizeBy
from pythonic_dax.sources import CSV, Database, Parquet


def product_df() -> pl.DataFrame:
    return pl.DataFrame({"ProductKey": [1, 2, 3], "Color": ["Red", "Green", "Blue"]})


def sales_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ProductKey": [1, 1, 2, 9, None],
            "Amount": [10.0, 5.5, 7.25, 100.0, 1.0],
            "OrderDate": [dt.date(2026, 1, d) for d in (1, 2, 2, 3, 3)],
            "ShipDate": [dt.date(2026, 1, d) for d in (2, 3, 3, 4, 4)],
        }
    )


def date_df() -> pl.DataFrame:
    return pl.DataFrame(
        {"Date": pl.date_range(dt.date(2026, 1, 1), dt.date(2026, 1, 31), eager=True)}
    ).with_columns(MonthName=pl.lit("January"), MonthNo=pl.lit(1))


@pytest.fixture
def model() -> pdax.Model:
    m = pdax.Model(description="Test model")
    m.add_table("Product", product_df())
    m.add_table("Sales", sales_df())
    return m


# --- tables and columns ---------------------------------------------------------


def test_add_table_infers_dax_types(model: pdax.Model) -> None:
    sales = model.table("Sales")
    assert sales["ProductKey"].data_type is DaxType.INTEGER
    assert sales["Amount"].data_type is DaxType.DECIMAL
    assert sales["OrderDate"].data_type is DaxType.DATETIME
    assert sales.data.schema["OrderDate"] == pl.Datetime("us")
    assert len(sales) == 5


def test_names_are_case_insensitive_like_dax(model: pdax.Model) -> None:
    assert model.table("sales") is model.table("Sales")
    assert model.table("Sales")["amount"] is model.table("Sales")["Amount"]
    with pytest.raises(ModelError, match="already exists"):
        model.add_table("SALES", sales_df())


def test_attribute_access_to_columns(model: pdax.Model) -> None:
    sales = model.table("Sales")
    assert sales.Amount is sales["Amount"]
    with pytest.raises(AttributeError):
        _ = sales.Nope


def test_duplicate_column_names_ignoring_case_are_rejected() -> None:
    frame = pl.DataFrame({"a": [1], "A": [2]})
    with pytest.raises(ModelError, match="unique ignoring case"):
        pdax.Model().add_table("T", frame)


def test_table_names_must_be_trimmed() -> None:
    with pytest.raises(ModelError, match="spaces"):
        pdax.Model().add_table(" T", product_df())


def test_data_type_override_and_column_metadata() -> None:
    m = pdax.Model()
    t = m.add_table(
        "Exams",
        pl.DataFrame({"Fee": [12.345], "VOID_FL": [False]}),
        columns={
            "Fee": {"data_type": "currency", "format_string": "$#,0.00"},
            "VOID_FL": {"description": "A flag for voids.", "display_name": "Voided?"},
        },
        description="One row per exam.",
    )
    assert t["Fee"].data_type is DaxType.CURRENCY
    assert t.data["Fee"].dtype == pl.Decimal(19, 4)
    assert t["Fee"].format_string == "$#,0.00"
    assert t["VOID_FL"].description == "A flag for voids."
    assert t["VOID_FL"].display_name == "Voided?"
    assert t.description == "One row per exam."


def test_column_props_for_unknown_column_fail() -> None:
    with pytest.raises(KeyError, match="Nope"):
        pdax.Model().add_table("T", product_df(), columns={"Nope": {"description": "x"}})


def test_display_name_falls_back_to_model_name(model: pdax.Model) -> None:
    col = model.table("Product")["Color"]
    assert col.display_name == "Color"
    col.set(display_name="Product Color")
    assert col.display_name == "Product Color"
    col.set(display_name=None)
    assert col.display_name == "Color"


def test_unknown_property_is_rejected(model: pdax.Model) -> None:
    with pytest.raises(TypeError, match="no property 'descripton'"):
        model.table("Product")["Color"].set(descripton="typo")


def test_summarize_by_defaults_follow_power_bi(model: pdax.Model) -> None:
    assert model.table("Sales")["Amount"].summarize_by is SummarizeBy.SUM
    assert model.table("Product")["Color"].summarize_by is SummarizeBy.NONE
    with pytest.raises(ModelError, match="summarize_by"):
        model.table("Sales")["Amount"].set(summarize_by="median")


def test_data_category_is_validated(model: pdax.Model) -> None:
    model.table("Product")["Color"].set(data_category="WebUrl")
    with pytest.raises(ModelError, match="data category"):
        model.table("Product")["Color"].set(data_category="Colour")


def test_sort_by_column_requires_one_sort_value_per_value() -> None:
    m = pdax.Model()
    d = m.add_table("Date", date_df())
    d["MonthName"].set(sort_by_column="MonthNo")
    assert d["MonthName"].sort_by_column is d["MonthNo"]
    with pytest.raises(ModelError, match="more than one value"):
        d["MonthNo"].set(sort_by_column="Date")  # one MonthNo, many Dates
    with pytest.raises(ModelError, match="itself"):
        d["MonthNo"].set(sort_by_column="MonthNo")


# --- date tables ----------------------------------------------------------------


def test_mark_as_date_table() -> None:
    m = pdax.Model()
    d = m.add_table("Date", date_df()).mark_as_date_table("Date")
    assert d.is_date_table and d.date_column is d["Date"]


@pytest.mark.parametrize(
    ("dates", "message"),
    [
        ([dt.date(2026, 1, 1), dt.date(2026, 1, 3)], "gaps"),
        ([dt.date(2026, 1, 1), dt.date(2026, 1, 1)], "duplicate"),
        ([dt.date(2026, 1, 1), None], "blanks"),
        ([dt.datetime(2026, 1, 1, 12)], "times of day"),
    ],
)
def test_date_table_rules(dates: list[object], message: str) -> None:
    m = pdax.Model()
    t = m.add_table("D", pl.DataFrame({"Date": dates}))
    with pytest.raises(ModelError, match=message):
        t.mark_as_date_table("Date")


# --- relationships --------------------------------------------------------------


def test_relationship_and_blank_row(model: pdax.Model) -> None:
    sales, product = model.table("Sales"), model.table("Product")
    rel = model.add_relationship(
        sales["ProductKey"], product["ProductKey"], description="Each sale's product."
    )
    assert rel.from_table is sales and rel.to_table is product
    assert rel.description == "Each sale's product."
    # Key 9 and the blank key have no product: they belong to Product's blank row.
    assert rel.unmatched_rows == 2
    assert product.has_blank_row
    assert not sales.has_blank_row


def test_columns_given_one_side_first_are_swapped(model: pdax.Model) -> None:
    rel = model.add_relationship(
        model.table("Product")["ProductKey"], model.table("Sales")["ProductKey"]
    )
    assert rel.from_table.name == "Sales"


def test_duplicate_keys_on_one_side_are_rejected() -> None:
    m = pdax.Model()
    a = m.add_table("A", pl.DataFrame({"k": [1, 1]}))
    b = m.add_table("B", pl.DataFrame({"k": [1, 1]}))
    with pytest.raises(ModelError, match="duplicate values"):
        m.add_relationship(a["k"], b["k"])


def test_text_keys_are_unique_ignoring_case() -> None:
    m = pdax.Model()
    fact = m.add_table("F", pl.DataFrame({"code": ["a", "a"]}))
    dim = m.add_table("D", pl.DataFrame({"code": ["A", "a"]}))
    with pytest.raises(ModelError, match="duplicate"):
        m.add_relationship(fact["code"], dim["code"])


def test_text_keys_match_ignoring_case() -> None:
    m = pdax.Model()
    fact = m.add_table("F", pl.DataFrame({"code": ["a", "B", "c"]}))
    dim = m.add_table("D", pl.DataFrame({"code": ["A", "b"]}))
    assert m.add_relationship(fact["code"], dim["code"]).unmatched_rows == 1


def test_related_columns_must_share_a_type() -> None:
    m = pdax.Model()
    a = m.add_table("A", pl.DataFrame({"k": [1]}))
    b = m.add_table("B", pl.DataFrame({"k": ["1"]}))
    with pytest.raises(ModelError, match="same data type"):
        m.add_relationship(a["k"], b["k"])


def test_relationship_to_same_table_is_rejected(model: pdax.Model) -> None:
    sales = model.table("Sales")
    with pytest.raises(ModelError, match="different tables"):
        model.add_relationship(sales["ProductKey"], sales["Amount"])


def test_one_to_one_filters_both_ways() -> None:
    m = pdax.Model()
    a = m.add_table("A", pl.DataFrame({"k": [1, 2]}))
    b = m.add_table("B", pl.DataFrame({"k": [1, 2]}))
    rel = m.add_relationship(a["k"], b["k"], cardinality="one_to_one")
    assert rel.cross_filter is CrossFilter.BOTH


def test_many_to_many_is_not_supported_yet(model: pdax.Model) -> None:
    with pytest.raises(pdax.NotSupportedError):
        model.add_relationship(
            model.table("Sales")["ProductKey"],
            model.table("Product")["ProductKey"],
            cardinality="many_to_many",
        )


def _role_playing_model() -> tuple[pdax.Model, pdax.Table, pdax.Table]:
    m = pdax.Model()
    sales = m.add_table("Sales", sales_df())
    date = m.add_table("Date", date_df())
    return m, sales, date


def test_second_relationship_between_same_tables_must_be_inactive() -> None:
    m, sales, date = _role_playing_model()
    m.add_relationship(sales["OrderDate"], date["Date"])
    with pytest.raises(ModelError, match="already an active relationship"):
        m.add_relationship(sales["ShipDate"], date["Date"])
    ship = m.add_relationship(sales["ShipDate"], date["Date"], active=False)
    assert not ship.is_active
    assert [r.is_active for r in m.relationships] == [True, False]


def test_set_active_checks_ambiguity() -> None:
    m, sales, date = _role_playing_model()
    order = m.add_relationship(sales["OrderDate"], date["Date"])
    ship = m.add_relationship(sales["ShipDate"], date["Date"], active=False)
    with pytest.raises(ModelError):
        m.set_active(ship)
    assert not ship.is_active
    m.set_active(order, False)
    m.set_active(ship)
    assert ship.is_active and not order.is_active


def _diamond(cross_filter: str = "single") -> tuple[pdax.Model, list[pdax.Table]]:
    # Date filters Sales directly, and Date filters Product which filters Sales.
    m = pdax.Model()
    date = m.add_table("Date", pl.DataFrame({"d": [1, 2]}))
    product = m.add_table("Product", pl.DataFrame({"k": [1, 2], "d": [1, 2]}))
    sales = m.add_table("Sales", pl.DataFrame({"k": [1, 2], "d": [1, 2]}))
    m.add_relationship(sales["k"], product["k"], cross_filter=cross_filter)
    m.add_relationship(product["d"], date["d"])
    return m, [date, product, sales]


def test_two_filter_paths_between_tables_are_rejected() -> None:
    m, (date, _, sales) = _diamond()
    with pytest.raises(ModelError, match="more than one filter path"):
        m.add_relationship(sales["d"], date["d"])
    m.add_relationship(sales["d"], date["d"], active=False)  # fine when inactive


def test_shared_dimensions_between_two_fact_tables_are_allowed() -> None:
    # Two fact tables both related to two dimensions: a cycle, but filters only
    # flow one way, so there is a single path between any two tables.
    m = pdax.Model()
    d1 = m.add_table("D1", pl.DataFrame({"k": [1]}))
    d2 = m.add_table("D2", pl.DataFrame({"k": [1]}))
    f1 = m.add_table("F1", pl.DataFrame({"a": [1], "b": [1]}))
    f2 = m.add_table("F2", pl.DataFrame({"a": [1], "b": [1]}))
    for f in (f1, f2):
        m.add_relationship(f["a"], d1["k"])
        m.add_relationship(f["b"], d2["k"])
    assert len(m.relationships) == 4


def test_both_direction_filters_can_create_ambiguity() -> None:
    m = pdax.Model()
    d1 = m.add_table("D1", pl.DataFrame({"k": [1]}))
    d2 = m.add_table("D2", pl.DataFrame({"k": [1]}))
    f1 = m.add_table("F1", pl.DataFrame({"a": [1], "b": [1]}))
    f2 = m.add_table("F2", pl.DataFrame({"a": [1], "b": [1]}))
    m.add_relationship(f1["a"], d1["k"], cross_filter="both")
    m.add_relationship(f1["b"], d2["k"])
    m.add_relationship(f2["a"], d1["k"])
    with pytest.raises(ModelError, match="more than one filter path"):
        # D2 -> F1 -> D1 -> F2, and now D2 -> F2 directly.
        m.add_relationship(f2["b"], d2["k"])


# --- measures -------------------------------------------------------------------


def test_add_measure_with_metadata(model: pdax.Model) -> None:
    measure = model.add_measure(
        "Avg Revenue per Exam",
        expression="placeholder until Phase 2",
        table="Sales",
        description="Total revenue divided by the number of exams.",
        display_name="Average Revenue per Exam",
        format_string="$#,0.00",
        display_folder="Revenue",
    )
    assert model["avg revenue per exam"] is measure
    assert measure.table is model.table("Sales")
    assert measure.display_name == "Average Revenue per Exam"
    assert measure.ref == "[Avg Revenue per Exam]"


def test_measure_home_table_defaults_to_first_table(model: pdax.Model) -> None:
    assert model.add_measure("M", None).table is model.table("Product")


def test_measure_names_are_unique_and_dont_clash_with_columns(model: pdax.Model) -> None:
    model.add_measure("Total", None)
    with pytest.raises(ModelError, match="already exists"):
        model.add_measure("TOTAL", None)
    with pytest.raises(ModelError, match="column"):
        model.add_measure("amount", None, table="Sales")


def test_measure_needs_a_table() -> None:
    with pytest.raises(ModelError, match="home table"):
        pdax.Model().add_measure("M", None)


# --- describe -------------------------------------------------------------------


def test_describe(model: pdax.Model) -> None:
    model.table("Product")["Color"].set(description="Marketing color.")
    model.add_relationship(model.table("Sales")["ProductKey"], model.table("Product")["ProductKey"])
    model.add_measure("Total", None, description="All sales.")
    info = model.describe()
    cols = info["columns"].filter(pl.col("column") == "Color")
    assert cols["description"].to_list() == ["Marketing color."]
    assert info["measures"]["description"].to_list() == ["All sales."]
    assert info["relationships"]["unmatched_rows"].to_list() == [2]
    assert info["tables"]["rows"].to_list() == [3, 5]


# --- sources and refresh ----------------------------------------------------------


def test_load_from_parquet(tmp_path: Path) -> None:
    for i, chunk in enumerate(sales_df().iter_slices(2)):
        chunk.write_parquet(tmp_path / f"part{i}.parquet")
    m = pdax.Model()
    sales = m.add_table("Sales", Parquet(str(tmp_path / "*.parquet"), columns=["Amount"]))
    assert sales.data.columns == ["Amount"]
    assert len(sales) == 5


def test_load_from_database() -> None:
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE product (ProductKey INTEGER, Color TEXT)")
    con.executemany("INSERT INTO product VALUES (?, ?)", [(1, "Red"), (2, "Green")])
    m = pdax.Model()
    product = m.add_table("Product", Database("SELECT * FROM product", con))
    assert product["ProductKey"].data_type is DaxType.INTEGER
    assert product.data["Color"].to_list() == ["Red", "Green"]
    assert "database query" in product.source.describe()


def test_load_from_csv_and_pandas(tmp_path: Path) -> None:
    path = tmp_path / "p.csv"
    product_df().write_csv(path)
    m = pdax.Model()
    assert len(m.add_table("A", CSV(path))) == 3
    assert m.add_table("B", pd.DataFrame({"x": [1.5]}))["x"].data_type is DaxType.DECIMAL


def test_unsupported_data_is_rejected() -> None:
    with pytest.raises(ModelError, match="DataFrame"):
        pdax.Model().add_table("T", [1, 2, 3])


def test_refresh_reloads_and_revalidates(tmp_path: Path) -> None:
    path = tmp_path / "product.parquet"
    product_df().write_parquet(path)
    m = pdax.Model()
    product = m.add_table("Product", Parquet(path))
    sales = m.add_table("Sales", sales_df())
    product["Color"].set(description="kept across refreshes")
    rel = m.add_relationship(sales["ProductKey"], product["ProductKey"])
    assert rel.unmatched_rows == 2

    pl.DataFrame({"ProductKey": [1, 2, 9], "Color": ["Red", "Green", "Teal"]}).write_parquet(path)
    m.refresh()
    assert product.data["Color"].to_list() == ["Red", "Green", "Teal"]
    assert product["Color"].description == "kept across refreshes"
    assert rel.unmatched_rows == 1

    # A refresh that breaks the model is rolled back.
    pl.DataFrame({"ProductKey": [1, 1], "Color": ["Red", "Red"]}).write_parquet(path)
    with pytest.raises(ModelError, match="duplicate"):
        m.refresh()
    assert product.data["Color"].to_list() == ["Red", "Green", "Teal"]
