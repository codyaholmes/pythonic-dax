"""Importing semantic models from TMDL folders and model.bim files."""

from __future__ import annotations

import copy
import datetime as dt
import json
from pathlib import Path
from typing import Any

import polars as pl
import pytest

import pythonic_dax as pdax
from pythonic_dax import DaxType, ModelError, NotSupportedError, _tmdl
from pythonic_dax.importing import build
from pythonic_dax.sources import Parquet

FIXTURES = Path(__file__).parent / "fixtures"
TMDL = FIXTURES / "shop.SemanticModel"
BIM = FIXTURES / "shop.bim"


def product() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "product_key": [1, 2, 3],
            "color": ["Red", "Blue", "Red"],
            "launch_date": [dt.date(2023, 1, 1)] * 3,
            "not_in_model": [0, 0, 0],
        }
    )


def sales() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "product_key": [1, 1, 2, 3, 4],
            "order_date": [
                dt.date(2024, m, d) for m, d in ((1, 5), (2, 5), (2, 9), (3, 1), (3, 2))
            ],
            "ship_date": [
                dt.date(2024, m, d) for m, d in ((1, 9), (3, 1), (2, 10), (3, 3), (3, 9))
            ],
            "qty": [1, 2, 3, 4, 5],
            "price": [10.0, 10.0, 5.5, 2.25, 1.0],
        }
    )


def data() -> dict[str, pl.DataFrame]:
    return {"Product": product(), "Sales": sales()}


@pytest.fixture(params=["tmdl", "bim"])
def model(request: pytest.FixtureRequest) -> pdax.Model:
    if request.param == "tmdl":
        return pdax.from_tmdl(TMDL, data=data(), culture="es-ES")
    return pdax.from_bim(BIM, data=data(), culture="es-ES")


# --- the TMDL reader ---------------------------------------------------------------


def test_tmdl_reads_objects_properties_and_expressions() -> None:
    raw = _tmdl.read_folder(TMDL / "definition")
    tables = {t["name"]: t for t in raw["model"]["tables"]}
    assert list(tables)[:4] == ["Product", "Sales", "Dates", "_Measures"]  # model.tmdl order
    key = tables["Product"]["columns"][0]
    assert key["description"] == "The product's key."
    assert key["isHidden"] is True and key["formatString"] == "0"
    line_total = tables["Sales"]["columns"][-1]
    assert line_total["type"] == "calculated"
    assert line_total["expression"] == "// quantity times price\nSales[Qty] * Sales[Price]"
    share = next(m for m in tables["_Measures"]["measures"] if m["name"] == "Red Share")
    assert share["expression"].splitlines()[-1] == "    DIVIDE ( red, total )"
    dates = tables["Dates"]["partitions"][0]["source"]
    assert dates["type"] == "calculated" and dates["expression"].startswith(
        "ADDCOLUMNS (\n    CALENDAR"
    )
    rel = raw["model"]["relationships"][1]
    assert (rel["fromTable"], rel["fromColumn"], rel["toTable"]) == ("Sales", "Order Date", "Dates")


def test_tmdl_quoted_names_and_unknown_objects() -> None:
    nodes = _tmdl.parse(
        "table 'It''s'\n\tcolumn 'A = B'\n\t\tdataType: string\n"
        "\tsomethingNew Thing\n\t\tsetting: 1\n"
    )
    table = nodes[0]
    assert table.name == "It's"
    assert table.children[0].name == "A = B"
    assert table.children[1].type == "somethingNew"  # kept as an object, not an error


def test_tmdl_quoted_sort_by_column() -> None:
    # Power BI writes sortByColumn: 'Year Month' when the name has a space
    nodes = _tmdl.parse(
        "table Dates\n\tcolumn 'Month Name'\n\t\tdataType: string\n"
        "\t\tsortByColumn: 'Year Month'\n"
        "\tcolumn 'Year Month'\n\t\tdataType: int64\n"
    )
    table = _tmdl.to_bim(nodes)["model"]["tables"][0]
    assert table["columns"][0]["sortByColumn"] == "Year Month"


def test_tmdl_errors_name_the_file_and_line(tmp_path: Path) -> None:
    (tmp_path / "model.tmdl").write_text("model Model\n\t\t\tculture: en-US\n")
    with pytest.raises(ModelError, match=r"model.tmdl, line 2: unexpected indentation"):
        _tmdl.read_folder(tmp_path)
    with pytest.raises(ModelError, match=r"no model\.tmdl"):
        _tmdl.read_folder(FIXTURES)


# --- the imported model -----------------------------------------------------------------


def test_tables_columns_and_types(model: pdax.Model) -> None:
    assert [t.name for t in model.tables] == ["Product", "Sales", "_Measures", "Dates"]
    product = model.table("Product")
    assert [c.name for c in product.columns] == [
        "ProductKey",
        "Color",
        "Launch Date",
        "Product Sales",
    ]
    assert model.table("Sales")["Price"].data_type is DaxType.CURRENCY  # dataType: decimal
    assert model.table("Sales")["Qty"].values.to_list() == [1, 2, 3, 4, 5]  # from "qty"
    assert len(model.table("_Measures")) == 0


def test_automatic_date_tables_are_left_out() -> None:
    model = pdax.from_tmdl(TMDL, data=data())
    assert not any(t.name.startswith("LocalDateTable") for t in model.tables)
    assert any("automatic date table" in note for note in model.import_notes)


def test_metadata(model: pdax.Model) -> None:
    product = model.table("Product")
    key = product["ProductKey"]
    assert (key.description, key.is_hidden, key.is_key, key.format_string) == (
        "The product's key.",
        True,
        True,
        "0",
    )
    assert key.annotations == {"SummarizationSetBy": "Automatic"}
    assert product.description == "One row per product."
    line_total = model.table("Sales")["Line Total"]
    assert (line_total.display_folder, line_total.summarize_by.value) == ("Amounts", "sum")
    total = model["Total Sales"]
    assert (total.format_string, total.display_folder, total.description) == (
        "$#,0.00",
        "Sales",
        "Sum of line totals.",
    )
    assert model["Units"].is_hidden
    assert model.description == "A made-up shop model for import tests."
    dates = model.table("Dates")
    assert dates.is_date_table and dates.date_column is dates["Date"]
    assert dates["Month"].sort_by_column is dates["MonthNo"]


def test_translations_become_display_names(model: pdax.Model) -> None:
    assert model.table("Product").display_name == "Producto"
    assert model.table("Product")["Color"].display_name == "Color del producto"
    assert model["Total Sales"].display_name == "Ventas totales"
    assert model.table("Dates").hierarchy("Calendar").display_name == "Calendario"
    assert model["Units"].display_name == "Units"  # no translation: the model name


def test_hierarchies(model: pdax.Model) -> None:
    calendar = model.table("Dates").hierarchy("Calendar")
    assert [(lv.name, lv.column.name, lv.ordinal) for lv in calendar.levels] == [
        ("Year", "Year", 0),
        ("Month", "Month", 1),
    ]


def test_relationships(model: pdax.Model) -> None:
    rels = {(r.from_column.ref, r.to_column.ref): r for r in model.relationships}
    assert set(rels) == {
        ("Sales[ProductKey]", "Product[ProductKey]"),
        ("Sales[Order Date]", "Dates[Date]"),
        ("Sales[Ship Date]", "Dates[Date]"),
    }
    assert not rels[("Sales[Ship Date]", "Dates[Date]")].is_active


def test_measures_parsed_from_dax(model: pdax.Model) -> None:
    assert model["Shipped Sales"].expression.to_dax() == (
        "CALCULATE([Total Sales], USERELATIONSHIP(Sales[Ship Date], Dates[Date]))"
    )
    assert model.evaluate(pdax.ROW("u", model["Units"])).item() == 15


def test_unsupported_measure_imports_but_refuses_to_evaluate(model: pdax.Model) -> None:
    median = model["Median Qty"]
    assert median.expression.to_dax() == "PERCENTILE.INC(Sales[Qty], 0.5)"
    with pytest.raises(
        NotSupportedError, match=r"PERCENTILE.INC \(in the measure \[Median Qty\]\)"
    ):
        model.evaluate(pdax.ROW("m", median))
    assert any("Median Qty" in note for note in model.import_notes)


def test_refresh_recomputes_calculated_objects(tmp_path: Path) -> None:
    path = tmp_path / "sales.parquet"
    sales().write_parquet(path)
    model = pdax.from_tmdl(TMDL, data=lambda name: Parquet(path) if name == "Sales" else product())
    assert model.table("Product")["Product Sales"].values.to_list() == [30.0, 16.5, 9.0]
    sales().with_columns(qty=pl.lit(1)).write_parquet(path)
    model.refresh()
    assert model.table("Product")["Product Sales"].values.to_list() == [20.0, 5.5, 2.25]
    assert len(model.table("Dates")) == 91


# --- options and errors --------------------------------------------------------------


def test_missing_and_extra_data() -> None:
    with pytest.raises(ModelError, match=r"no data given for these tables: \['Sales'\]"):
        pdax.from_tmdl(TMDL, data={"Product": product()})
    with pytest.raises(ModelError, match="aren't import tables"):
        pdax.from_tmdl(TMDL, data={**data(), "Dates": product()})
    with pytest.raises(ModelError, match=r"has no column for \['Qty \(source column qty\)'\]"):
        pdax.from_tmdl(TMDL, data={"Product": product(), "Sales": sales().drop("qty")})


def test_columns_match_by_model_name_too() -> None:
    renamed = sales().rename({"qty": "Qty"})
    model = pdax.from_tmdl(TMDL, data={"Product": product(), "Sales": renamed})
    assert model.table("Sales")["Qty"].values.sum() == 15


def test_overrides_and_skip_tables() -> None:
    model = pdax.from_tmdl(TMDL, data=data(), overrides={"Median Qty": "AVERAGE(Sales[Qty])"})
    assert model.evaluate(pdax.ROW("m", model["Median Qty"])).item() == 3.0
    model = pdax.from_tmdl(TMDL, data=data(), overrides={"units": pdax.BLANK()})
    assert model.evaluate(pdax.ROW("u", model["Units"])).item() is None
    with pytest.raises(ModelError, match="overrides name objects the model doesn't have"):
        pdax.from_tmdl(TMDL, data=data(), overrides={"Nope": "1"})
    with pytest.raises(ModelError, match="skip_tables names tables"):
        pdax.from_tmdl(TMDL, data=data(), skip_tables=["Nope"])


def spec() -> dict[str, Any]:
    return json.loads(BIM.read_text())  # type: ignore[no-any-return]


def test_refused_features() -> None:
    raw = spec()
    raw["model"]["tables"][0]["partitions"][0]["mode"] = "directQuery"
    with pytest.raises(NotSupportedError, match="directquery storage mode"):
        build(raw, data())

    raw = spec()
    raw["model"]["tables"].append({"name": "Calc", "calculationGroup": {}, "columns": []})
    with pytest.raises(NotSupportedError, match="calculation groups"):
        build(raw, data())
    assert "Calc" not in [t.name for t in build(raw, data(), skip_tables=["Calc"]).tables]

    raw = spec()
    raw["model"]["relationships"][0]["toCardinality"] = "many"
    with pytest.raises(NotSupportedError, match="many-to-many"):
        build(raw, data())

    raw = spec()
    raw["model"]["functions"] = [{"name": "Double", "expression": "(x) => x * 2"}]
    with pytest.raises(NotSupportedError, match="user-defined functions"):
        build(raw, data())


def test_calculated_column_with_an_error_is_left_out() -> None:
    # Power BI keeps such a column in an error state (e.g. a table that was deleted)
    raw = spec()
    product_table = raw["model"]["tables"][0]
    calc = next(c for c in product_table["columns"] if c.get("type") == "calculated")
    calc["expression"] = "COUNTROWS('No Such Table')"
    model = build(raw, data())
    assert calc["name"] not in model.table(product_table["name"])
    assert any("whose DAX gives an error" in n for n in model.import_notes)


def test_unsupported_calculated_column_names_itself() -> None:
    raw = spec()
    product_table = raw["model"]["tables"][0]
    calc = next(c for c in product_table["columns"] if c.get("type") == "calculated")
    calc["expression"] = "PERCENTILE.INC(Sales[Qty], 0.5)"
    with pytest.raises(
        NotSupportedError, match=r"in the calculated column Product\[Product Sales\]"
    ):
        build(raw, data())
    fixed = build(raw, data(), overrides={"Product[Product Sales]": "1"})
    assert fixed.table("Product")["Product Sales"].values.to_list() == [1, 1, 1]


def test_circular_measures() -> None:
    raw = spec()
    measures = next(t for t in raw["model"]["tables"] if t["name"] == "_Measures")["measures"]
    measures.append({"name": "A", "expression": "[B] + 1"})
    measures.append({"name": "B", "expression": "[A] + 1"})
    model = build(copy.deepcopy(raw), data())
    with pytest.raises(NotSupportedError, match="circular dependency"):
        model.evaluate(pdax.ROW("a", model["A"]))


def test_measures_can_be_left_out() -> None:
    full = pdax.from_tmdl(TMDL, data=data())
    model = pdax.from_tmdl(TMDL, data=data(), measures=False)
    # Product[Product Sales] = [Total Sales], so that measure is kept.
    assert [m.name for m in model.measures] == ["Total Sales"]
    assert any("left out" in n and "Total Sales" in n for n in model.import_notes)
    product = model.table("Product")
    assert model.evaluate(product).equals(full.evaluate(full.table("Product")))


def test_calculated_columns_can_be_left_out() -> None:
    model = pdax.from_tmdl(TMDL, data=data(), calculated_columns=False)
    assert "Line Total" not in model.table("Sales")
    assert "Product Sales" not in model.table("Product")
    assert "left out 2 calculated columns, as asked" in model.import_notes
    assert model.value(model["Units"]) == 15
    # [Total Sales] = SUM(Sales[Line Total]) still imports, and refuses when used.
    with pytest.raises(NotSupportedError, match=r"left out with calculated_columns=False"):
        model.value(model["Total Sales"])


def test_only_named_measures_are_kept() -> None:
    model = pdax.from_tmdl(TMDL, data=data(), measures=["red share"], calculated_columns=False)
    # [Red Share] uses [Total Sales], so that one is kept too.
    assert sorted(m.name for m in model.measures) == ["Red Share", "Total Sales"]
    assert any("kept 2, the ones named and those they use" in n for n in model.import_notes)
    with pytest.raises(ModelError, match="doesn't have"):
        pdax.from_tmdl(TMDL, data=data(), measures=["Nope"])
    with pytest.raises(TypeError):
        pdax.from_tmdl(TMDL, data=data(), measures="Units")  # type: ignore[arg-type]


def test_left_out_columns_drop_what_needs_them() -> None:
    raw = spec()
    tables = {t["name"]: t for t in raw["model"]["tables"]}
    sales = tables["Sales"]
    sales["columns"].append(
        {
            "name": "Key2",
            "type": "calculated",
            "dataType": "int64",
            "expression": "Sales[ProductKey]",
        }
    )
    qty = next(c for c in sales["columns"] if c["name"] == "Qty")
    qty["sortByColumn"] = "Key2"
    raw["model"]["relationships"].append(
        {
            "name": "r9",
            "fromTable": "Sales",
            "fromColumn": "Key2",
            "toTable": "Product",
            "toColumn": "ProductKey",
            "isActive": False,
        }
    )
    model = build(raw, data(), calculated_columns=False)
    assert model.table("Sales")["Qty"].sort_by_column is None
    plain = build(spec(), data(), calculated_columns=False)
    assert len(model.relationships) == len(plain.relationships)
    assert any("left out the relationship" in n and "Key2" in n for n in model.import_notes)
