"""Phase 8: the framework-neutral report layer and the Streamlit add-on."""

from __future__ import annotations

import datetime as dt
import sys
from decimal import Decimal
from pathlib import Path

import polars as pl
import pytest

import pythonic_dax as pdax
from pythonic_dax import CALCULATE, DIVIDE, SUM
from pythonic_dax.report import Report, format_value

ROOT = Path(__file__).parent.parent


@pytest.fixture
def model() -> pdax.Model:
    m = pdax.Model()
    product = m.add_table(
        "Product",
        pl.DataFrame(
            {
                "ProductKey": [1, 2, 3],
                "Color": ["Red", "Blue", "Red"],
                "Size": ["M", "L", "S"],
                "SizeOrder": [2, 3, 1],
            }
        ),
        columns={
            "Color": {"description": "Paint color.", "display_name": "Colour"},
            "SizeOrder": {"is_hidden": True},
        },
    )
    product["Size"].sort_by_column = product["SizeOrder"]
    sales = m.add_table(
        "Sales",
        pl.DataFrame(
            {
                "ProductKey": [1, 1, 2, 3, 3],
                "Date": [dt.datetime(2024, 1, d) for d in (1, 2, 3, 4, 5)],
                "Amount": [10.0, 5.0, 2.5, 1.0, 100.0],
            }
        ),
    )
    m.add_relationship(sales["ProductKey"], product["ProductKey"])
    total = m.add_measure(
        "Total", SUM(sales["Amount"]), table=sales, format_string="#,0.00", description="All."
    )
    m.add_measure(
        "Share",
        DIVIDE(total, CALCULATE(total, pdax.ALL(product))),
        table=sales,
        format_string="0.0%",
    )
    return m


def test_card_is_formatted_and_labelled(model: pdax.Model) -> None:
    card = Report(model).card(model["Total"])
    assert (card.value, card.text, card.label, card.tooltip) == (118.5, "118.50", "Total", "All.")


def test_table_has_a_total_row_evaluated_like_power_bi(model: pdax.Model) -> None:
    product = model.table("Product")
    visual = Report(model).table([product["Color"]], [model["Total"], model["Share"]])
    assert visual.total == 2
    assert visual.data.columns == ["Product[Color]", "[Total]", "[Share]"]
    assert visual.data.rows() == [
        ("Blue", 2.5, 2.5 / 118.5),
        ("Red", 116.0, 116 / 118.5),
        (None, 118.5, 1.0),
    ]
    assert visual.text.columns == ["Colour", "Total", "Share"]
    assert visual.text.rows()[-1] == ("Total", "118.50", "100.0%")
    assert visual.fields[0].tooltip == "Paint color."
    assert visual.body().height == 2


def test_rows_follow_the_sort_by_column(model: pdax.Model) -> None:
    product = model.table("Product")
    visual = Report(model).table([product["Size"]], [model["Total"]], totals=False)
    assert visual.data["Product[Size]"].to_list() == ["S", "M", "L"]
    by_value = Report(model).table(
        [product["Size"]], [model["Total"]], totals=False, sort_by=model["Total"]
    )
    assert by_value.data["[Total]"].to_list() == [2.5, 15.0, 101.0]


def test_slicers_filter_and_clear(model: pdax.Model) -> None:
    report = Report(model)
    product, sales = model.table("Product"), model.table("Sales")
    report.select(product["Color"], ["Red"])
    assert report.card(model["Total"]).value == 116.0
    report.select_range(sales["Date"], dt.date(2024, 1, 2), dt.date(2024, 1, 4))
    assert report.card(model["Total"]).value == 6.0
    report.select_range(sales["Date"], None, dt.date(2024, 1, 1))
    assert report.card(model["Total"]).value == 10.0
    report.select(product["Color"], [])
    report.clear(sales["Date"])
    assert report.card(model["Total"]).value == 118.5


def test_cross_filter_skips_the_source_visual(model: pdax.Model) -> None:
    report = Report(model)
    product = model.table("Product")
    report.select(product["Color"], ["Blue"], source="colors")
    own = report.table([product["Color"]], [model["Total"]], name="colors")
    assert own.body().height == 2  # the clicked visual still shows every color
    other = report.table([product["Size"]], [model["Total"]], name="sizes")
    assert other.body()["Product[Size]"].to_list() == ["L"]
    report.clear(product["Color"])  # a slicer clear doesn't touch the visual's selection
    assert len(report.selections) == 1


def test_choices_and_fields(model: pdax.Model) -> None:
    report = Report(model)
    assert report.choices(model.table("Product")["Size"]) == ["S", "M", "L"]
    keys = [f.key for f in report.fields()]
    assert "Product[SizeOrder]" not in keys
    assert {"[Total]", "[Share]", "Product[Color]"} <= set(keys)


def test_format_value() -> None:
    assert format_value(None, "0.00") == ""
    assert format_value(Decimal("1234.5"), "$#,0.00") == "$1,234.50"
    assert format_value(dt.datetime(2024, 3, 5), "m/d/yyyy") == "3/5/2024"
    with pytest.raises(pdax.NotSupportedError):
        format_value(1.5, None, "de-DE")


def test_core_does_not_import_streamlit() -> None:
    import subprocess

    code = "import pythonic_dax, pythonic_dax.report, sys; print('streamlit' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_demo_app_runs() -> None:
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(ROOT / "examples" / "sales_report" / "app.py"), default_timeout=120)
    app.run()
    assert not app.exception
    labels = [m.label for m in app.metric]
    assert labels == ["Sales", "Margin", "Margin %", "Orders", "Units"]
    assert app.metric[0].value.startswith("$")
    app.sidebar.selectbox[0].select_index(0).run()  # Year: All
    assert not app.exception
