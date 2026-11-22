"""DAX format strings converted for Streamlit (streamlit_format and streamlit_values).

The golden case tests/golden/cases/streamlit/format.yaml proves the converted formats
show what Power BI shows. These tests pin down the conversions, the refusals, and that
the module's copies of Streamlit's formatting match Streamlit's own frontend code.
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import json
import shutil
import subprocess
from decimal import Decimal
from pathlib import Path

import polars as pl
import pytest

import pythonic_dax as pdax
from golden.harness import CASES_DIR, load_case, run_case
from pythonic_dax._display import render_moment, render_printf

CONVERSIONS = [
    ("#,0", "%,.0f"),
    ("#,##0.00", "%,.2f"),
    ("0.0", "%.1f"),
    ("0.0%", "%.1f%%"),
    ("0.00%;-0.00%;0.00%", "%.2f%%"),
    ("#,0.00;-#,0.00;#,0.00", "%,.2f"),
    ('#,0,"K"', "%,.0fK"),
    ('#,0.0,,"M"', "%,.1fM"),  # commas after the point don't scale
    ('0" units"', "%.0f units"),
    ("Fixed", "%.2f"),
    ("Standard", "%,.2f"),
    ("Percent", "%.2f%%"),
    ("yyyy-mm-dd", "YYYY-MM-DD"),
    ("mmm yyyy", "MMM YYYY"),
    ("dddd, d mmmm", "dddd, D MMMM"),
    ('"Q"q yyyy', "[Q]Q YYYY"),
    ("h:nn AM/PM", "h:mm A"),
    ("hh:mm", "HH:mm"),  # mm after an hour is minutes
    ("hh:nn:ss am/pm", "hh:mm:ss a"),
    ("Short Date", "M/D/YYYY"),
    ("Long Date", "dddd, MMMM D, YYYY"),
]


@pytest.mark.parametrize(("dax", "streamlit"), CONVERSIONS)
def test_conversion(dax: str, streamlit: str) -> None:
    assert pdax.streamlit_format(dax) == streamlit


REFUSED = [
    "",
    "General Number",
    "General Date",
    "Scientific",
    "0.00E+00",
    "0.##",  # optional decimals
    "#.00",  # no digit before the point
    "000",  # zero padding
    "0;(0)",  # negatives in parentheses
    "Currency",  # ($1.00) for negatives
    "$#,0",  # Power BI shows -$5, Streamlit $-5
    '0;-0;"zero"',  # a text zero section
    "0.0;-0.0;0",  # a zero section that differs
]


@pytest.mark.parametrize("dax", REFUSED)
def test_refused(dax: str) -> None:
    with pytest.raises(pdax.NotSupportedError):
        pdax.streamlit_format(dax)


def test_nonnegative_allows_formats_that_only_differ_for_negatives() -> None:
    assert pdax.streamlit_format("$#,0", nonnegative=True) == "$%,.0f"
    assert pdax.streamlit_format("Currency", nonnegative=True) == "$%,.2f"
    assert pdax.streamlit_format("0;(0)", nonnegative=True) == "%.0f"


def test_values_are_scaled_and_rounded_like_power_bi() -> None:
    assert pdax.streamlit_values(1.005, "0.00") == 1.01  # the browser alone shows 1.00
    assert pdax.streamlit_values(0.256, "0.0%") == 25.6
    assert pdax.streamlit_values(1234567, '#,0,"K"') == 1235.0
    assert pdax.streamlit_values(Decimal("2.5"), "0") == 3.0
    shown = pdax.streamlit_values(-0.004, "0.00")
    assert shown == 0 and str(shown) == "0.0"  # no -0.0, which Streamlit shows as -0.00
    assert pdax.streamlit_values(None, "0.00") is None


def test_values_keep_the_container() -> None:
    series = pl.Series("Sales", [1.005, None, -2.5])
    out = pdax.streamlit_values(series, "#,0.00")
    assert out.name == "Sales" and out.dtype == pl.Float64
    assert out.to_list() == [1.01, None, -2.5]
    assert pdax.streamlit_values([0.5, 0.25], "0%") == [50.0, 25.0]
    dates = pl.Series("Date", [dt.date(2024, 3, 5)])
    assert pdax.streamlit_values(dates, "mmm yyyy").equals(dates)


def test_values_from_pandas_come_back_as_polars() -> None:
    pd = pytest.importorskip("pandas")
    out = pdax.streamlit_values(pd.Series([0.1234], name="Margin"), "0.0%")
    assert isinstance(out, pl.Series) and out.to_list() == [12.3]


def test_values_that_would_show_differently_are_refused() -> None:
    with pytest.raises(pdax.NotSupportedError, match=r"\(\$5\.00\)"):
        pdax.streamlit_values(-5, "Currency", nonnegative=True)
    with pytest.raises(pdax.NotSupportedError):
        pdax.streamlit_values("text", "0.00")
    with pytest.raises(pdax.NotSupportedError):
        pdax.streamlit_values(True, "0")
    with pytest.raises(pdax.NotSupportedError):
        pdax.streamlit_values(45000, "yyyy-mm-dd")  # a serial number, not a date


def test_streamlit_golden_case_runs_without_refusals() -> None:
    """The golden case would only be an expected failure if streamlit_values refused
    one of its values; make sure it really runs."""
    case = load_case(CASES_DIR / "streamlit" / "format.yaml")
    assert run_case(case).height == 1


# --- the copies of Streamlit's formatting against Streamlit itself -----------------------

PRINTF_CASES = [
    ("%,.2f", 1234567.891),
    ("%,.2f", -1234.5),
    ("%,.2f", 1.005),  # the browser rounds the binary double down
    ("%.2f", -0.004),  # -0.00
    ("%.2f", -0.0),
    ("%.0f", 2.5),
    ("%.0f", -2.5),
    ("%.0f", 0.5),
    ("%,.0f", 999.5),
    ("%,.0f", 123456789012.0),
    ("%.1f%%", 25.6),
    ("$%,.2f", -42.5),  # $-42.50
    ("%,.1fM", -1.2345),
    ("%.0f units", 42.0),
    ("%.3f", 0.0005),
    ("%,.2f", 0.125),
    ("%,.2f", 1e15 + 0.25),
]
MOMENT_CASES = [
    ("YYYY-MM-DD", dt.datetime(2024, 3, 5)),
    ("dddd, MMMM D, YYYY", dt.datetime(2024, 3, 5)),
    ("ddd D MMM YY", dt.datetime(2009, 12, 27)),
    ("[Q]Q YYYY", dt.datetime(2024, 8, 5)),
    ("h:mm A", dt.datetime(2024, 3, 5, 13, 4, 5)),
    ("hh:mm:ss a", dt.datetime(2024, 3, 5, 0, 5, 9)),
    ("HH:mm", dt.datetime(2024, 3, 5, 9, 7)),
    ("H:m:s", dt.datetime(2024, 3, 5, 9, 7, 3)),
    ("M/D/YYYY", dt.datetime(999, 1, 2)),
]


def _streamlit_js_dir() -> Path | None:
    spec = importlib.util.find_spec("streamlit")
    if spec is None or spec.origin is None:
        return None
    path = Path(spec.origin).parent / "static" / "static" / "js"
    return path if path.is_dir() else None


def test_python_copies_match_streamlit_frontend() -> None:
    node = shutil.which("node")
    js_dir = _streamlit_js_dir()
    if node is None or js_dir is None:
        pytest.skip("needs node and the streamlit package")
    cases = [{"printf": f, "value": v} for f, v in PRINTF_CASES] + [
        {"moment": f, "utc": [v.year, v.month, v.day, v.hour, v.minute, v.second]}
        for f, v in MOMENT_CASES
    ]
    script = Path(__file__).with_name("streamlit_render.mjs")
    done = subprocess.run(
        [node, str(script), str(js_dir)],
        input=json.dumps(cases),
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert done.returncode == 0, done.stderr
    streamlit = json.loads(done.stdout)
    ours = [render_printf(f, v) for f, v in PRINTF_CASES] + [
        render_moment(f, v) for f, v in MOMENT_CASES
    ]
    assert ours == streamlit


def test_render_printf_examples() -> None:
    # What the browser shows, from Streamlit's sprintf-js fork and JavaScript's toFixed.
    assert render_printf("%,.2f", 1.005) == "1.00"
    assert render_printf("%.2f", -0.004) == "-0.00"
    assert render_printf("%.2f", -0.0) == "0.00"
    assert render_printf("$%,.2f", -42.5) == "$-42.50"
    assert render_printf("%.0f", -2.5) == "-3"
