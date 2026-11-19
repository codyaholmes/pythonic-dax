"""A Power BI style sales report in Streamlit, calculated by pythonic-dax.

Run it from the repository root:

    pip install -e ".[streamlit]"
    streamlit run examples/sales_report/app.py

Click a bar or a table row to filter the other visuals; click it again to clear.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

import model as sales_model

import pythonic_dax.streamlit as pst

st.set_page_config(page_title="Sales report", layout="wide")


@pst.cached_model
def load():
    return sales_model.build()


model = load()
Sales, Product, Customer, Date = (model.table(n) for n in ("Sales", "Product", "Customer", "Date"))
m = model.__getitem__

st.title("Sales report")
st.caption("Made-up bike shop data. Every number is a DAX measure evaluated by pythonic-dax.")

report = pst.report(model)

with st.sidebar:
    st.header("Filters")
    pst.dropdown(report, Date["Year"], default=2024)
    pst.date_range(report, Date["Date"])
    pst.multiselect(report, Customer["Region"])
    pst.multiselect(report, Customer["Segment"])
    pst.dropdown(report, Product["Category"])

cards = st.columns(5)
with cards[0]:
    pst.card(report, m("Sales Amount"), delta=m("Sales YoY %"))
with cards[1]:
    pst.card(report, m("Margin"))
with cards[2]:
    pst.card(report, m("Margin %"))
with cards[3]:
    pst.card(report, m("Orders"))
with cards[4]:
    pst.card(report, m("Units"))

left, right = st.columns([3, 2])
with left:
    st.subheader("Sales by month")
    pst.bar_chart(report, Date["Month"], m("Sales Amount"), key="by_month")
with right:
    st.subheader("Sales by region")
    pst.bar_chart(
        report,
        Customer["Region"],
        m("Sales Amount"),
        key="by_region",
        horizontal=True,
        sort_by=m("Sales Amount"),
        descending=True,
    )

st.subheader("Products")
pst.table(
    report,
    [Product["Product"]],
    [
        m("Sales Amount"),
        m("Sales PY"),
        m("Sales YoY %"),
        m("Margin"),
        m("Margin %"),
        m("Units"),
    ],
    key="by_product",
    sort_by=m("Sales Amount"),
    descending=True,
)


def describe(selection) -> str:
    if selection.values is not None:
        shown = ", ".join("(Blank)" if v is None else str(v) for v in selection.values)
        return f"{selection.column.display_name}: {shown}"
    start = f"{selection.start:%m/%d/%Y}" if selection.start else "start"
    end = f"{selection.end:%m/%d/%Y}" if selection.end else "end"
    return f"{selection.column.display_name}: {start} to {end}"


if report.selections:
    st.caption("Filters: " + "; ".join(describe(s) for s in report.selections))
