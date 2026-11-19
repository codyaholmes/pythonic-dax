"""The demo's semantic model. It doesn't import Streamlit, so the same model can back
any front end (see app.py for the Streamlit report)."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import polars as pl

import pythonic_dax as pdax
from pythonic_dax import (
    CALCULATE,
    DISTINCTCOUNT,
    DIVIDE,
    SAMEPERIODLASTYEAR,
    SUM,
    SUMX,
    TOTALYTD,
)

DATA = Path(__file__).parent / "data"


def calendar(first: dt.date, last: dt.date) -> pl.DataFrame:
    days = pl.date_range(first, last, "1d", eager=True)
    return pl.DataFrame({"Date": days}).with_columns(
        Year=pl.col("Date").dt.year().cast(pl.Int64),
        Quarter=pl.format("Q{}", pl.col("Date").dt.quarter()),
        Month=pl.col("Date").dt.strftime("%b %Y"),
        MonthKey=(pl.col("Date").dt.year() * 100 + pl.col("Date").dt.month()).cast(pl.Int64),
    )


def build() -> pdax.Model:
    model = pdax.Model(description="Made-up bike shop sales, 2023 to 2024.")
    sales = model.add_table(
        "Sales",
        pl.read_csv(DATA / "sales.csv", try_parse_dates=True),
        description="One row per order line.",
        columns={
            "OrderID": {"display_name": "Order", "summarize_by": "none"},
            "UnitPrice": {"data_type": "currency", "format_string": "$#,0.00"},
            "UnitCost": {"data_type": "currency", "format_string": "$#,0.00"},
            "CustomerKey": {"is_hidden": True},
            "ProductKey": {"is_hidden": True},
        },
    )
    product = model.add_table(
        "Product",
        pl.read_csv(DATA / "products.csv"),
        columns={
            "Category": {"description": "Bikes, Accessories or Clothing."},
            "ProductKey": {"is_hidden": True},
            "ListPrice": {"display_name": "List Price", "format_string": "$#,0.00"},
            "UnitCost": {"is_hidden": True},
        },
    )
    customer = model.add_table(
        "Customer",
        pl.read_csv(DATA / "customers.csv"),
        columns={
            "Region": {"description": "Sales region of the customer's address."},
            "Segment": {"description": "Consumer or Business buyer."},
            "CustomerKey": {"is_hidden": True},
        },
    )
    dates = model.add_table(
        "Date",
        calendar(dt.date(2023, 1, 1), dt.date(2024, 12, 31)),
        columns={
            "Date": {"format_string": "m/d/yyyy", "description": "Order date."},
            "MonthKey": {"is_hidden": True},
        },
    )
    dates["Month"].sort_by_column = dates["MonthKey"]
    dates.mark_as_date_table("Date")

    model.add_relationship(sales["OrderDate"], dates["Date"])
    model.add_relationship(sales["ProductKey"], product["ProductKey"])
    model.add_relationship(sales["CustomerKey"], customer["CustomerKey"])

    revenue = model.add_measure(
        "Sales Amount",
        SUMX(sales, sales["Quantity"] * sales["UnitPrice"]),
        table=sales,
        display_name="Sales",
        description="Quantity times unit price, after discounts.",
        format_string="$#,0",
        display_folder="Sales",
    )
    cost = model.add_measure(
        "Total Cost",
        SUMX(sales, sales["Quantity"] * sales["UnitCost"]),
        table=sales,
        format_string="$#,0",
        description="Quantity times unit cost.",
        display_folder="Sales",
    )
    margin = model.add_measure(
        "Margin",
        revenue - cost,
        table=sales,
        format_string="$#,0",
        description="Sales minus cost.",
        display_folder="Profit",
    )
    model.add_measure(
        "Margin %",
        DIVIDE(margin, revenue),
        table=sales,
        format_string="0.0%",
        description="Margin as a share of sales.",
        display_folder="Profit",
    )
    model.add_measure(
        "Units",
        SUM(sales["Quantity"]),
        table=sales,
        format_string="#,0",
        description="Items sold.",
        display_folder="Sales",
    )
    model.add_measure(
        "Orders",
        DISTINCTCOUNT(sales["OrderID"]),
        table=sales,
        format_string="#,0",
        description="Number of distinct orders.",
        display_folder="Sales",
    )
    model.add_measure(
        "Sales YTD",
        TOTALYTD(revenue, dates["Date"]),
        table=sales,
        format_string="$#,0",
        description="Sales from January 1 to the last date in context.",
        display_folder="Time intelligence",
    )
    prior = model.add_measure(
        "Sales PY",
        CALCULATE(revenue, SAMEPERIODLASTYEAR(dates["Date"])),
        table=sales,
        display_name="Sales Last Year",
        format_string="$#,0",
        description="Sales for the same dates one year earlier.",
        display_folder="Time intelligence",
    )
    model.add_measure(
        "Sales YoY %",
        DIVIDE(revenue - prior, prior),
        table=sales,
        display_name="Growth vs Last Year",
        format_string="+0.0%;-0.0%;0.0%",
        description="Change in sales over the same dates last year.",
        display_folder="Time intelligence",
    )
    return model
