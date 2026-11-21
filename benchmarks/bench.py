"""Time typical report visuals on a made-up sales model with 1M or 10M fact rows.

    python benchmarks/bench.py --rows 1000000
    python benchmarks/bench.py --rows 10000000 --repeat 3

The model has the same tables, relationships and measures as the demo in
examples/sales_report, with a larger Sales table. Data is generated with a fixed seed,
so runs are comparable. Each visual is timed from the query to the formatted result,
the way a report page asks for it.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples" / "sales_report"))

import pythonic_dax as pdax  # noqa: E402
from pythonic_dax import (  # noqa: E402
    CALCULATE,
    DISTINCTCOUNT,
    DIVIDE,
    SAMEPERIODLASTYEAR,
    SUM,
    SUMX,
    TOTALYTD,
)
from pythonic_dax.report import Report  # noqa: E402

FIRST, LAST = dt.date(2023, 1, 1), dt.date(2024, 12, 31)


def make_data(rows: int, seed: int = 20261121) -> dict[str, pl.DataFrame]:
    """Products and customers like the demo's, and ``rows`` order lines."""
    from make_data import PRODUCTS, REGIONS, SEGMENTS  # type: ignore[import-not-found]

    rng = np.random.default_rng(seed)
    products = pl.DataFrame(
        PRODUCTS,
        schema=["ProductKey", "Product", "Category", "Color", "ListPrice", "UnitCost"],
        orient="row",
    )
    customers = 2_000
    customer = pl.DataFrame(
        {
            "CustomerKey": np.arange(1, customers + 1),
            "Customer": [f"Customer {k:04d}" for k in range(1, customers + 1)],
            "Region": rng.choice(REGIONS, customers),
            "Segment": rng.choice(SEGMENTS, customers),
        }
    )
    days = (LAST - FIRST).days + 1
    product_keys = rng.integers(1, len(PRODUCTS) + 1, rows)
    list_price = products["ListPrice"].to_numpy()[product_keys - 1]
    unit_cost = products["UnitCost"].to_numpy()[product_keys - 1]
    discount = rng.choice([0.0, 0.0, 0.0, 0.05, 0.10], rows)
    sales = pl.DataFrame(
        {
            "OrderID": np.arange(rows) // 2 + 1000,
            "OrderDate": pl.Series(
                np.datetime64(FIRST) + rng.integers(0, days, rows).astype("timedelta64[D]")
            ).cast(pl.Date),
            "CustomerKey": rng.integers(1, customers + 1, rows),
            "ProductKey": product_keys,
            "Quantity": rng.integers(1, 7, rows),
            "UnitPrice": np.round(list_price * (1 - discount), 2),
            "UnitCost": unit_cost,
        }
    )
    return {"Sales": sales, "Product": products, "Customer": customer}


def build(rows: int) -> pdax.Model:
    from model import calendar  # type: ignore[import-not-found]

    data = make_data(rows)
    model = pdax.Model()
    sales = model.add_table(
        "Sales",
        data["Sales"],
        columns={"UnitPrice": {"data_type": "currency"}, "UnitCost": {"data_type": "currency"}},
    )
    product = model.add_table("Product", data["Product"])
    customer = model.add_table("Customer", data["Customer"])
    dates = model.add_table("Date", calendar(FIRST, LAST))
    dates["Month"].sort_by_column = dates["MonthKey"]
    dates.mark_as_date_table("Date")
    model.add_relationship(sales["OrderDate"], dates["Date"])
    model.add_relationship(sales["ProductKey"], product["ProductKey"])
    model.add_relationship(sales["CustomerKey"], customer["CustomerKey"])

    revenue = model.add_measure(
        "Sales Amount", SUMX(sales, sales["Quantity"] * sales["UnitPrice"]), table=sales
    )
    cost = model.add_measure(
        "Total Cost", SUMX(sales, sales["Quantity"] * sales["UnitCost"]), table=sales
    )
    margin = model.add_measure("Margin", revenue - cost, table=sales)
    model.add_measure("Margin %", DIVIDE(margin, revenue), table=sales)
    model.add_measure("Units", SUM(sales["Quantity"]), table=sales)
    model.add_measure("Orders", DISTINCTCOUNT(sales["OrderID"]), table=sales)
    model.add_measure("Sales YTD", TOTALYTD(revenue, dates["Date"]), table=sales)
    prior = model.add_measure(
        "Sales PY", CALCULATE(revenue, SAMEPERIODLASTYEAR(dates["Date"])), table=sales
    )
    model.add_measure("Sales YoY %", DIVIDE(revenue - prior, prior), table=sales)
    return model


def visuals(model: pdax.Model) -> dict[str, Callable[[Report], Any]]:
    m = model.__getitem__
    product, customer, dates = (model.table(n) for n in ("Product", "Customer", "Date"))
    return {
        "card: Sales": lambda r: r.card(m("Sales Amount")),
        "card: Orders": lambda r: r.card(m("Orders")),
        "card: Margin %": lambda r: r.card(m("Margin %")),
        "card: Sales YoY %": lambda r: r.card(m("Sales YoY %")),
        "chart: Sales by month (24)": lambda r: r.table([dates["Month"]], [m("Sales Amount")]),
        "chart: Sales by region": lambda r: r.table([customer["Region"]], [m("Sales Amount")]),
        "table: products x 6 measures": lambda r: r.table(
            [product["Product"]],
            [
                m("Sales Amount"),
                m("Sales PY"),
                m("Sales YoY %"),
                m("Margin"),
                m("Margin %"),
                m("Units"),
            ],
        ),
        "matrix: category x year, YTD": lambda r: r.table(
            [product["Category"], dates["Year"]], [m("Sales Amount"), m("Sales YTD")]
        ),
        "table: customers (2,000) x 3": lambda r: r.table(
            [customer["Customer"]], [m("Sales Amount"), m("Units"), m("Orders")]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rows", type=int, default=1_000_000)
    parser.add_argument("--repeat", type=int, default=1, help="runs per visual (best is shown)")
    parser.add_argument("--only", help="run only visuals whose name contains this text")
    parser.add_argument("--slicer", action="store_true", help="also filter Region = West")
    args = parser.parse_args()

    start = time.perf_counter()
    model = build(args.rows)
    print(f"model with {args.rows:,} sales rows built in {time.perf_counter() - start:.2f} s")
    report = Report(model)
    if args.slicer:
        report.select(model.table("Customer")["Region"], ["West"])
    worst = 0.0
    for name, visual in visuals(model).items():
        if args.only and args.only.lower() not in name.lower():
            continue
        best = float("inf")
        for _ in range(args.repeat):
            start = time.perf_counter()
            visual(report)
            best = min(best, time.perf_counter() - start)
        worst = max(worst, best)
        print(f"  {name:<34} {best * 1000:>9.0f} ms")
    print(f"slowest visual: {worst * 1000:.0f} ms")


if __name__ == "__main__":
    main()
