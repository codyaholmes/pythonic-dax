"""Write the demo's made-up data to data/*.csv. Run once; the CSVs are committed so the
same rows can be loaded into Power BI Desktop to compare numbers."""

from __future__ import annotations

import datetime as dt
import random
from pathlib import Path

import polars as pl

HERE = Path(__file__).parent
rng = random.Random(20261119)

PRODUCTS = [
    # key, name, category, color, list price, unit cost
    (1, "Trail 500", "Bikes", "Red", 1200.0, 780.0),
    (2, "Trail 700", "Bikes", "Black", 1650.0, 1050.0),
    (3, "Road 300", "Bikes", "Silver", 980.0, 640.0),
    (4, "Road 900", "Bikes", "Red", 2400.0, 1500.0),
    (5, "City Cruiser", "Bikes", "Blue", 640.0, 410.0),
    (6, "Helmet Pro", "Accessories", "Black", 85.0, 34.0),
    (7, "Helmet Lite", "Accessories", "White", 49.0, 18.0),
    (8, "Bottle Cage", "Accessories", "Silver", 12.0, 4.0),
    (9, "Bike Light", "Accessories", "Black", 35.0, 13.0),
    (10, "Lock", "Accessories", "Black", 42.0, 17.0),
    (11, "Jersey", "Clothing", "Blue", 65.0, 26.0),
    (12, "Gloves", "Clothing", "Black", 28.0, 10.0),
    (13, "Shorts", "Clothing", "Black", 55.0, 21.0),
    (14, "Rain Jacket", "Clothing", "Yellow", 120.0, 52.0),
]
REGIONS = ["North", "South", "East", "West"]
SEGMENTS = ["Consumer", "Business"]


def main() -> None:
    (HERE / "data").mkdir(exist_ok=True)
    pl.DataFrame(
        PRODUCTS,
        schema=["ProductKey", "Product", "Category", "Color", "ListPrice", "UnitCost"],
        orient="row",
    ).write_csv(HERE / "data" / "products.csv")

    customers = []
    for key in range(1, 61):
        customers.append((key, f"Customer {key:02d}", rng.choice(REGIONS), rng.choice(SEGMENTS)))
    pl.DataFrame(
        customers, schema=["CustomerKey", "Customer", "Region", "Segment"], orient="row"
    ).write_csv(HERE / "data" / "customers.csv")

    sales = []
    start = dt.date(2023, 1, 1)
    order = 1000
    for day in range(731):
        date = start + dt.timedelta(days=day)
        growth = 1.0 + 0.25 * (date.year - 2023)
        season = 1.3 if date.month in (5, 6, 7) else (0.8 if date.month in (1, 2) else 1.0)
        for _ in range(rng.randint(2, 9)):
            if rng.random() > 0.55 * growth * season:
                continue
            order += 1
            customer = rng.randint(1, 60)
            for _ in range(rng.randint(1, 3)):
                product = rng.choice(PRODUCTS)
                qty = rng.randint(1, 2) if product[2] == "Bikes" else rng.randint(1, 6)
                discount = rng.choice([0.0, 0.0, 0.0, 0.05, 0.10])
                price = round(product[4] * (1 - discount), 2)
                sales.append((order, date, customer, product[0], qty, price, product[5]))
    pl.DataFrame(
        sales,
        schema=[
            "OrderID",
            "OrderDate",
            "CustomerKey",
            "ProductKey",
            "Quantity",
            "UnitPrice",
            "UnitCost",
        ],
        orient="row",
    ).write_csv(HERE / "data" / "sales.csv")
    print(len(sales), "sales rows")


if __name__ == "__main__":
    main()
