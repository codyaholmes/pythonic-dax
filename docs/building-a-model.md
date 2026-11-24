# Building a model

This guide walks through building a semantic model with pythonic-dax, the same way you
would build one in Power BI Desktop: load tables, set column properties, add calculated
columns and measures, relate the tables, mark a date table, and then query it.

Every code block on this page runs, top to bottom, as one script. A test
(`tests/test_docs.py`) runs them on every change, so the examples stay correct.

**Contents**

1. [Sample data](#1-sample-data)
2. [The model](#2-the-model)
3. [Tables](#3-tables)
4. [Columns and their metadata](#4-columns-and-their-metadata)
5. [Data types](#5-data-types)
6. [Relationships](#6-relationships)
7. [The date table](#7-the-date-table)
8. [Calculated columns](#8-calculated-columns)
9. [Measures](#9-measures)
10. [Writing expressions: operators, VAR and parentheses](#10-writing-expressions-operators-var-and-parentheses)
11. [Querying the model](#11-querying-the-model)
12. [DAX text](#12-dax-text)
13. [Importing a Power BI model](#13-importing-a-power-bi-model)
14. [Describing the model](#14-describing-the-model)
15. [Errors](#15-errors)

## 1. Sample data

The examples use a small, made-up shop: products, customers, order lines and a
calendar. The first block writes the products to a Parquet file and the customers to a
SQLite database, so the [Tables](#3-tables) section can show each kind of source.

```python
import datetime as dt
import sqlite3

import polars as pl

products = pl.DataFrame(
    {
        "ProductKey": [1, 2, 3, 4],
        "Product": ["Road Bike", "Helmet", "Jersey", "Gloves"],
        "Category": ["Bikes", "Accessories", "Clothing", "Clothing"],
        "Color": ["Red", "Blue", "Red", "Black"],
        "ListPrice": [1200.0, 60.0, 45.0, 25.0],
    }
)
products.write_parquet("products.parquet")

connection = sqlite3.connect(":memory:")
connection.execute("CREATE TABLE customers (CustomerKey INTEGER, Name TEXT, Region TEXT)")
connection.executemany(
    "INSERT INTO customers VALUES (?, ?, ?)",
    [(1, "Ana", "West"), (2, "Ben", "East"), (3, "Cleo", "West")],
)

sales = pl.DataFrame(
    {
        "OrderID": [101, 101, 102, 103, 104, 105],
        "OrderDate": [
            dt.date(2024, 1, 5),
            dt.date(2024, 1, 5),
            dt.date(2024, 2, 9),
            dt.date(2024, 3, 1),
            dt.date(2025, 1, 20),
            dt.date(2025, 2, 14),
        ],
        "ShipDate": [
            dt.date(2024, 1, 8),
            dt.date(2024, 1, 8),
            dt.date(2024, 3, 2),
            dt.date(2024, 3, 4),
            dt.date(2025, 1, 22),
            dt.date(2025, 3, 1),
        ],
        "CustomerKey": [1, 1, 2, 3, 2, None],  # the last order has no customer
        "ProductKey": [1, 2, 3, 4, 2, 3],
        "Quantity": [1, 2, 3, 4, 1, 2],
        "UnitPrice": [1150.0, 60.0, 45.0, 22.5, 55.0, 45.0],
    }
)
```

## 2. The model

A `Model` holds tables, relationships and measures, like a Power BI semantic model.
Everything else hangs off it.

```python
import pythonic_dax as pdax

model = pdax.Model(description="A made-up bike shop.")
```

## 3. Tables

`model.add_table(name, data)` loads a table and returns it. As in Power BI's Import
mode, the data is read once and held in memory. `model.refresh()` reads every source
again, like a dataset refresh, and recomputes calculated columns and tables.

`data` can be:

| Data | What to pass |
|---|---|
| A Polars DataFrame or LazyFrame | the frame itself |
| A pandas DataFrame | the frame itself (it's converted to Polars) |
| Parquet files | `Parquet("path.parquet")`, a glob such as `Parquet("sales/*.parquet")`, a list of paths, or a cloud path with `storage_options` |
| A CSV file | `CSV("path.csv")` |
| A SQL query over an open connection | `Database("SELECT ...", connection)`, with any DBAPI2 connection (sqlite3, pyodbc, psycopg, duckdb, Snowflake) or a SQLAlchemy engine |
| A SQL query over a connection string | `DatabaseURI("SELECT ...", "postgresql://...")`, which uses connectorx or ADBC and is faster for large pulls |

The sources live in `pythonic_dax.sources`:

```python
from pythonic_dax.sources import Database, Parquet

Sales = model.add_table("Sales", sales, description="One row per order line.")
Product = model.add_table("Product", Parquet("products.parquet"))
Customer = model.add_table(
    "Customer",
    Database("SELECT * FROM customers", connection),
    description="People who placed an order.",
    display_name="Customers",
)
```

A table is also a DAX table reference, so `Sales` can go anywhere DAX takes a table,
such as `COUNTROWS(Sales)` or `FILTER(Sales, ...)`. Look a table up by name with
`model.table("Sales")`, and a column with `Sales["Quantity"]`. Names ignore case, as
in DAX.

```python
assert model.table("sales") is Sales
assert Sales["quantity"] is Sales["Quantity"]
assert len(Sales) == 6
Sales.data  # the stored rows, as a Polars DataFrame
```

Table properties are `description`, `display_name`, `is_hidden` and `annotations`. Set
them when you add the table, or later with `.set()`:

```python
Product.set(description="Everything the shop sells.")
```

## 4. Columns and their metadata

Every column carries the properties Power BI's model stores for it. All of them are
optional.

| Property | What it does |
|---|---|
| `description` | Text for tooltips and documentation pages. |
| `display_name` | The name reports show. DAX still uses the model name. Without one, `display_name` returns the model name. |
| `format_string` | A Power BI format string, such as `"$#,0.00"`, `"0.0%"` or `"mmm yyyy"`. |
| `display_folder` | The folder the field shows in, such as `"Sales"`, or `r"Sales\Detail"` for a subfolder. |
| `is_hidden` | Hides the field from field lists. It still works in DAX. |
| `sort_by_column` | Another column of the same table that orders this one, such as Month by MonthNumber. Each value must map to one sort value, as in Power BI. |
| `data_category` | A Power BI data category, such as `"City"`, `"Country"` or `"WebUrl"`. |
| `summarize_by` | The default aggregation: `"sum"`, `"none"`, `"count"`, `"average"`, `"min"`, `"max"`, `"distinct_count"` or `"default"`. |
| `is_key` | Marks a column that is unique per row. |
| `annotations` | A dictionary of your own text values. |

Set them in `add_table(columns=...)`, with `.set()`, or by assigning the attribute:

```python
Orders = model.add_table(
    "Orders",
    sales.select("OrderID", "OrderDate").unique(),
    columns={
        "OrderID": {"display_name": "Order", "summarize_by": "none", "is_key": True},
        "OrderDate": {"format_string": "m/d/yyyy", "description": "The day the order was placed."},
    },
    is_hidden=True,
)
Product["ListPrice"].set(display_name="List Price", format_string="$#,0.00")
Product["ProductKey"].is_hidden = True
Sales["ProductKey"].set(is_hidden=True)
Sales["CustomerKey"].set(is_hidden=True)
Sales["OrderID"].set(display_name="Order", summarize_by="none")

assert Product["ListPrice"].display_name == "List Price"
assert Product["Color"].display_name == "Color"  # falls back to the model name
```

An unknown property raises a `TypeError` that lists the ones you can set, so a typo
doesn't go unnoticed.

## 5. Data types

Power BI stores every column as one of six types, and so does pythonic-dax. Each source
type is converted the way Power BI's import converts it:

| DAX type | Name to pass | Stored as | Comes from |
|---|---|---|---|
| Whole Number | `"integer"` | `Int64` | integer columns |
| Decimal Number | `"decimal"` | `Float64` | float columns |
| Fixed Decimal Number (currency) | `"currency"` | `Decimal(19, 4)` | only when you ask for it |
| Date/Time | `"datetime"` | `Datetime("us")` | date and datetime columns (a date is midnight) |
| Text | `"string"` | `String` | text columns, compared without case |
| True/False | `"boolean"` | `Boolean` | boolean columns |

To use another type than the one inferred, give `data_type` in `columns`. The usual case
is money, which Power BI models as Fixed Decimal Number:

```python
from pythonic_dax import DaxType

Sales["UnitPrice"].data_type  # DaxType.DECIMAL, inferred from Float64

Returns = model.add_table(
    "Returns",
    pl.DataFrame({"OrderID": [102], "Refund": [45.0]}),
    columns={"Refund": {"data_type": "currency", "format_string": "$#,0.00"}},
)
assert Returns["Refund"].data_type is DaxType.CURRENCY
```

Measures over a currency column return currency too, and their results come back as
`decimal.Decimal` values (`Decimal(19, 4)` in a frame), exactly as Power BI computes
them. The [Streamlit guide](streamlit.md) shows how to turn them into floats for charts.

## 6. Relationships

`model.add_relationship(many_side_column, one_side_column)` relates two tables. Filters
flow from the one side to the many side, as in Power BI:

```python
from pythonic_dax import CrossFilter

model.add_relationship(Sales["ProductKey"], Product["ProductKey"])
to_customer = model.add_relationship(Sales["CustomerKey"], Customer["CustomerKey"])
model.add_relationship(Sales["OrderID"], Orders["OrderID"], cross_filter="both")
```

The rules are Power BI's:

* The one side can't have duplicate keys. If you give the columns the wrong way round,
  they are swapped, as Desktop does.
* Both columns must have the same data type.
* Only one active path may run between two tables. A second relationship between the
  same tables must be inactive (`active=False`).
* Rows on the many side whose key has no match (or is blank) belong to the one side's
  hidden blank row, so their totals still show up under a blank value. The sample's
  last order has no customer: `to_customer.unmatched_rows` is 1.

Options:

| Argument | Values | Default |
|---|---|---|
| `cardinality` | `"many_to_one"`, `"one_to_one"` (always filters both ways). Many-to-many raises `NotSupportedError`. | `"many_to_one"` |
| `cross_filter` | `"single"` (one side filters the many side) or `"both"` | `"single"` |
| `active` | `True` or `False` | `True` |
| `name`, `description` | text | a generated name |

```python
assert to_customer.unmatched_rows == 1
assert model.relationships[2].cross_filter is CrossFilter.BOTH
```

To change filter direction for one calculation only, use `CROSSFILTER` inside
`CALCULATE`, as in DAX. Inactive relationships are used through `USERELATIONSHIP`. Both
are shown in [Measures](#9-measures), after the date table they need.

`model.set_active(relationship, active)` switches which relationship is the active
one, and checks the one-active-path rule again.

## 7. The date table

Time intelligence (`TOTALYTD`, `SAMEPERIODLASTYEAR`, `DATEADD` and the rest) needs a
date table with one row per day. Build it as a DataFrame, mark it, and relate the
fact table's date columns to it. Here the order date is the active relationship, and
the ship date is an inactive one for `USERELATIONSHIP`:

```python
days = pl.date_range(dt.date(2024, 1, 1), dt.date(2025, 12, 31), "1d", eager=True)
Date = model.add_table(
    "Date",
    pl.DataFrame({"Date": days}).with_columns(
        Year=pl.col("Date").dt.year().cast(pl.Int64),
        Month=pl.col("Date").dt.strftime("%b"),
        MonthNumber=pl.col("Date").dt.month().cast(pl.Int64),
    ),
    columns={"Date": {"format_string": "m/d/yyyy"}, "MonthNumber": {"is_hidden": True}},
)
Date["Month"].sort_by_column = Date["MonthNumber"]  # Jan, Feb, Mar rather than Apr, Aug, Dec
Date.mark_as_date_table("Date")

model.add_relationship(Sales["OrderDate"], Date["Date"])
model.add_relationship(Sales["ShipDate"], Date["Date"], active=False)
```

`mark_as_date_table` checks what Power BI checks: the column is Date/Time, with no
blanks, no duplicates, no time of day, and no missing days between the first and last
date.

You can also build the calendar in DAX, as a calculated table:

```python
from pythonic_dax import CALENDAR, DATE, YEAR

Calendar = model.add_calculated_table(
    "Calendar", CALENDAR(DATE(2024, 1, 1), DATE(2025, 12, 31)), is_hidden=True
)
model.add_column(Calendar, "Year", YEAR(Calendar["Date"]))
assert [c.name for c in Calendar.columns] == ["Date", "Year"]
```

Calculated tables are evaluated with no filters when added, and again on `refresh()`.
Add their extra columns as [calculated columns](#8-calculated-columns), as above.

## 8. Calculated columns

`model.add_column(table, name, expression)` adds a calculated column, evaluated once for
each row, in that row's context, like a calculated column in Power BI:

```python
from pythonic_dax import IF, RELATED, SUM

line_total = model.add_column(
    Sales, "Line Total", Sales["Quantity"] * Sales["UnitPrice"], format_string="$#,0.00"
)
model.add_column(Sales, "Category", RELATED(Product["Category"]))
model.add_column(Sales, "Size", IF(line_total >= 100, "Large", "Small"))

Sales.data.select("OrderID", "Line Total", "Category", "Size")
```

* The type is worked out from the expression. Pass `data_type=` to override it.
* `add_column` takes the same metadata as a column does: `description`,
  `display_name`, `format_string`, `display_folder`, `is_hidden`, `data_category`,
  `summarize_by` and `annotations`.
* `RELATED` follows a relationship to the one side. `RELATEDTABLE` goes the other way
  and returns the related rows.
* A measure or `CALCULATE` in a calculated column turns the current row into a filter
  (context transition), as in Power BI. So a measure gives a per-row result:

```python
from pythonic_dax import COUNTROWS, RELATEDTABLE

model.add_column(Product, "Order Lines", COUNTROWS(RELATEDTABLE(Sales)))
```

Calculated columns are computed when added and on every `refresh()`.

## 9. Measures

`model.add_measure(name, expression)` adds a measure and returns it. Measures are
evaluated in the filter context of each query cell, like Power BI measures.

### A home table for measures

Every measure has a home table, as in Power BI. It decides nothing about the result;
it only says where the measure is listed. Many Power BI models keep their measures in a
table with no columns. Do the same with an empty DataFrame:

```python
Measures = model.add_table("Measures", pl.DataFrame(), description="All the measures.")
```

Pass `table=Measures` (or `table="Measures"`) to put a measure there. Without `table`,
a measure goes in the model's first table.

### Adding measures

```python
from pythonic_dax import CALCULATE, DISTINCTCOUNT, DIVIDE, SUMX

sales_amount = model.add_measure(
    "Sales Amount",
    SUMX(Sales, Sales["Quantity"] * Sales["UnitPrice"]),
    table=Measures,
    display_name="Sales",
    description="Quantity times unit price.",
    format_string="$#,0",
    display_folder="Sales",
)
model.add_measure(
    "Orders",
    DISTINCTCOUNT(Sales["OrderID"]),
    table=Measures,
    format_string="#,0",
    display_folder="Sales",
)
```

The properties are `description`, `display_name`, `format_string`, `display_folder`,
`is_hidden`, `data_category` and `annotations`, the same as for columns. They can be
changed later with `.set()`.

### Referring to other measures

A measure is also a reference to itself, like `[Sales Amount]` in DAX. Use it in other
expressions in either of two ways:

* **The returned value**, when the measure was defined in the same file:
  `sales_amount` above.
* **`model["Name"]`**, which looks a measure up by name (ignoring case). This works
  anywhere you have the model, such as a Streamlit page that imports it.

```python
from pythonic_dax import ALL

model.add_measure(
    "Avg Order Value",
    DIVIDE(sales_amount, model["Orders"]),
    table=Measures,
    format_string="$#,0.00",
)
model.add_measure(
    "Share of Sales",
    DIVIDE(sales_amount, CALCULATE(sales_amount, ALL(Product))),
    table=Measures,
    format_string="0.0%",
)
assert model["avg order value"].expression.to_dax() == "DIVIDE([Sales Amount], [Orders])"
```

Both are the same object, so a measure that refers to another one always gets its
current definition. A measure can't refer to itself (that raises the same circular
dependency error as Power BI).

### Time intelligence and relationships in measures

```python
from pythonic_dax import BOTH, CROSSFILTER, SAMEPERIODLASTYEAR, TOTALYTD, USERELATIONSHIP

model.add_measure("Sales YTD", TOTALYTD(sales_amount, Date["Date"]), table=Measures)
model.add_measure(
    "Sales LY", CALCULATE(sales_amount, SAMEPERIODLASTYEAR(Date["Date"])), table=Measures
)
model.add_measure(
    "Sales by Ship Date",
    CALCULATE(sales_amount, USERELATIONSHIP(Sales["ShipDate"], Date["Date"])),
    table=Measures,
)
model.add_measure(
    "Customers Buying",
    CALCULATE(
        COUNTROWS(Customer),
        CROSSFILTER(Sales["CustomerKey"], Customer["CustomerKey"], BOTH),
    ),
    table=Measures,
)
```

`USERELATIONSHIP` turns on the inactive relationship for that one calculation.
`CROSSFILTER` changes a relationship's direction (`BOTH`, `ONEWAY` or `NONE`), here so
that a filter on products counts the customers who bought them.

### Seeing the DAX

Every expression can print itself as DAX, and every measure as a definition:

```python
print(model["Sales by Ship Date"].definition())
# MEASURE Measures[Sales by Ship Date] =
#     CALCULATE([Sales Amount], USERELATIONSHIP(Sales[ShipDate], 'Date'[Date]))
```

## 10. Writing expressions: operators, VAR and parentheses

Expressions are Python calls that build DAX, so they read almost like DAX. The
differences come from Python's syntax.

### Operators

| DAX | Python |
|---|---|
| `+ - * / ^` | `+ - * / **` |
| `= <> < <= > >=` | `== != < <= > >=` |
| `&&` | `&` |
| `\|\|` | `\|` |
| `NOT(x)` | `~x` or `NOT(x)` |
| `x IN {"a", "b"}` | `x.IN(["a", "b"])` |
| `"a" & "b"` (join text) | `x.concat(y)` |

`and`, `or` and `not` don't work: Python can't hand them to the library. Use `&`, `|`
and `~`.

### Parentheses

In Python, `&` and `|` bind more tightly than `==` and `>`. **Put every comparison in
parentheses** when you combine them:

```python
red_bikes = (Product["Color"] == "Red") & (Product["Category"] == "Bikes")
assert red_bikes.to_dax() == 'Product[Color] = "Red" && Product[Category] = "Bikes"'
```

Without them, `Product["Color"] == "Red" & Product["Category"] == "Bikes"` means
`Product["Color"] == ("Red" & Product["Category"]) == "Bikes"`, a chained comparison
that Python can't build, so it raises a `TypeError` that says to add parentheses.

### Other operators in use

```python
from pythonic_dax import FILTER

model.add_column(
    Product, "Label", Product["Product"].concat(" (").concat(Product["Color"]).concat(")")
)
model.add_measure(
    "Clothing and Accessories",
    CALCULATE(sales_amount, Product["Category"].IN(["Clothing", "Accessories"])),
    table=Measures,
)
model.add_measure(
    "Big Lines",
    COUNTROWS(FILTER(Sales, (Sales["Quantity"] > 1) | ~(Sales["UnitPrice"] < 100))),
    table=Measures,
)
```

### VAR and RETURN

`VAR(Name=expression, ...)` defines variables, and `.RETURN(...)` gives the result. Pass
a `lambda v:` to `RETURN`, and read each variable as `v.Name`. As in DAX, each
variable is evaluated once, where it's defined, so a variable keeps its value inside a
later `CALCULATE`:

```python
from pythonic_dax import VAR

model.add_measure(
    "Growth",
    VAR(
        ThisYear=sales_amount,
        LastYear=CALCULATE(sales_amount, SAMEPERIODLASTYEAR(Date["Date"])),
    ).RETURN(lambda v: DIVIDE(v.ThisYear - v.LastYear, v.LastYear)),
    table=Measures,
    format_string="0.0%",
)
```

A later variable can use an earlier one by taking `v` too:

```python
model.add_column(
    Sales,
    "Discount",
    VAR(
        List=RELATED(Product["ListPrice"]),
        Off=lambda v: v.List - Sales["UnitPrice"],
    ).RETURN(lambda v: IF(v.List > 0, v.Off / v.List, 0)),
    format_string="0.0%",
)
```

`VAR` works the same in measures, calculated columns and table expressions.

## 11. Querying the model

### evaluate and SUMMARIZECOLUMNS

`model.evaluate(table_expression)` runs a query, like DAX `EVALUATE`, and returns a
Polars DataFrame. A report visual is a `SUMMARIZECOLUMNS` query: group-by columns, then
optional filter tables, then a dictionary of named measures:

```python
from pythonic_dax import SUMMARIZECOLUMNS, TREATAS

by_category = model.evaluate(
    SUMMARIZECOLUMNS(
        Product["Category"],
        Date["Year"],
        TREATAS(["West"], Customer["Region"]),  # a slicer on Region = West
        {"Sales": sales_amount, "Orders": model["Orders"]},
    )
)
assert by_category.columns == ["Product[Category]", "Date[Year]", "[Sales]", "[Orders]"]
```

* **Column names** follow Power BI: `Table[Column]` for group-by columns and `[Name]` for
  measures. The table name is never quoted there: a column of the `Date` table is
  `Date[Year]`, though DAX writes it `'Date'[Year]`.
* **Rows with only blank measures are left out**, as in Power BI.
* **Rows come in no particular order.** Sort the frame yourself (with the sort-by column,
  if there is one; see the [Streamlit guide](streamlit.md)).
* **Filters** are tables: `TREATAS([...], column)` for a list of values, or
  `FILTER(ALL(column), condition)` and `KEEPFILTERS(...)` as in DAX.
* Subtotals and a grand total come from `ROLLUPADDISSUBTOTAL`, as in a Power BI matrix.

### model.value for one number

`model.value(expression, *filters)` returns one plain Python value, as a card shows.
It runs `EVALUATE ROW("Value", CALCULATE(expression, filters...))`, so the filters are
`CALCULATE` filter arguments. BLANK comes back as `None`:

```python
from decimal import Decimal

total = model.value(sales_amount)
west = model.value(sales_amount, Customer["Region"] == "West")
by_name = model.value("Sales Amount", 'Customer[Region] = "West"')  # name and DAX text
assert west == by_name
assert model.value(sales_amount, Customer["Region"] == "North") is None  # BLANK
```

A Whole Number comes back as `int`, a Decimal Number as `float`, a Fixed Decimal Number
as `Decimal`, a Date/Time as `datetime`, Text as `str` and True/False as `bool`.

### The same query as DAX

`model.to_dax_query(...)` writes the DAX query that `evaluate` runs, with a
`DEFINE MEASURE` for every measure it uses, so you can paste it into Power BI Desktop's
DAX query view and compare:

```python
print(model.to_dax_query(SUMMARIZECOLUMNS(Product["Category"], {"Sales": sales_amount})))
```

## 12. DAX text

Expressions can also be written as DAX text. `model.parse_dax` turns text into the same
expression the Python functions build, so every check and engine rule applies:

```python
expression = model.parse_dax('CALCULATE([Sales Amount], Product[Color] = "Red")')
assert model.value(expression) == model.value(sales_amount, Product["Color"] == "Red")

model.add_measure(
    "Red Sales",
    model.parse_dax('CALCULATE([Sales Amount], Product[Color] = "Red")'),
    table=Measures,
)
model.add_column(
    Product, "Price Band", model.parse_dax('IF([ListPrice] >= 100, "High", "Low")', table=Product)
)
```

For a calculated column's formula, pass `table=` so `[ListPrice]` means that table's
column.

`model.evaluate_dax` runs a whole query, as in DAX query view. `DEFINE MEASURE` and
`DEFINE COLUMN` last only for the query:

```python
frame = model.evaluate_dax(
    """
    DEFINE
        MEASURE Measures[Units] = SUM(Sales[Quantity])
    EVALUATE
        SUMMARIZECOLUMNS(Product[Category], "Units", [Units], "Sales", [Sales Amount])
    """
)
assert frame.columns == ["Product[Category]", "[Units]", "[Sales]"]
```

## 13. Importing a Power BI model

A model built in Power BI Desktop can be imported, so you don't have to write it again.
Save the report as a Power BI project (`.pbip`) and point `from_tmdl` at its
`.SemanticModel` folder, or use `from_bim` for a `model.bim` file:

```python
shop = pdax.from_tmdl(
    "tests/fixtures/shop.SemanticModel",  # an example model in this repository
    data={
        "Product": pl.DataFrame(
            {
                "product_key": [1, 2],
                "color": ["Red", "Blue"],
                "launch_date": [dt.date(2023, 1, 1)] * 2,
            }
        ),
        "Sales": pl.DataFrame(
            {
                "product_key": [1, 2, 2],
                "order_date": [dt.date(2024, 1, 5)] * 3,
                "ship_date": [dt.date(2024, 1, 9)] * 3,
                "qty": [1, 2, 3],
                "price": [10.0, 5.5, 5.5],
            }
        ),
    },
    overrides={"Median Qty": "AVERAGE(Sales[Qty])"},
)
assert shop.value("Total Sales") == 37.5  # 1 x 10 + 2 x 5.5 + 3 x 5.5
```

* **What's imported:** tables, column types, relationships (active and inactive,
  cardinality and direction), measures, calculated columns and tables, hierarchies, the
  date table mark, and metadata: descriptions, format strings, display folders, hidden
  flags, sort-by columns, data categories, summarize-by, key columns and annotations.
  Measures and calculated columns are parsed from their DAX.
* **The data comes from you.** Power BI loads data with Power Query, which this library
  doesn't run. `data` maps each table name to a source (a DataFrame, `Parquet`,
  `Database` and so on), or is a function that takes a table name and returns one, such
  as `lambda name: Parquet(f"data/{name}.parquet")`. Source columns are matched by the
  model's `sourceColumn` names (as above, `product_key` becomes `ProductKey`), or else by
  the column names.
* **`overrides`** replaces a measure (by name), a calculated column (`"Table[Column]"`)
  or a calculated table with Python or other DAX text. Here it replaces a measure that
  uses `PERCENTILE.INC`, which the engine doesn't evaluate.
* **`skip_tables`** leaves tables out, with their relationships. **`culture`** picks a
  translation, such as `"es-ES"`, whose captions become display names.
* **`measures=False`** leaves out the measures, so you can write your own on the
  imported tables. Measures that a calculated column or calculated table uses are kept,
  and `model.import_notes` names them.
  A list of names, such as `measures=["Total Sales"]`, keeps those measures and the
  ones they use.
* **`calculated_columns=False`** leaves out the calculated columns, for rebuilding
  them yourself. A measure that reads one still imports and raises `NotSupportedError`
  when it's used, and relationships, sort-by columns and hierarchies that need one are
  left out, each named in `model.import_notes`.
* **Refused rather than guessed:** calculation groups, DirectQuery tables, many-to-many
  relationships and DAX user-defined functions. A measure that uses something
  unsupported still imports and raises `NotSupportedError` when it's used.
  `model.import_notes` lists what was left out.

```python
for note in shop.import_notes:
    print(note)
```

## 14. Describing the model

`model.describe()` returns the model's metadata as four Polars DataFrames, `tables`,
`columns`, `measures` and `relationships`. Use them for documentation pages, field
pickers and tooltips:

```python
info = model.describe()
measures = info["measures"]
assert measures.columns == [
    "measure",
    "table",
    "display_name",
    "description",
    "format_string",
    "display_folder",
    "is_hidden",
]
info["columns"].filter(~pl.col("is_hidden")).select("table", "column", "display_name")
```

The objects themselves list their parts too: `model.tables`, `model.measures`,
`model.relationships`, `Table.columns` and `Table.hierarchies`.

## 15. Errors

pythonic-dax never returns an answer that might differ from Power BI. Instead it raises:

* **`DaxError`** where Power BI would show an error, often as soon as you write the
  expression: `SUM` of something that isn't a column, columns from two tables in one
  `CALCULATE` filter, `USERELATIONSHIP` on columns with no relationship.
* **`ModelError`** for a model Power BI wouldn't accept: duplicate keys on a
  relationship's one side, a second active path between two tables, a bad date table.
* **`NotSupportedError`** where the engine can't yet be sure of Power BI's answer. The
  message says what isn't supported.

```python
from pythonic_dax import DaxError

try:
    SUM(Sales["Quantity"] * 2)
except DaxError as error:
    print(error)
```
