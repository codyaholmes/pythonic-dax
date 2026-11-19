# Pythonic DAX

Python functions that behave exactly like DAX in Power BI, used as the calculation
engine behind Streamlit apps.

> **Guiding ethos:** the Python should operate as closely as possible to how DAX works
> in Power BI today.

```python
import pythonic_dax as pdax
from pythonic_dax import ALL, CALCULATE, DIVIDE, SUM, SUMMARIZECOLUMNS

model = pdax.Model()
Sales = model.add_table("Sales", sales_df)
Product = model.add_table("Product", product_df)
model.add_relationship(Sales["ProductKey"], Product["ProductKey"])

total = model.add_measure("Total Sales", SUM(Sales["Amount"]))
share = model.add_measure("Share", DIVIDE(total, CALCULATE(total, ALL(Product))))

model.evaluate(SUMMARIZECOLUMNS(Product["Color"], {"Total": total, "Share": share}))
```

## Status

**Phase 1 (data model) is done.** You can build a model the way Power BI's semantic model works:

```python
from pythonic_dax.sources import Database, Parquet

model = pdax.Model(description="Exam revenue")
Exams = model.add_table(
    "Exams",
    Parquet("data/exams/*.parquet"),
    description="One row per exam taken.",
    columns={"VOID_FL": {"description": "A flag for voids.", "display_name": "Voided?"}},
)
Date = model.add_table("Date", Database("SELECT * FROM dim_date", connection))
Date.mark_as_date_table("Date")

model.add_relationship(Exams["ExamDate"], Date["Date"])
model.add_relationship(Exams["BilledDate"], Date["Date"], active=False)  # for USERELATIONSHIP

model.describe()["columns"]  # all metadata as a DataFrame
model.refresh()  # reload every source, like a dataset refresh
```

* **Data types:** they follow Power BI: Whole Number, Decimal, Fixed Decimal (currency),
  Date/Time, Text and True/False.
* **Names and text keys:** they are matched without regard to case, as in DAX.
* **Relationships:** they follow Power BI's rules. The one side can't have duplicate
  keys, related columns must share a type, and only one active filter path may run
  between two tables. Unmatched keys go to the hidden blank row.
* **Metadata:** every table, column and measure carries description, display name,
  format string, display folder, hidden flag, data category, default summarization,
  sort-by column and annotations. All of them are optional.

**Phase 2 (Python expression API) is done.** Every planned DAX function is a Python
function that builds an expression and checks its arguments the way Power BI does:

```python
from pythonic_dax import (
    CALCULATE,
    COUNTROWS,
    DIVIDE,
    SUM,
    SUMMARIZECOLUMNS,
    USERELATIONSHIP,
    VAR,
)

revenue = model.add_measure("Revenue", SUM(Exams["Fee"]), description="Billed revenue.")
billed = model.add_measure(
    "Revenue by Billed Date",
    CALCULATE(revenue, USERELATIONSHIP(Exams["BilledDate"], Date["Date"])),
)
avg = model.add_measure(
    "Avg Revenue per Exam",
    VAR(ValidExams=CALCULATE(COUNTROWS(Exams), Exams["VOID_FL"] == False)).RETURN(
        lambda v: DIVIDE(revenue, v.ValidExams)
    ),
    display_name="Average Revenue per Exam",
)
print(avg.expression.to_dax())
print(model.to_dax_query(SUMMARIZECOLUMNS(Date["Year"], {"Avg": avg})))  # paste into Desktop
```

* **Operators:** Python operators map to DAX: `+ - * / **`, `== != < <= > >=`, `&` for
  `&&`, `|` for `||`, `~` for `NOT`, `.IN([...])` for `IN`, `.concat()` for `&`.
* **Errors as you write:** wrong arguments raise the error Power BI would, when the
  formula is written. Examples: a non-column argument to `SUM`, columns from two tables
  in a `CALCULATE` filter, `USERELATIONSHIP` on columns with no relationship.

**Phase 3 (evaluation core) is done.** `model.evaluate()` runs a query and returns a
Polars DataFrame shaped like Power BI's `EVALUATE` result:

```python
total = model.add_measure("Total Sales", SUM(Sales["Amount"]), table="Sales")
model.add_column(Product, "Sales Rows", COUNTROWS(RELATEDTABLE(Sales)))  # calculated column
model.evaluate(SUMMARIZECOLUMNS(Product["Color"], {"Total": total}))
# columns "Product[Color]" and "[Total]"; colors with no sales are left out
```

* **Filters across relationships:** from the one side to the many side, and back when a
  relationship filters both ways. Unmatched keys land on the one side's hidden blank row.
* **Evaluated so far:** `SUM`, `AVERAGE`, `MIN`, `MAX`, `COUNT`, `COUNTA`, `COUNTBLANK`,
  `COUNTROWS`, `DISTINCTCOUNT`, `DISTINCTCOUNTNOBLANK`, `IF`, `AND`, `OR`, `NOT`, `BLANK`,
  `TRUE`, `FALSE`, `VAR`, `IN`, every operator, measure references (with context
  transition), `RELATED`, `RELATEDTABLE`, calculated columns, `ROW`, and
  `SUMMARIZECOLUMNS` with group-by columns and named expressions.
* **Checked against Power BI Desktop:** BLANK arithmetic, comparisons, currency types and
  blank-row behavior come from queries run in Desktop (see `tests/golden/cases`).

**Phase 4 (CALCULATE and filter modifiers) is done.**

```python
share = CALCULATE(total, ALLSELECTED(Product["Color"]))
by_ship = CALCULATE(total, USERELATIONSHIP(Sales["ShipDate"], Date["Date"]))
red = CALCULATE(total, Product["Color"] == "Red", KEEPFILTERS(Sales["Qty"] > 1))
```

* **CALCULATE and CALCULATETABLE:** filter arguments are evaluated first, then context
  transition, then the modifiers, then the filters, in Power BI's order. True/False
  filters work like `FILTER(ALL(<columns>), <condition>)`, blank row included. Tables
  filter their expanded table, so `CALCULATE(..., Sales)` filters Product too.
* **Modifiers:** `ALL`, `ALLEXCEPT`, `ALLNOBLANKROW`, `ALLSELECTED` (inside
  SUMMARIZECOLUMNS), `REMOVEFILTERS`, `KEEPFILTERS`, `USERELATIONSHIP`, `CROSSFILTER`
  (BOTH, ONEWAY, NONE) and `TREATAS`.
* **Inspecting the filter context:** `VALUES`, `DISTINCT`, `FILTERS`, `HASONEVALUE`,
  `HASONEFILTER`, `SELECTEDVALUE`, `ISFILTERED`, `ISCROSSFILTERED`, `ISINSCOPE` and
  `LOOKUPVALUE`. A one-column table used as a value gives its single value.
* **Checked against Power BI Desktop:** 15 golden cases in `tests/golden/cases/calculate`.

**Phase 5 (iterators, table functions and window functions) is done.**

```python
from pythonic_dax import ALLSELECTED, OFFSET, ORDERBY, RANKX, ROLLUPADDISSUBTOTAL, SUMX, TREATAS

margin = SUMX(Sales, Sales["Qty"] * Sales["Price"])
rank = RANKX(ALLSELECTED(Product["Color"]), total)
prev = CALCULATE(total, OFFSET(-1, ALLSELECTED(Date["Month"]), ORDERBY(Date["Month"])))

# A matrix: rows per category and color, subtotals per category, and a grand total.
model.evaluate(
    SUMMARIZECOLUMNS(
        ROLLUPADDISSUBTOTAL(
            Product["Category"], "IsCategoryTotal", Product["Color"], "IsColorTotal"
        ),
        TREATAS(["Red", "Blue"], Product["Color"]),
        {"Total": total, "Rank": rank},
    )
)
```

* **Iterators:** `SUMX`, `AVERAGEX`, `MINX`, `MAXX`, `COUNTX`, `COUNTAX`, `PRODUCTX`,
  `CONCATENATEX`, `RANKX`, `EARLIER` and `EARLIEST`. A measure or `CALCULATE` inside an
  iterator turns the row into a filter on the row's expanded table.
* **Table functions:** `FILTER`, `ADDCOLUMNS`, `SELECTCOLUMNS`, `SUMMARIZE` (group-by
  columns), `GROUPBY` with `CURRENTGROUP`, `TOPN`, `CROSSJOIN`, `GENERATE`,
  `GENERATEALL`, `UNION`, `INTERSECT`, `EXCEPT`, `NATURALINNERJOIN`,
  `NATURALLEFTOUTERJOIN`, `DATATABLE`, `GENERATESERIES`, table constructors and
  `ALLSELECTED` as a table. Columns keep their lineage, so a renamed column still
  filters its model column.
* **SUMMARIZECOLUMNS:** filter tables, and subtotals with `ROLLUPADDISSUBTOTAL`. As in
  Power BI, group-by values come from the filters on their own table only;
  relationships don't narrow them.
* **Window functions:** `OFFSET`, `INDEX`, `WINDOW`, `RANK` and `ROWNUMBER` with
  `ORDERBY`, `PARTITIONBY` and BLANKS `DEFAULT`, `FIRST` or `LAST`. When several rows
  are current (a total row), `OFFSET`, `INDEX` and `WINDOW` return the union of their
  results and `RANK` and `ROWNUMBER` return BLANK.
* **Refused rather than guessed:** an order Power BI would settle internally
  (`CONCATENATEX` or `TOPN` without an order, window functions whose `ORDERBY` doesn't
  identify each row), window functions without a relation, and `MATCHBY`.
* **Checked against Power BI Desktop:** 30 golden cases in `tests/golden/cases`
  (`iterators`, `tables`, `summarizecolumns` and `window`), including a matrix with
  subtotals and a grand total that matches cell for cell.

Anything else raises `NotSupportedError` naming the phase that brings it. The engine
never guesses at an answer that might differ from Power BI.

## Development

```bash
pip install -e ".[dev]"
pytest            # unit tests and golden tests
ruff check . && ruff format --check .
mypy
```

Golden tests check the engine against Power BI. See [tests/golden/README.md](tests/golden/README.md).
