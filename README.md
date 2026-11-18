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

Evaluating DAX functions comes in Phases 2 and 3. Every DAX function in the plan can
already be imported. Calling one that isn't built yet raises `NotSupportedError`,
naming the phase that delivers it. The engine never guesses at an answer that might
differ from Power BI.

## Development

```bash
pip install -e ".[dev]"
pytest            # unit tests and golden tests
ruff check . && ruff format --check .
mypy
```

Golden tests check the engine against Power BI. See [tests/golden/README.md](tests/golden/README.md).
