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

Phase 0 (foundations). The public API is laid out, but the engine is not built yet.
Every DAX function in the plan can already be imported. Calling one that isn't built
raises `NotSupportedError`, naming the phase that delivers it. The engine never guesses
at an answer that might differ from Power BI.

## Development

```bash
pip install -e ".[dev]"
pytest            # unit tests and golden tests
ruff check . && ruff format --check .
mypy
```

Golden tests check the engine against Power BI. See [tests/golden/README.md](tests/golden/README.md).
