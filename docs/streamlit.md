# Using a model with Streamlit

pythonic-dax returns data and metadata: Polars frames, display names, descriptions and
format strings. Drawing them is your app's job. This guide shows patterns for doing that
in Streamlit, in your own app code, so you stay in control of every chart and table.

It assumes you have a model (see [Building a model](building-a-model.md)).

> The code blocks on this page make up a small, working app. The first block is
> `model.py`; the others, in order, are `app.py`. A test (`tests/test_docs.py`) runs
> it with Streamlit's test runner on every change. To try it, save the two files at the
> root of this repository and run `streamlit run app.py`.

## 1. Build the model in its own module

Keep the model in a module that doesn't import Streamlit, so it can be tested, used from
a notebook, or put behind another front end later:

```python
# model.py
from pathlib import Path

import polars as pl

import pythonic_dax as pdax
from pythonic_dax import DIVIDE, SUM, SUMX
from pythonic_dax.sources import CSV

DATA = Path("examples/sales_report/data")  # the demo data in this repository

model = pdax.Model(description="Made-up bike shop sales.")
Sales = model.add_table(
    "Sales",
    CSV(DATA / "sales.csv"),
    columns={"UnitPrice": {"data_type": "currency"}, "UnitCost": {"data_type": "currency"}},
)
Product = model.add_table("Product", CSV(DATA / "products.csv"))
Customer = model.add_table(
    "Customer",
    CSV(DATA / "customers.csv"),
    columns={"Region": {"description": "Sales region of the customer's address."}},
)

first, last = Sales["OrderDate"].values.min(), Sales["OrderDate"].values.max()
Date = model.add_table(
    "Date",
    pl.DataFrame({"Date": pl.datetime_range(first, last, "1d", eager=True)}).with_columns(
        Year=pl.col("Date").dt.year().cast(pl.Int64),
        Month=pl.col("Date").dt.strftime("%b"),
        MonthNumber=pl.col("Date").dt.month().cast(pl.Int64),
    ),
    columns={"MonthNumber": {"is_hidden": True}},
)
Date["Month"].sort_by_column = Date["MonthNumber"]
Date.mark_as_date_table("Date")

model.add_relationship(Sales["OrderDate"], Date["Date"])
model.add_relationship(Sales["ProductKey"], Product["ProductKey"])
model.add_relationship(Sales["CustomerKey"], Customer["CustomerKey"])

Measures = model.add_table("Measures", pl.DataFrame())
revenue = model.add_measure(
    "Sales Amount",
    SUMX(Sales, Sales["Quantity"] * Sales["UnitPrice"]),
    table=Measures,
    display_name="Sales",
    description="Quantity times unit price, after **discounts**.",
    format_string="#,0",
)
model.add_measure(
    "Units",
    SUM(Sales["Quantity"]),
    table=Measures,
    format_string="#,0",
    description="Items sold.",
)
cost = SUMX(Sales, Sales["Quantity"] * Sales["UnitCost"])
model.add_measure(
    "Margin %",
    DIVIDE(revenue - cost, revenue),
    table=Measures,
    format_string="0.0%",
    description="Sales minus cost, as a share of sales.",
)
```

## 2. Build it once, keep per-user state in session state

Streamlit runs `app.py` from the top on every click. Python runs an imported module
only the first time it's imported, so `from model import ...` builds the model once per
server process, and every rerun and every user shares it:

```python
# app.py
import altair as alt
import polars as pl
import streamlit as st

import pythonic_dax as pdax
from pythonic_dax import FORMAT, SUMMARIZECOLUMNS, TREATAS

from model import Customer, Date, Product, model
```

If the build depends on something chosen in the app (a data size, a database), wrap it
in a function decorated with `@st.cache_resource` instead, which also builds once per
set of arguments.

Because the model is shared, **treat it as read-only in app code**: don't add measures
or change metadata there. Keep what belongs to one user, such as their choices, in
`st.session_state`, which Streamlit keeps per browser session across reruns. Widgets
with a `key` store their value there:

```python
st.session_state.setdefault("visits", 0)
st.session_state["visits"] += 1

with st.sidebar:
    years = Date["Year"].values.unique().sort().to_list()
    st.selectbox(Date["Year"].display_name, years, index=len(years) - 1, key="year")
    regions = Customer["Region"].values.unique().sort().to_list()
    st.multiselect(
        Customer["Region"].display_name,
        regions,
        key="regions",
        help=Customer["Region"].description,
    )
```

Use the column's `display_name` for the label and its `description` for the help
tooltip, so the app shows what the model says.

## 3. Turn choices into filters

A slicer in Power BI is a filter on a column. In a query, that is a `TREATAS` filter
table. An empty selection means no filter, as in Power BI:

```python
def filters():
    chosen = [TREATAS([st.session_state["year"]], Date["Year"])]
    if st.session_state["regions"]:
        chosen.append(TREATAS(st.session_state["regions"], Customer["Region"]))
    return chosen
```

## 4. Query, then prepare the frame

A visual is one `SUMMARIZECOLUMNS` query. Its frame names columns as Power BI does:
`Date[Month]` for a group-by column and `[Sales Amount]` for a measure, named by the key
you give it. The table name is never quoted there, even where DAX quotes it (`column.ref`
is `'Date'[Month]`, because Date is a DAX word), so build the header from the names:

```python
def header(column):
    return f"{column.table.name}[{column.name}]"
```

A few steps turn the frame into one ready for a chart:

1. **Add the sort-by column.** Rows come in no particular order. Power BI adds a
   column's sort-by column to the query and sorts on it; do the same, then drop it.
2. **Cast Fixed Decimal to Float64.** Measures over currency columns come back as
   Polars `Decimal`, which charts don't plot. Cast only the copy you draw; the model's
   values stay exact.
3. **Rename to display names**, so the chart shows "Sales" for `[Sales Amount]`.

```python
def visual(column, measures):
    """One group-by column and some measures, sorted as Power BI sorts them, with
    display names as column names."""
    sort = column.sort_by_column
    group_by = [column, sort] if sort is not None else [column]
    frame = model.evaluate(SUMMARIZECOLUMNS(*group_by, *filters(), {m.name: m for m in measures}))
    if sort is not None:
        frame = frame.sort(header(sort)).drop(header(sort))
    else:
        frame = frame.sort(header(column))
    frame = frame.with_columns(pl.col(pl.Decimal).cast(pl.Float64))
    return frame.rename(
        {header(column): column.display_name, **{f"[{m.name}]": m.display_name for m in measures}}
    )


by_month = visual(Date["Month"], [model["Sales Amount"], model["Units"]])
```

## 5. Charts: keep the order

Altair sorts a text axis alphabetically unless told otherwise, which turns Jan, Feb,
Mar into Apr, Aug, Dec. The frame is already in the model's order, so pass `sort=None`
to keep it:

```python
sales = model["Sales Amount"]
month, sales_label = Date["Month"].display_name, sales.display_name
chart = (
    alt.Chart(by_month.to_pandas())
    .mark_bar()
    .encode(
        x=alt.X(month, sort=None, title=month),
        y=alt.Y(sales_label, title=sales_label),
        tooltip=[month, sales_label],
    )
)
st.subheader(f"{sales_label} by month")
st.altair_chart(chart)
```

## 6. Formatting numbers like Power BI

Power BI formats a value with its format string. Streamlit takes a different syntax.
`pdax.streamlit_format` converts the format string, and `pdax.streamlit_values`
prepares the values to go with it (it scales percentages, rounds as Power BI rounds, and
turns `Decimal` into floats). Always use the two together, with the same format string:

```python
def metric(measure):
    fmt = measure.format_string
    value = model.value(measure, *filters())
    st.metric(
        measure.display_name,
        pdax.streamlit_values(value, fmt) if value is not None else None,
        format=pdax.streamlit_format(fmt),
        help=measure.description,
    )


cards = st.columns(3)
for card, name in zip(cards, ["Sales Amount", "Units", "Margin %"], strict=True):
    with card:
        metric(model[name])
```

`model.value(measure, *filters)` gives one value, as a card shows. BLANK comes back as
`None`, which `st.metric` shows as a dash.

The same pair formats table columns, through `st.column_config.NumberColumn`:

```python
by_product = visual(Product["Category"], [sales, model["Margin %"]])
margin = model["Margin %"]
by_product = by_product.with_columns(
    pdax.streamlit_values(by_product[margin.display_name], margin.format_string)
)
st.dataframe(
    by_product,
    hide_index=True,
    column_config={
        sales_label: st.column_config.NumberColumn(
            format=pdax.streamlit_format(sales.format_string), help=sales.description
        ),
        margin.display_name: st.column_config.NumberColumn(
            format=pdax.streamlit_format(margin.format_string), help=margin.description
        ),
    },
)
```

Streamlit can't show every Power BI format. `streamlit_format` raises
`NotSupportedError` rather than show something Power BI wouldn't:

* Text before the number, such as `"$#,0"`, shows negatives as `$-5` in Streamlit but
  `-$5` in Power BI. If the values are never negative, pass `nonnegative=True` to both
  functions.
* For anything else, such as `"+0.0%;-0.0%;0.0%"`, ask the model for the text itself
  with `FORMAT`, and show it as text. This is exactly what Power BI shows:

```python
margin_text = model.value(FORMAT(margin, "+0.0%;-0.0%;0.0%"), *filters())
st.caption(f"Margin: {margin_text}")
```

## 7. Field descriptions

`model.describe()` returns the model's metadata as frames. Descriptions can hold
Markdown, and `st.table` renders Markdown in its cells, so a glossary page is a few lines:

```python
glossary = (
    model.describe()["measures"]
    .filter(~pl.col("is_hidden"))
    .select(Measure="display_name", Description="description")
    .sort("Measure")
)
with st.expander("What the numbers mean"):
    st.table(glossary)
```

## Summary

| Do | Why |
|---|---|
| Build the model in a module with no Streamlit import, and import it in the app. | It's built once and shared, and works with any front end. |
| Keep per-user choices in `st.session_state` (or widget keys). | Reruns and other users don't change the model. |
| Turn selections into `TREATAS` filters, and leave the filter out when nothing is selected. | That is how a Power BI slicer filters. |
| Add the sort-by column to the query, sort on it, then drop it. | Rows come back in no particular order. |
| Use `sort=None` in Altair. | Altair sorts text axes alphabetically. |
| Cast `pl.Decimal` to `pl.Float64` for charts. | Charts don't plot decimals. |
| Rename `Table[Column]` and `[Measure]` columns to display names. | Reports show display names. |
| Use `streamlit_format` and `streamlit_values` together, or `FORMAT` for text. | Streamlit then shows what Power BI shows. |
| Use `description` for `help=` tooltips and glossary tables. | The model's documentation reaches the user. |
