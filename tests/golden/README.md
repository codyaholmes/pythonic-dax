# Golden tests

Golden tests prove the engine gives the same answer as Power BI. Each case is one YAML
file under `cases/`, grouped in folders by topic.

## Case format

```yaml
id: summarizecolumns-basic          # unique, used as the pytest id
description: What behavior this pins down, in plain words.
source: documented                  # documented | powerbi
reference: https://...              # where the expected answer comes from
powerbi_version: null               # required when source is powerbi, e.g. "2.149.1203.0"
model:
  tables:
    Product:
      columns: {ProductKey: integer, Color: string}
      rows:
        - [1, Red]
  relationships:
    - from: Sales[ProductKey]        # the many side
      to: Product[ProductKey]       # the one side
      # cross_filter: both | single (default single); active: false; cardinality: one_to_one
python: |                           # must assign `result`
  total = model.add_measure("Total Sales", SUM(Sales["Amount"]))
  result = model.evaluate(SUMMARIZECOLUMNS(Product["Color"], {"Total": total}))
dax: |                              # what model.to_dax_query() prints for it (checked by a test)
  DEFINE MEASURE Sales[Total Sales] = SUM(Sales[Amount])
  EVALUATE SUMMARIZECOLUMNS('Product'[Color], "Total", [Total Sales])
expected:
  ordered: false                    # true only when the query has ORDER BY
  columns:                          # names exactly as Power BI's EVALUATE returns them
    "Product[Color]": string
    "[Total]": decimal
  rows:
    - [Red, 15.5]
    - [null, 100.0]                 # null means BLANK
```

To pin down an error instead of a result:

```yaml
expected:
  error: {type: DaxError, message_matches: "cannot convert"}
```

### Types

| Case type | DAX type | Polars dtype |
|---|---|---|
| `integer` | Whole Number | `Int64` |
| `decimal` | Decimal Number | `Float64` |
| `currency` | Fixed Decimal Number | `Decimal(19, 4)` |
| `datetime`, `date` | Date/Time | `Datetime("us")` |
| `string` | Text | `String` |
| `boolean` | True/False | `Boolean` |

## Rules

- **Code in cases:** the code sees `model`, `pdax`, every DAX function name, and each
  table by name (spaces become `_`).
- **Comparing results:** results must match column names, dtypes and values. Doubles
  are compared within 1e-12, and everything else must match exactly. BLANK only
  equals BLANK, never 0 or `""`.
- **Unbuilt features:** a case that hits an unbuilt feature (`NotSupportedError`) is
  reported as xfail. When the feature lands, the case starts passing without being
  edited.
- **Where answers come from:** `documented` cases take their answers from Microsoft
  Learn, dax.guide or SQLBI. `powerbi` cases were run in Power BI Desktop, and record
  the version. We prefer `powerbi`, so once a documented case has been run in Desktop,
  switch it over.
