# Pythonic DAX: working rules

**Guiding ethos:** the Python should operate as closely as possible to how DAX works in
Power BI today. When a convenient Python idiom and Power BI's behavior disagree, Power BI
wins.

- The public API is Python functions named and behaving like DAX (`CALCULATE`, `SUMX`,
  `FILTER`, ...). They build lazy expressions, and the model evaluates them in filter
  context. A DAX text parser is an optional later extra, not the main interface.
- Never return an answer that might differ from Power BI. Raise
  `pythonic_dax.NotSupportedError` instead.
- Every behavior change needs a golden case in `tests/golden/cases` (see its README).
  Prefer expected results captured from Power BI Desktop (`source: powerbi`).
- DAX function signatures live in `src/pythonic_dax/_specs.py`. After changing them,
  run `python scripts/generate_functions.py` to regenerate `_functions.py` (a test
  checks it's current). Functions with name/value pairs are hand-written in
  `_special.py`.
- `_coverage.py` maps each function to the phase that **evaluates** it. When a
  function's evaluation is built, update its entry.
- Polars is the storage backend. Pandas is accepted as input only.
- Checks before pushing: `ruff check . && ruff format --check . && mypy && pytest`.
- The plan, decisions, progress log and semantics notes live in Cody's Obsidian vault,
  not in this repo.
