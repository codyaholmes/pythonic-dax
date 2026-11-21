# Benchmarks

`bench.py` times typical report visuals (cards, charts, a product table with prior-year
measures, a year-to-date matrix, a 2,000-row customer table) on the demo's model with a
made-up Sales table of any size:

```
python benchmarks/bench.py --rows 1000000
python benchmarks/bench.py --rows 10000000 --repeat 2   # warm: caches kept between queries
python benchmarks/bench.py --rows 10000000 --slicer      # with Region = West selected
```

Results on a 4-core cloud machine (2026-11-21, Polars 2.1, Python 3.13):

| Sales rows | Before Phase 9 (slowest visual) | Phase 9, cold | Phase 9, warm |
|---|---|---|---|
| 100,000 | 20.6 s | 0.07 s | 0.07 s |
| 1,000,000 | (minutes) | 0.12 s | 0.12 s |
| 10,000,000 | (hours) | 0.83 s | 0.89 s |

"Cold" runs each visual once on a fresh model; "warm" is the best of two runs, as when
a report page reruns with the same model.
