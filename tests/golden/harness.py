"""Golden-test harness: load a case file, run it on the engine, compare to Power BI.

A case is a YAML file under ``tests/golden/cases``. See ``tests/golden/README.md`` for
the format. The harness is deliberately independent of engine internals: it only uses
the public API (``Model``, the DAX functions) so the same cases keep working as the
engine grows.
"""

from __future__ import annotations

import datetime as dt
import math
import re
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any

import polars as pl
import yaml

import pythonic_dax as pdax

CASES_DIR = Path(__file__).parent / "cases"

# DAX data types as written in case files, mapped to the Polars dtype that stores them.
DAX_TYPES: dict[str, pl.DataType] = {
    "integer": pl.Int64(),  # Whole Number
    "decimal": pl.Float64(),  # Decimal Number (a double in Power BI)
    "currency": pl.Decimal(precision=19, scale=4),  # Fixed Decimal Number
    "datetime": pl.Datetime("us"),  # Date/Time
    "date": pl.Datetime("us"),  # Date is Date/Time at midnight in DAX
    "string": pl.String(),  # Text
    "boolean": pl.Boolean(),  # True/False
}

_COLUMN_REF = re.compile(r"^'?(?P<table>[^'\[]+)'?\[(?P<column>[^\]]+)\]$")

# Doubles from Power BI are compared with a tight tolerance; everything else is exact.
REL_TOL = 1e-12
ABS_TOL = 1e-12


class CaseFormatError(ValueError):
    """A case file does not follow the golden-test format."""


@dataclass
class Expected:
    columns: dict[str, str] = field(default_factory=dict)
    rows: list[list[Any]] = field(default_factory=list)
    ordered: bool = False
    error: dict[str, str] | None = None


@dataclass
class Case:
    id: str
    path: Path
    description: str
    source: str
    python: str
    tables: dict[str, dict[str, Any]]
    relationships: list[dict[str, Any]]
    expected: Expected
    dax: str | None = None
    reference: str | None = None
    powerbi_version: str | None = None


def parse_column_ref(ref: str) -> tuple[str, str]:
    """Split ``Table[Column]`` or ``'My Table'[Column]`` into its parts."""
    match = _COLUMN_REF.match(ref.strip())
    if not match:
        raise CaseFormatError(f"not a column reference: {ref!r}")
    return match["table"], match["column"]


def load_case(path: Path) -> Case:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise CaseFormatError(f"{path.name}: top level must be a mapping")
    for key in ("id", "description", "source", "python", "expected"):
        if key not in raw:
            raise CaseFormatError(f"{path.name}: missing required key {key!r}")
    if raw["source"] not in ("documented", "powerbi"):
        raise CaseFormatError(f"{path.name}: source must be 'documented' or 'powerbi'")
    if raw["source"] == "powerbi" and not raw.get("powerbi_version"):
        raise CaseFormatError(f"{path.name}: powerbi cases must record powerbi_version")

    model = raw.get("model") or {}
    tables = model.get("tables") or {}
    for name, table in tables.items():
        for col, dax_type in table["columns"].items():
            if dax_type not in DAX_TYPES:
                raise CaseFormatError(f"{path.name}: {name}[{col}] has unknown type {dax_type!r}")
        width = len(table["columns"])
        for row in table.get("rows") or []:
            if len(row) != width:
                raise CaseFormatError(f"{path.name}: row {row!r} in {name} needs {width} values")

    relationships = model.get("relationships") or []
    for rel in relationships:
        parse_column_ref(rel["from"])
        parse_column_ref(rel["to"])

    exp = raw["expected"]
    expected = Expected(
        columns=exp.get("columns") or {},
        rows=exp.get("rows") or [],
        ordered=bool(exp.get("ordered", False)),
        error=exp.get("error"),
    )
    if expected.error is None:
        for col, dax_type in expected.columns.items():
            if dax_type not in DAX_TYPES:
                raise CaseFormatError(f"{path.name}: expected column {col} has type {dax_type!r}")
        for row in expected.rows:
            if len(row) != len(expected.columns):
                raise CaseFormatError(f"{path.name}: expected row {row!r} has the wrong width")

    return Case(
        id=raw["id"],
        path=path,
        description=raw["description"],
        source=raw["source"],
        python=raw["python"],
        tables=tables,
        relationships=relationships,
        expected=expected,
        dax=raw.get("dax"),
        reference=raw.get("reference"),
        powerbi_version=raw.get("powerbi_version"),
    )


def discover_cases(root: Path = CASES_DIR) -> list[Case]:
    return [load_case(p) for p in sorted(root.rglob("*.yaml"))]


def _to_python(value: Any, dax_type: str) -> Any:
    if value is None:
        return None  # BLANK
    if dax_type in ("datetime", "date") and isinstance(value, str):
        return dt.datetime.fromisoformat(value)
    if dax_type in ("datetime", "date") and isinstance(value, dt.date):
        return dt.datetime.combine(value, dt.time())
    if dax_type == "currency":
        return Decimal(str(value))
    return value


def build_frame(columns: dict[str, str], rows: list[list[Any]]) -> pl.DataFrame:
    """Build a Polars frame from case data, using the DAX type of each column."""
    names = list(columns)
    data = {
        name: [_to_python(row[i], columns[name]) for row in rows] for i, name in enumerate(names)
    }
    schema = {name: DAX_TYPES[columns[name]] for name in names}
    return pl.DataFrame(data, schema=schema)


def _identifier(table_name: str) -> str:
    """Name a table gets inside case code: ``'Sales Order'`` becomes ``Sales_Order``."""
    return re.sub(r"\W", "_", table_name)


@dataclass
class Run:
    """What happened when a case ran: the model and the queries passed to evaluate()."""

    model: pdax.Model
    queries: list[str] = field(default_factory=list)


def run_case(case: Case, run: Run | None = None) -> pl.DataFrame:
    """Build the model and run the case's Python code. Returns the ``result`` frame.

    Pass a :class:`Run` to record the DAX of every query the code evaluates.
    """
    model = pdax.Model()
    if run is not None:
        run.model = model
        real_evaluate = model.evaluate

        def recording_evaluate(table_expression: Any, *args: Any, **kwargs: Any) -> Any:
            order_by = kwargs.get("order_by", ())
            run.queries.append(model.to_dax_query(table_expression, order_by=order_by))
            return real_evaluate(table_expression, *args, **kwargs)

        model.evaluate = recording_evaluate  # type: ignore[method-assign]
    namespace: dict[str, Any] = {name: getattr(pdax, name) for name in pdax.__all__}
    namespace.update(pdax=pdax, model=model)

    for name, table in case.tables.items():
        frame = build_frame(table["columns"], table.get("rows") or [])
        namespace[_identifier(name)] = model.add_table(name, frame)

    for rel in case.relationships:
        f_table, f_col = parse_column_ref(rel["from"])
        t_table, t_col = parse_column_ref(rel["to"])
        options = {k: v for k, v in rel.items() if k not in ("from", "to")}
        model.add_relationship(
            namespace[_identifier(f_table)][f_col],
            namespace[_identifier(t_table)][t_col],
            **options,
        )

    exec(compile(case.python, str(case.path), "exec"), namespace)
    if "result" not in namespace:
        raise CaseFormatError(f"{case.path.name}: python code must assign `result`")
    result = namespace["result"]
    if not isinstance(result, pl.DataFrame):
        raise TypeError(f"result must be a polars DataFrame, got {type(result).__name__}")
    return result


def normalize_dax(text: str) -> str:
    """DAX text with whitespace collapsed, for comparing queries."""
    return " ".join(text.split())


def _values_equal(actual: Any, expected: Any) -> bool:
    if expected is None or actual is None:
        return actual is None and expected is None  # BLANK only equals BLANK here
    if isinstance(expected, float) or isinstance(actual, float):
        if isinstance(actual, bool) or isinstance(expected, bool):
            return False
        return math.isclose(float(actual), float(expected), rel_tol=REL_TOL, abs_tol=ABS_TOL)
    return bool(actual == expected)


def compare(actual: pl.DataFrame, expected: Expected) -> list[str]:
    """Return a list of differences between the result and the expected table."""
    problems: list[str] = []
    want_cols = list(expected.columns)
    if actual.columns != want_cols:
        return [f"columns differ: got {actual.columns}, want {want_cols}"]

    for name, dax_type in expected.columns.items():
        want_dtype = DAX_TYPES[dax_type]
        if actual.schema[name] != want_dtype:
            problems.append(f"column {name}: dtype {actual.schema[name]}, want {want_dtype}")
    if problems:
        return problems

    want = build_frame(expected.columns, expected.rows).rows()
    got = actual.rows()
    if len(got) != len(want):
        return [f"row count: got {len(got)}, want {len(want)}"]

    if not expected.ordered:
        # DAX gives no row order without ORDER BY, so compare as multisets.
        def sort_key(row: tuple[Any, ...]) -> tuple[Any, ...]:
            return tuple((v is None, str(v)) for v in row)

        got, want = sorted(got, key=sort_key), sorted(want, key=sort_key)

    for i, (g, w) in enumerate(zip(got, want, strict=True)):
        if not all(_values_equal(a, e) for a, e in zip(g, w, strict=True)):
            problems.append(f"row {i}: got {g}, want {w}")
    return problems
