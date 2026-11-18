"""Tests for the golden-test harness itself, so a broken harness can't hide bugs."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from golden.harness import (
    CaseFormatError,
    Expected,
    build_frame,
    compare,
    discover_cases,
    load_case,
    parse_column_ref,
)


def test_all_cases_load() -> None:
    cases = discover_cases()
    assert cases, "no golden cases found"
    ids = [c.id for c in cases]
    assert len(ids) == len(set(ids)), "case ids must be unique"


@pytest.mark.parametrize(
    ("ref", "parts"),
    [
        ("Sales[Amount]", ("Sales", "Amount")),
        ("'Sales Order'[Order Date]", ("Sales Order", "Order Date")),
    ],
)
def test_parse_column_ref(ref: str, parts: tuple[str, str]) -> None:
    assert parse_column_ref(ref) == parts


def test_parse_column_ref_rejects_garbage() -> None:
    with pytest.raises(CaseFormatError):
        parse_column_ref("Sales.Amount")


def test_build_frame_uses_dax_types() -> None:
    frame = build_frame(
        {"k": "integer", "amt": "currency", "when": "date", "ok": "boolean"},
        [[1, 1.5, "2026-01-31", True], [None, None, None, None]],
    )
    assert frame.schema["k"] == pl.Int64
    assert frame.schema["amt"] == pl.Decimal(precision=19, scale=4)
    assert frame.schema["when"] == pl.Datetime("us")
    assert frame.row(1) == (None, None, None, None)


def _expected(rows: list[list[object]], ordered: bool = False) -> Expected:
    return Expected(columns={"T[c]": "string", "[v]": "decimal"}, rows=rows, ordered=ordered)


def _actual(rows: list[tuple[object, ...]]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema={"T[c]": pl.String, "[v]": pl.Float64}, orient="row")


def test_compare_matches_ignoring_order() -> None:
    actual = _actual([("b", 2.0), ("a", 1.0), (None, 3.0)])
    assert compare(actual, _expected([["a", 1.0], [None, 3.0], ["b", 2.0]])) == []


def test_compare_respects_order_when_required() -> None:
    actual = _actual([("b", 2.0), ("a", 1.0)])
    assert compare(actual, _expected([["a", 1.0], ["b", 2.0]], ordered=True))


def test_compare_blank_is_not_zero() -> None:
    actual = _actual([("a", 0.0)])
    assert compare(actual, _expected([["a", None]]))


def test_compare_float_tolerance_is_tight() -> None:
    assert compare(_actual([("a", 0.1 + 0.2)]), _expected([["a", 0.3]])) == []
    assert compare(_actual([("a", 0.3001)]), _expected([["a", 0.3]]))


def test_compare_checks_dtype() -> None:
    actual = pl.DataFrame({"T[c]": ["a"], "[v]": [1]})  # Int64 instead of Float64
    assert any("dtype" in p for p in compare(actual, _expected([["a", 1.0]])))


def test_compare_checks_column_names() -> None:
    actual = pl.DataFrame({"c": ["a"], "v": [1.0]})
    assert any("columns differ" in p for p in compare(actual, _expected([["a", 1.0]])))


def test_powerbi_case_needs_version(tmp_path: Path) -> None:
    case = tmp_path / "bad.yaml"
    case.write_text(
        "id: x\ndescription: x\nsource: powerbi\npython: result = None\n"
        "expected: {columns: {}, rows: []}\n"
    )
    with pytest.raises(CaseFormatError, match="powerbi_version"):
        load_case(case)
