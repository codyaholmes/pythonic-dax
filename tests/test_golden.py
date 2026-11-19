"""Run every golden case against the engine.

A case whose feature is not built yet raises ``NotSupportedError`` and is reported as
an expected failure (xfail), so CI stays green while the gap stays visible. Any other
mismatch with Power BI is a real failure.
"""

from __future__ import annotations

import contextlib
import re

import pytest

import pythonic_dax as pdax
from golden.harness import Case, Run, compare, discover_cases, normalize_dax, run_case

CASES = discover_cases()


@pytest.mark.parametrize("case", CASES, ids=[c.id for c in CASES])
def test_golden(case: Case) -> None:
    error = case.expected.error
    try:
        result = run_case(case)
    except pdax.NotSupportedError as exc:
        pytest.xfail(str(exc))
    except Exception as exc:
        if error is None:
            raise
        assert type(exc).__name__ == error["type"], f"raised {type(exc).__name__}: {exc}"
        if "message_matches" in error:
            assert re.search(error["message_matches"], str(exc)), str(exc)
        return

    assert error is None, f"expected {error['type']} but the engine returned a result"
    problems = compare(result, case.expected)
    assert not problems, "\n".join(problems)


DAX_CASES = [c for c in CASES if c.dax]


@pytest.mark.parametrize("case", DAX_CASES, ids=[c.id for c in DAX_CASES])
def test_golden_dax_matches_python(case: Case) -> None:
    """The DAX the Python query prints is the DAX the case runs in Power BI Desktop."""
    run = Run(model=pdax.Model())
    expected_error = Exception if case.expected.error else pdax.NotSupportedError
    with contextlib.suppress(expected_error):
        run_case(case, run)
    assert run.queries, "the case's Python code never called model.evaluate()"
    assert normalize_dax(run.queries[-1]) == normalize_dax(case.dax or "")
