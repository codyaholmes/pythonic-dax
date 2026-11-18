"""Run every golden case against the engine.

A case whose feature is not built yet raises ``NotSupportedError`` and is reported as
an expected failure (xfail), so CI stays green while the gap stays visible. Any other
mismatch with Power BI is a real failure.
"""

from __future__ import annotations

import re

import pytest

import pythonic_dax as pdax
from golden.harness import Case, compare, discover_cases, run_case

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
