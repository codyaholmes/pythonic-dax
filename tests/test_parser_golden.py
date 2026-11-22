"""Run every golden case's DAX text through the parser.

Each case's ``dax`` is the query Power BI Desktop ran to produce the expected result,
so parsing and evaluating it must give the same answer, and the parsed expression must
print the same DAX back.
"""

from __future__ import annotations

import re

import pytest

import pythonic_dax as pdax
from golden.harness import (
    Case,
    add_case_tables,
    compare,
    discover_cases,
    new_case_model,
    normalize_dax,
)
from pythonic_dax import _parser

# An imported model already has the calculated columns its queries define.
DAX_CASES = [c for c in discover_cases() if c.dax and c.tmdl is None]


@pytest.mark.parametrize("case", DAX_CASES, ids=[c.id for c in DAX_CASES])
def test_golden_dax_text(case: Case) -> None:
    model = new_case_model(case)
    add_case_tables(case, model)
    error = case.expected.error
    try:
        result = _parser.evaluate_query(model, case.dax or "")
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
    if case.expected.scalar:  # a Model.value() case: its query is a one-cell ROW
        result = result.item()
    problems = compare(result, case.expected)
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize("case", DAX_CASES, ids=[c.id for c in DAX_CASES])
def test_golden_dax_round_trip(case: Case) -> None:
    model = new_case_model(case)
    add_case_tables(case, model)
    query = _parser.split_query(case.dax or "")
    if query.order_by or case.expected.error:
        pytest.skip("not a plain query")
    try:
        builder = _parser._QueryBuilder(model, query)
        node = builder.build()
    except pdax.NotSupportedError as exc:
        pytest.xfail(str(exc))
    assert normalize_dax(model.to_dax_query(node)) == normalize_dax(case.dax or "")
