from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import polars as pl
import pytest

import pythonic_dax as pdax


def test_every_planned_function_is_exported() -> None:
    missing = [name for name in pdax.PLANNED if name not in pdax.__all__]
    assert missing == []


def test_misspelled_function_is_an_import_error() -> None:
    with pytest.raises(ImportError):
        from pythonic_dax import CALCULAT  # type: ignore[attr-defined]  # noqa: F401


def test_not_supported_is_a_not_implemented_error() -> None:
    assert issubclass(pdax.NotSupportedError, NotImplementedError)


def test_unbuilt_function_names_its_phase() -> None:
    model = pdax.Model()
    model.add_table("T", pl.DataFrame({"x": [1]}))
    with pytest.raises(pdax.NotSupportedError, match="Phase 6"):
        model.evaluate(pdax.ROW("v", pdax.ABS(-1)))


def test_planned_names_are_upper_case_dax_names() -> None:
    assert all(name.isupper() for name in pdax.PLANNED)
    assert len(pdax.PLANNED) > 150


def test_functions_have_docstrings_with_dax_signature() -> None:
    assert pdax.DIVIDE.__doc__ is not None
    assert "DIVIDE(<numerator>, <denominator>[, <alternate_result>])" in pdax.DIVIDE.__doc__


def test_generated_functions_are_up_to_date() -> None:
    script = Path(__file__).parent.parent / "scripts" / "generate_functions.py"
    result = subprocess.run(
        [sys.executable, str(script), "--check"], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
