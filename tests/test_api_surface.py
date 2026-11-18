from __future__ import annotations

import pytest

import pythonic_dax as pdax


def test_planned_functions_import_and_raise_not_supported() -> None:
    from pythonic_dax import CALCULATE

    with pytest.raises(pdax.NotSupportedError, match=r"CALCULATE .*Phase 4"):
        CALCULATE()


def test_misspelled_function_is_an_import_error() -> None:
    with pytest.raises(ImportError):
        from pythonic_dax import CALCULAT  # type: ignore[attr-defined]  # noqa: F401


def test_not_supported_is_a_not_implemented_error() -> None:
    assert issubclass(pdax.NotSupportedError, NotImplementedError)


def test_model_methods_name_their_phase() -> None:
    with pytest.raises(pdax.NotSupportedError, match="Phase 1"):
        pdax.Model().add_table("Sales", None)


def test_planned_names_are_upper_case_dax_names() -> None:
    assert all(name.isupper() for name in pdax.PLANNED)
    assert len(pdax.PLANNED) > 150
