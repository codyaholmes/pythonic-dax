"""Pythonic DAX: Python functions that behave exactly like DAX in Power BI.

>>> import pythonic_dax as pdax
>>> from pythonic_dax import CALCULATE, SUM, ALL
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NoReturn

from . import sources
from ._coverage import PLANNED
from .errors import DaxError, ModelError, NotSupportedError
from .model import Model
from .objects import (
    Cardinality,
    Column,
    CrossFilter,
    Measure,
    Relationship,
    SummarizeBy,
    Table,
)
from .types import DaxType

__version__ = "0.0.1"

__all__ = [
    "PLANNED",
    "Cardinality",
    "Column",
    "CrossFilter",
    "DaxError",
    "DaxType",
    "Measure",
    "Model",
    "ModelError",
    "NotSupportedError",
    "Relationship",
    "SummarizeBy",
    "Table",
    "__version__",
    "sources",
]


def _planned_stub(name: str, phase: int) -> Callable[..., NoReturn]:
    def stub(*args: Any, **kwargs: Any) -> NoReturn:
        raise NotSupportedError(name, phase=phase)

    stub.__name__ = stub.__qualname__ = name
    stub.__doc__ = f"DAX {name}. Not implemented yet (planned for Phase {phase})."
    return stub


def __getattr__(name: str) -> Any:
    # DAX functions that are planned but not built yet resolve to stubs that raise
    # NotSupportedError, so the full API surface can be imported today.
    phase = PLANNED.get(name)
    if phase is None:
        raise AttributeError(f"module 'pythonic_dax' has no attribute {name!r}")
    return _planned_stub(name, phase)
