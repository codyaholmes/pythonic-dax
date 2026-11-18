"""The semantic model: tables, relationships and measures.

This is the Phase 0 skeleton. The public shape is fixed here so tests and docs can
target it; the behavior arrives in Phase 1 (data model) and Phase 3 (evaluation).
"""

from __future__ import annotations

from typing import Any

from .errors import NotSupportedError


class Model:
    """A Power BI style semantic model."""

    def add_table(self, name: str, data: Any) -> Any:
        """Add a table from a Polars or pandas DataFrame."""
        raise NotSupportedError("Model.add_table", phase=1)

    def add_relationship(self, from_column: Any, to_column: Any, **options: Any) -> Any:
        """Add a relationship between two columns (many side first)."""
        raise NotSupportedError("Model.add_relationship", phase=1)

    def add_measure(self, name: str, expression: Any) -> Any:
        """Add a measure and return a reference to it."""
        raise NotSupportedError("Model.add_measure", phase=3)

    def evaluate(self, table_expression: Any) -> Any:
        """Evaluate a table expression, like DAX ``EVALUATE``, and return a DataFrame."""
        raise NotSupportedError("Model.evaluate", phase=3)
