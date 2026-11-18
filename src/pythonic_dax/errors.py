"""Error types.

The guiding rule of this library is to behave exactly like DAX in Power BI. When the
engine meets something it cannot yet do the Power BI way, it raises
:class:`NotSupportedError` instead of guessing.
"""

from __future__ import annotations


class DaxError(Exception):
    """An error Power BI itself would raise when evaluating the expression."""


class ModelError(ValueError):
    """The model is invalid, in a way Power BI would also refuse.

    For example, a duplicate key on the one side of a relationship, or a second
    active filter path between two tables.
    """


class NotSupportedError(NotImplementedError):
    """A DAX feature this library does not reproduce yet.

    Raised instead of returning an answer that might differ from Power BI.
    """

    def __init__(self, feature: str, phase: int | None = None) -> None:
        self.feature = feature
        self.phase = phase
        where = " (planned for a later phase)" if phase is None else f" (planned for Phase {phase})"
        super().__init__(f"{feature} is not supported yet{where}.")
