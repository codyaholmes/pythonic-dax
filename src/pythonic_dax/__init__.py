"""Pythonic DAX: Python functions that behave exactly like DAX in Power BI.

>>> import pythonic_dax as pdax
>>> from pythonic_dax import CALCULATE, SUM, ALL, DIVIDE
>>> total = model.add_measure("Total Sales", SUM(Sales["Amount"]))
>>> share = model.add_measure("Share", DIVIDE(total, CALCULATE(total, ALL(Product))))
>>> share.expression.to_dax()
'DIVIDE([Total Sales], CALCULATE([Total Sales], ALL(Product)))'
"""

from __future__ import annotations

from . import sources
from ._coverage import PLANNED
from ._functions import *  # noqa: F403
from ._functions import __all__ as _function_names
from ._special import *  # noqa: F403
from ._special import __all__ as _special_names
from .errors import DaxError, ModelError, NotSupportedError
from .expr import VAR, Expr, Keyword, ModifierExpr, TableExpr
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

# DAX keywords used as arguments. Keywords that share a name with a DAX function
# (YEAR, MONTH, DAY, QUARTER, HOUR, MINUTE, SECOND, CURRENCY, ABS) are passed as the
# function itself, as in DAX: DATEADD(Date["Date"], -1, YEAR). Text such as "year"
# works everywhere too.
ASC = Keyword("ASC")
DESC = Keyword("DESC")
BOTH = Keyword("BOTH")
NONE = Keyword("NONE")
ONEWAY = Keyword("ONEWAY")
ONEWAY_LEFTFILTERSRIGHT = Keyword("ONEWAY_LEFTFILTERSRIGHT")
ONEWAY_RIGHTFILTERSLEFT = Keyword("ONEWAY_RIGHTFILTERSLEFT")
WEEK = Keyword("WEEK")
SKIP = Keyword("SKIP")
DENSE = Keyword("DENSE")
INTEGER = Keyword("INTEGER")
DOUBLE = Keyword("DOUBLE")
STRING = Keyword("STRING")
BOOLEAN = Keyword("BOOLEAN")
DATETIME = Keyword("DATETIME")
REL = Keyword("REL")
DEFAULT = Keyword("DEFAULT")
FIRST = Keyword("FIRST")
LAST = Keyword("LAST")

_KEYWORD_NAMES = [
    "ASC", "BOOLEAN", "BOTH", "DATETIME", "DEFAULT", "DENSE", "DESC", "DOUBLE", "FIRST",
    "INTEGER", "LAST", "NONE", "ONEWAY", "ONEWAY_LEFTFILTERSRIGHT",
    "ONEWAY_RIGHTFILTERSLEFT", "REL", "SKIP", "STRING", "WEEK",
]  # fmt: skip

__all__ = [
    "PLANNED",
    "VAR",
    "Cardinality",
    "Column",
    "CrossFilter",
    "DaxError",
    "DaxType",
    "Expr",
    "Keyword",
    "Measure",
    "Model",
    "ModelError",
    "ModifierExpr",
    "NotSupportedError",
    "Relationship",
    "SummarizeBy",
    "Table",
    "TableExpr",
    "__version__",
    "sources",
    *_function_names,
    *_special_names,
    *_KEYWORD_NAMES,
]
