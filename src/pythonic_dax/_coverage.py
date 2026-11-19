"""Registry of DAX functions and the plan phase that delivers each one.

Every name here can already be imported (``from pythonic_dax import CALCULATE``).
Until a function is implemented, calling it raises
:class:`~pythonic_dax.errors.NotSupportedError` naming its phase, so code written
against the API fails loudly instead of silently. Misspelled names still raise
``ImportError``.

Phase numbers follow the project plan:

* 3: evaluation core
* 4: CALCULATE and filter modifiers
* 5: iterators and table functions
* 6: scalar function library
* 7: time intelligence
"""

from __future__ import annotations

from typing import Final

# fmt: off
_BY_PHASE: Final[dict[int, tuple[str, ...]]] = {
    3: (
        "SUM", "AVERAGE", "MIN", "MAX", "COUNT", "COUNTA", "COUNTBLANK", "COUNTROWS",
        "DISTINCTCOUNT", "DISTINCTCOUNTNOBLANK", "BLANK", "TRUE", "FALSE",
        "IF", "AND", "OR", "NOT", "IN", "VAR", "RELATED", "RELATEDTABLE",
    ),
    4: (
        "CALCULATE", "CALCULATETABLE", "ALL", "ALLEXCEPT", "ALLNOBLANKROW",
        "ALLSELECTED", "REMOVEFILTERS", "KEEPFILTERS", "USERELATIONSHIP", "CROSSFILTER",
        "TREATAS", "VALUES", "DISTINCT", "HASONEVALUE", "HASONEFILTER", "SELECTEDVALUE",
        "ISFILTERED", "ISCROSSFILTERED", "ISINSCOPE", "FILTERS", "LOOKUPVALUE",
    ),
    5: (
        "SUMX", "AVERAGEX", "MINX", "MAXX", "COUNTX", "COUNTAX", "CONCATENATEX", "RANKX",
        "PRODUCTX", "EARLIER", "EARLIEST", "FILTER", "ADDCOLUMNS", "SELECTCOLUMNS",
        "SUMMARIZE", "SUMMARIZECOLUMNS", "GROUPBY", "CURRENTGROUP", "TOPN", "CROSSJOIN",
        "GENERATE", "GENERATEALL", "UNION", "INTERSECT", "EXCEPT", "NATURALINNERJOIN",
        "NATURALLEFTOUTERJOIN", "DATATABLE", "ROW", "GENERATESERIES", "OFFSET", "INDEX",
        "WINDOW", "RANK", "ROWNUMBER", "ORDERBY", "PARTITIONBY",
    ),
    6: (
        "DIVIDE", "SWITCH", "COALESCE", "IFERROR", "ISBLANK", "ISERROR", "ISNUMBER",
        "ISTEXT", "ISLOGICAL", "ISEMPTY", "ABS", "ROUND", "ROUNDUP", "ROUNDDOWN", "INT",
        "TRUNC", "MOD", "POWER", "SQRT", "EXP", "LN", "LOG", "LOG10", "SIGN", "CEILING",
        "FLOOR", "MROUND", "QUOTIENT", "CONCATENATE", "LEFT", "RIGHT", "MID", "LEN",
        "UPPER", "LOWER", "TRIM", "SUBSTITUTE", "REPLACE", "SEARCH", "FIND", "REPT",
        "EXACT", "CONTAINSSTRING", "CONTAINSSTRINGEXACT", "FORMAT", "VALUE", "FIXED",
        "CONVERT", "CURRENCY", "DATE", "TIME", "DATEVALUE", "TIMEVALUE", "YEAR", "MONTH",
        "DAY", "HOUR", "MINUTE", "SECOND", "WEEKDAY", "WEEKNUM", "QUARTER", "TODAY",
        "NOW", "EDATE", "EOMONTH", "DATEDIFF", "CONTAINS", "CONTAINSROW",
    ),
    7: (
        "CALENDAR", "CALENDARAUTO", "DATESYTD", "DATESQTD", "DATESMTD", "TOTALYTD",
        "TOTALQTD", "TOTALMTD", "SAMEPERIODLASTYEAR", "DATEADD", "PARALLELPERIOD",
        "DATESINPERIOD", "DATESBETWEEN", "PREVIOUSDAY", "PREVIOUSMONTH",
        "PREVIOUSQUARTER", "PREVIOUSYEAR", "NEXTDAY", "NEXTMONTH", "NEXTQUARTER",
        "NEXTYEAR", "FIRSTDATE", "LASTDATE", "STARTOFMONTH", "STARTOFQUARTER",
        "STARTOFYEAR", "ENDOFMONTH", "ENDOFQUARTER", "ENDOFYEAR", "OPENINGBALANCEMONTH",
        "OPENINGBALANCEQUARTER", "OPENINGBALANCEYEAR", "CLOSINGBALANCEMONTH",
        "CLOSINGBALANCEQUARTER", "CLOSINGBALANCEYEAR",
    ),
}
# fmt: on

PLANNED: Final[dict[str, int]] = {
    name: phase for phase, names in _BY_PHASE.items() for name in names
}
"""Map of DAX function name to the plan phase that implements it."""
