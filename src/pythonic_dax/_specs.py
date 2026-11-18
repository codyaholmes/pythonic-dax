"""Signatures of the DAX functions: parameter kinds, return kind and a summary.

The signature mini-language mirrors the DAX reference:

* ``name:KIND`` is a required parameter, ``name:KIND?`` an optional one.
* ``name:KIND*`` repeats zero or more times, ``name:KIND+`` one or more (last only).
* Kinds: ``S`` scalar, ``T`` table, ``C`` column reference, ``TC`` table or column,
  ``F`` CALCULATE filter (boolean, table or modifier), ``D`` dates (column or
  table), ``K(A|B)`` keyword from the listed set.

Return kinds: ``S`` scalar, ``T`` table, ``M`` CALCULATE modifier.

Functions whose arguments come in name/value pairs (``ADDCOLUMNS``, ``SWITCH``,
``SUMMARIZECOLUMNS`` ...) are written by hand in :mod:`pythonic_dax._special`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .types import DaxType

TypeRule = DaxType | str | None
"""A fixed DAX type, ``"arg0"`` for the type of the first argument, or unknown."""


@dataclass(frozen=True)
class Param:
    name: str
    kind: str
    optional: bool = False
    repeat: str = ""  # "", "*" or "+"
    keywords: tuple[str, ...] = ()

    @property
    def python_name(self) -> str:
        return self.name + "_" if self.name in _PY_RESERVED else self.name


@dataclass(frozen=True)
class Spec:
    name: str
    params: tuple[Param, ...]
    returns: str
    summary: str
    type_rule: TypeRule = None
    numeric_column: bool = False  # DAX refuses Text and True/False columns here

    @property
    def signature(self) -> str:
        """The signature as the DAX reference writes it."""
        text = ""
        for i, p in enumerate(self.params):
            sep = ", " if i else ""
            item = f"<{p.name}>"
            if p.repeat == "+":
                text += f"{sep}{item}[, {item} ...]"
            elif p.repeat == "*":
                text += f"[{sep}{item} ...]"
            elif p.optional:
                text += f"[{sep}{item}]"
            else:
                text += f"{sep}{item}"
        return f"{self.name}({text})"


_PY_RESERVED = {"from", "else", "type", "in", "is", "not", "and", "or"}

_PARAM = re.compile(r"^(?P<name>\w+):(?P<kind>TC|[STCFDKA])(?:\((?P<kw>[\w|]+)\))?(?P<mod>[?*+]?)$")


def _parse(signature: str) -> tuple[Param, ...]:
    if not signature:
        return ()
    params = []
    for raw in signature.split(","):
        m = _PARAM.match(raw.strip())
        if not m:
            raise ValueError(f"bad parameter spec {raw!r}")
        mod = m["mod"]
        params.append(
            Param(
                name=m["name"],
                kind=m["kind"],
                optional=mod in ("?", "*"),
                repeat=mod if mod in ("*", "+") else "",
                keywords=tuple(m["kw"].split("|")) if m["kw"] else (),
            )
        )
    for p in params[:-1]:
        if p.repeat:
            raise ValueError(f"only the last parameter can repeat: {signature}")
    return tuple(params)


INT_, D_, B, S_, DT, C_ = (
    DaxType.INTEGER,
    DaxType.DECIMAL,
    DaxType.BOOLEAN,
    DaxType.STRING,
    DaxType.DATETIME,
    DaxType.CURRENCY,
)

INTERVALS = "DAY|MONTH|QUARTER|YEAR"
DATEDIFF_INTERVALS = "SECOND|MINUTE|HOUR|DAY|WEEK|MONTH|QUARTER|YEAR"
ORDER = "ASC|DESC"
BLANKS = "DEFAULT|FIRST|LAST"

# name: (signature, returns, type rule, summary[, numeric_column])
_TABLE: dict[str, tuple[str, str, TypeRule, str] | tuple[str, str, TypeRule, str, bool]] = {
    # --- aggregation and logic (Phase 3) ---
    "SUM": ("column:C", "S", "arg0", "Adds all the numbers in a column.", True),
    "AVERAGE": ("column:C", "S", D_, "The average of the numbers in a column.", True),
    "COUNT": ("column:C", "S", INT_, "Counts the non-blank values in a column."),
    "COUNTA": ("column:C", "S", INT_, "Counts the non-blank values in a column."),
    "COUNTBLANK": ("column:C", "S", INT_, "Counts the blank values in a column."),
    "COUNTROWS": ("table:T?", "S", INT_, "Counts the rows of a table."),
    "DISTINCTCOUNT": ("column:C", "S", INT_, "Counts the distinct values in a column."),
    "DISTINCTCOUNTNOBLANK": ("column:C", "S", INT_, "Counts distinct values, ignoring BLANK."),
    "BLANK": ("", "S", None, "Returns BLANK."),
    "TRUE": ("", "S", B, "Returns TRUE."),
    "FALSE": ("", "S", B, "Returns FALSE."),
    "IF": (
        "logical_test:S, value_if_true:S, value_if_false:S?",
        "S",
        None,
        "Returns one value if the test is TRUE and another if it is FALSE.",
    ),
    "AND": ("logical1:S, logical2:S", "S", B, "TRUE when both arguments are TRUE."),
    "OR": ("logical1:S, logical2:S", "S", B, "TRUE when either argument is TRUE."),
    "NOT": ("logical:S", "S", B, "Changes TRUE to FALSE and FALSE to TRUE."),
    # --- CALCULATE and filters (Phase 4) ---
    "CALCULATE": (
        "expression:S, filter:F*",
        "S",
        "arg0",
        "Evaluates an expression in a filter context changed by the filters.",
    ),
    "CALCULATETABLE": (
        "table:T, filter:F*",
        "T",
        None,
        "Evaluates a table expression in a filter context changed by the filters.",
    ),
    "ALL": (
        "table_or_column:TC*",
        "T",
        None,
        "Returns all rows or values, ignoring filters. With no arguments, removes all "
        "filters (only inside CALCULATE).",
    ),
    "ALLEXCEPT": (
        "table:T, column:C+",
        "T",
        None,
        "Removes all filters from a table except those on the listed columns.",
    ),
    "ALLNOBLANKROW": (
        "table_or_column:TC+",
        "T",
        None,
        "Like ALL, but without the blank row added for unmatched keys.",
    ),
    "ALLSELECTED": (
        "table_or_column:TC*",
        "T",
        None,
        "Removes filters from inside the query while keeping outside filters.",
    ),
    "REMOVEFILTERS": (
        "table_or_column:TC*",
        "M",
        None,
        "Removes filters from tables or columns (only inside CALCULATE).",
    ),
    "KEEPFILTERS": (
        "expression:F",
        "M",
        None,
        "Adds a filter without replacing existing filters on the same columns.",
    ),
    "USERELATIONSHIP": (
        "column1:C, column2:C",
        "M",
        None,
        "Uses an inactive relationship for this calculation (only inside CALCULATE).",
    ),
    "CROSSFILTER": (
        "column1:C, column2:C, direction:K(BOTH|NONE|ONEWAY|ONEWAY_LEFTFILTERSRIGHT|"
        "ONEWAY_RIGHTFILTERSLEFT)",
        "M",
        None,
        "Changes a relationship's filter direction for this calculation.",
    ),
    "TREATAS": (
        "table:T, column:C+",
        "T",
        None,
        "Applies the values of a table as filters on unrelated columns.",
    ),
    "VALUES": (
        "table_or_column:TC",
        "T",
        None,
        "The distinct values of a column (with the blank row if there is one).",
    ),
    "DISTINCT": (
        "table_or_column:TC",
        "T",
        None,
        "The distinct values or rows (without the blank row).",
    ),
    "HASONEVALUE": ("column:C", "S", B, "TRUE when the column is filtered to one value."),
    "HASONEFILTER": ("column:C", "S", B, "TRUE when the column has exactly one filter value."),
    "SELECTEDVALUE": (
        "column:C, alternate_result:S?",
        "S",
        "arg0",
        "The column's value when filtered to one value, otherwise the alternate result.",
    ),
    "ISFILTERED": ("table_or_column:TC", "S", B, "TRUE when filtered directly."),
    "ISCROSSFILTERED": ("table_or_column:TC", "S", B, "TRUE when filtered at all."),
    "ISINSCOPE": ("column:C", "S", B, "TRUE when the column is a grouping level."),
    "FILTERS": ("column:C", "T", None, "The values directly filtering a column."),
    "RELATED": ("column:C", "S", "arg0", "A value from the one side of a relationship."),
    "RELATEDTABLE": ("table:T", "T", None, "The rows of a table related to the current row."),
    # --- iterators and table functions (Phase 5) ---
    "SUMX": ("table:T, expression:S", "S", "arg1", "Sums an expression over a table."),
    "AVERAGEX": ("table:T, expression:S", "S", D_, "Averages an expression over a table."),
    "MINX": ("table:T, expression:S", "S", "arg1", "The smallest value of an expression."),
    "MAXX": ("table:T, expression:S", "S", "arg1", "The largest value of an expression."),
    "COUNTX": ("table:T, expression:S", "S", INT_, "Counts non-blank results."),
    "COUNTAX": ("table:T, expression:S", "S", INT_, "Counts non-blank results."),
    "PRODUCTX": ("table:T, expression:S", "S", D_, "Multiplies an expression over a table."),
    "RANKX": (
        "table:T, expression:S, value:S?, order:K(ASC|DESC)?, ties:K(SKIP|DENSE)?",
        "S",
        INT_,
        "The rank of a value among the expression's values over a table.",
    ),
    "EARLIER": ("column:C, number:S?", "S", "arg0", "A value from an outer row context."),
    "EARLIEST": ("column:C", "S", "arg0", "A value from the outermost row context."),
    "FILTER": ("table:T, filter:S", "T", None, "The rows of a table where the condition is TRUE."),
    "CURRENTGROUP": ("", "T", None, "The rows of the current group, inside GROUPBY."),
    "CROSSJOIN": ("table:T+", "T", None, "Every combination of rows of the tables."),
    "GENERATE": ("table1:T, table2:T", "T", None, "Joins each row with a table it produces."),
    "GENERATEALL": (
        "table1:T, table2:T",
        "T",
        None,
        "Like GENERATE, keeping rows that produce no rows.",
    ),
    "UNION": ("table:T+", "T", None, "Appends tables with the same columns."),
    "INTERSECT": ("table1:T, table2:T", "T", None, "Rows found in both tables."),
    "EXCEPT": ("table1:T, table2:T", "T", None, "Rows of the first table not in the second."),
    "NATURALINNERJOIN": ("left:T, right:T", "T", None, "Inner join on common columns."),
    "NATURALLEFTOUTERJOIN": ("left:T, right:T", "T", None, "Left join on common columns."),
    "GENERATESERIES": (
        "start:S, end:S, increment:S?",
        "T",
        None,
        "A one-column table of values from start to end.",
    ),
    "PARTITIONBY": ("column:C+", "T", None, "The partitions of a window function."),
    "OFFSET": (
        "delta:S, relation:T?, orderBy:T?, blanks:K(DEFAULT|FIRST|LAST)?, partitionBy:T?, "
        "matchBy:T?",
        "T",
        None,
        "The row a given number of rows away within the relation.",
    ),
    "INDEX": (
        "position:S, relation:T?, orderBy:T?, blanks:K(DEFAULT|FIRST|LAST)?, partitionBy:T?, "
        "matchBy:T?",
        "T",
        None,
        "The row at an absolute position within the relation.",
    ),
    "WINDOW": (
        "from:S, from_type:K(ABS|REL), to:S, to_type:K(ABS|REL), relation:T?, orderBy:T?, "
        "blanks:K(DEFAULT|FIRST|LAST)?, partitionBy:T?, matchBy:T?",
        "T",
        None,
        "The rows within a window of positions.",
    ),
    "RANK": (
        "ties:K(SKIP|DENSE)?, relation:T?, orderBy:T?, blanks:K(DEFAULT|FIRST|LAST)?, "
        "partitionBy:T?, matchBy:T?",
        "S",
        INT_,
        "The rank of the current row within the relation.",
    ),
    "ROWNUMBER": (
        "relation:T?, orderBy:T?, blanks:K(DEFAULT|FIRST|LAST)?, partitionBy:T?, matchBy:T?",
        "S",
        INT_,
        "The unique number of the current row within the relation.",
    ),
    # --- scalar library (Phase 6) ---
    "DIVIDE": (
        "numerator:S, denominator:S, alternate_result:S?",
        "S",
        D_,
        "Divides, returning the alternate result (BLANK by default) when dividing by 0.",
    ),
    "COALESCE": ("value:S+", "S", None, "The first argument that isn't BLANK."),
    "IFERROR": ("value:S, value_if_error:S", "S", None, "The value, or a fallback on error."),
    "ISBLANK": ("value:S", "S", B, "TRUE when the value is BLANK."),
    "ISERROR": ("value:S", "S", B, "TRUE when the value is an error."),
    "ISNUMBER": ("value:S", "S", B, "TRUE when the value is a number."),
    "ISTEXT": ("value:S", "S", B, "TRUE when the value is text."),
    "ISLOGICAL": ("value:S", "S", B, "TRUE when the value is TRUE or FALSE."),
    "ISEMPTY": ("table:T", "S", B, "TRUE when the table has no rows."),
    "ABS": ("number:S", "S", "arg0", "The absolute value."),
    "ROUND": ("number:S, num_digits:S", "S", D_, "Rounds to a number of digits."),
    "ROUNDUP": ("number:S, num_digits:S", "S", D_, "Rounds away from zero."),
    "ROUNDDOWN": ("number:S, num_digits:S", "S", D_, "Rounds toward zero."),
    "INT": ("number:S", "S", INT_, "Rounds down to an integer."),
    "TRUNC": ("number:S, num_digits:S?", "S", D_, "Removes the fractional part."),
    "MOD": ("number:S, divisor:S", "S", None, "The remainder after division."),
    "POWER": ("number:S, power:S", "S", D_, "A number raised to a power."),
    "SQRT": ("number:S", "S", D_, "The square root."),
    "EXP": ("number:S", "S", D_, "e raised to a power."),
    "LN": ("number:S", "S", D_, "The natural logarithm."),
    "LOG": ("number:S, base:S?", "S", D_, "The logarithm in a base (10 by default)."),
    "LOG10": ("number:S", "S", D_, "The base-10 logarithm."),
    "SIGN": ("number:S", "S", INT_, "-1, 0 or 1 depending on the sign."),
    "CEILING": ("number:S, significance:S", "S", None, "Rounds up to a multiple."),
    "FLOOR": ("number:S, significance:S", "S", None, "Rounds down to a multiple."),
    "MROUND": ("number:S, multiple:S", "S", None, "Rounds to the nearest multiple."),
    "QUOTIENT": ("numerator:S, denominator:S", "S", INT_, "The integer part of a division."),
    "CONCATENATE": ("text1:S, text2:S", "S", S_, "Joins two text values."),
    "LEFT": ("text:S, num_chars:S?", "S", S_, "Characters from the start of the text."),
    "RIGHT": ("text:S, num_chars:S?", "S", S_, "Characters from the end of the text."),
    "MID": ("text:S, start_num:S, num_chars:S", "S", S_, "Characters from the middle."),
    "LEN": ("text:S", "S", INT_, "The number of characters."),
    "UPPER": ("text:S", "S", S_, "Converts to upper case."),
    "LOWER": ("text:S", "S", S_, "Converts to lower case."),
    "TRIM": ("text:S", "S", S_, "Removes extra spaces."),
    "SUBSTITUTE": (
        "text:S, old_text:S, new_text:S, instance_num:S?",
        "S",
        S_,
        "Replaces text with new text.",
    ),
    "REPLACE": (
        "old_text:S, start_num:S, num_chars:S, new_text:S",
        "S",
        S_,
        "Replaces part of the text by position.",
    ),
    "SEARCH": (
        "find_text:S, within_text:S, start_num:S?, not_found_value:S?",
        "S",
        INT_,
        "The position of text, ignoring case.",
    ),
    "FIND": (
        "find_text:S, within_text:S, start_num:S?, not_found_value:S?",
        "S",
        INT_,
        "The position of text, matching case.",
    ),
    "REPT": ("text:S, num_times:S", "S", S_, "Repeats text."),
    "EXACT": ("text1:S, text2:S", "S", B, "TRUE when two texts match exactly, with case."),
    "CONTAINSSTRING": ("within_text:S, find_text:S", "S", B, "TRUE when text contains text."),
    "CONTAINSSTRINGEXACT": (
        "within_text:S, find_text:S",
        "S",
        B,
        "TRUE when text contains text, matching case.",
    ),
    "FORMAT": (
        "value:S, format_string:S, locale_name:S?",
        "S",
        S_,
        "Formats a value as text with a format string.",
    ),
    "VALUE": ("text:S", "S", D_, "Converts text to a number."),
    "FIXED": (
        "number:S, decimals:S?, no_commas:S?",
        "S",
        S_,
        "Formats a number as text with fixed decimals.",
    ),
    "CONVERT": (
        "expression:S, datatype:K(INTEGER|DOUBLE|STRING|BOOLEAN|CURRENCY|DATETIME)",
        "S",
        None,
        "Converts a value to a data type.",
    ),
    "CURRENCY": ("value:S", "S", C_, "Converts a value to Fixed Decimal (currency)."),
    "DATE": ("year:S, month:S, day:S", "S", DT, "A date from its parts."),
    "TIME": ("hour:S, minute:S, second:S", "S", DT, "A time from its parts."),
    "DATEVALUE": ("date_text:S", "S", DT, "Converts text to a date."),
    "TIMEVALUE": ("time_text:S", "S", DT, "Converts text to a time."),
    "YEAR": ("date:S", "S", INT_, "The year of a date."),
    "MONTH": ("date:S", "S", INT_, "The month of a date, 1 to 12."),
    "DAY": ("date:S", "S", INT_, "The day of the month."),
    "HOUR": ("datetime:S", "S", INT_, "The hour, 0 to 23."),
    "MINUTE": ("datetime:S", "S", INT_, "The minute, 0 to 59."),
    "SECOND": ("datetime:S", "S", INT_, "The second, 0 to 59."),
    "WEEKDAY": ("date:S, return_type:S?", "S", INT_, "The day of the week as a number."),
    "WEEKNUM": ("date:S, return_type:S?", "S", INT_, "The week number of the year."),
    "QUARTER": ("date:S", "S", INT_, "The quarter, 1 to 4."),
    "TODAY": ("", "S", DT, "Today's date."),
    "NOW": ("", "S", DT, "The current date and time."),
    "EDATE": ("start_date:S, months:S", "S", DT, "The date a number of months away."),
    "EOMONTH": ("start_date:S, months:S", "S", DT, "The last day of a month."),
    "DATEDIFF": (
        f"date1:S, date2:S, interval:K({DATEDIFF_INTERVALS})",
        "S",
        INT_,
        "The number of interval boundaries between two dates.",
    ),
    "CONTAINSROW": ("table:T, value:S+", "S", B, "TRUE when the table contains the row."),
    # --- time intelligence (Phase 7) ---
    "CALENDAR": ("start_date:S, end_date:S", "T", None, "A table of contiguous dates."),
    "CALENDARAUTO": ("fiscal_year_end_month:S?", "T", None, "A date table covering the model."),
    "DATESYTD": ("dates:D, year_end_date:S?", "T", None, "Dates from the start of the year."),
    "DATESQTD": ("dates:D", "T", None, "Dates from the start of the quarter."),
    "DATESMTD": ("dates:D", "T", None, "Dates from the start of the month."),
    "TOTALYTD": (
        "expression:S, dates:D, filter:F?, year_end_date:S?",
        "S",
        "arg0",
        "The year-to-date value of an expression.",
    ),
    "TOTALQTD": (
        "expression:S, dates:D, filter:F?",
        "S",
        "arg0",
        "The quarter-to-date value of an expression.",
    ),
    "TOTALMTD": (
        "expression:S, dates:D, filter:F?",
        "S",
        "arg0",
        "The month-to-date value of an expression.",
    ),
    "SAMEPERIODLASTYEAR": ("dates:D", "T", None, "The same dates one year earlier."),
    "DATEADD": (
        f"dates:D, number_of_intervals:S, interval:K({INTERVALS})",
        "T",
        None,
        "The dates shifted by a number of intervals.",
    ),
    "PARALLELPERIOD": (
        "dates:D, number_of_intervals:S, interval:K(MONTH|QUARTER|YEAR)",
        "T",
        None,
        "Whole periods shifted by a number of intervals.",
    ),
    "DATESINPERIOD": (
        f"dates:D, start_date:S, number_of_intervals:S, interval:K({INTERVALS})",
        "T",
        None,
        "The dates in a period starting at a date.",
    ),
    "DATESBETWEEN": ("dates:D, start_date:S, end_date:S", "T", None, "The dates in a range."),
    "PREVIOUSDAY": ("dates:D", "T", None, "The day before the first date."),
    "PREVIOUSMONTH": ("dates:D", "T", None, "The month before the first date."),
    "PREVIOUSQUARTER": ("dates:D", "T", None, "The quarter before the first date."),
    "PREVIOUSYEAR": ("dates:D, year_end_date:S?", "T", None, "The year before the first date."),
    "NEXTDAY": ("dates:D", "T", None, "The day after the last date."),
    "NEXTMONTH": ("dates:D", "T", None, "The month after the last date."),
    "NEXTQUARTER": ("dates:D", "T", None, "The quarter after the last date."),
    "NEXTYEAR": ("dates:D, year_end_date:S?", "T", None, "The year after the last date."),
    "FIRSTDATE": ("dates:D", "T", None, "The first date in the current context."),
    "LASTDATE": ("dates:D", "T", None, "The last date in the current context."),
    "STARTOFMONTH": ("dates:D", "T", None, "The first date of the month."),
    "STARTOFQUARTER": ("dates:D", "T", None, "The first date of the quarter."),
    "STARTOFYEAR": ("dates:D, year_end_date:S?", "T", None, "The first date of the year."),
    "ENDOFMONTH": ("dates:D", "T", None, "The last date of the month."),
    "ENDOFQUARTER": ("dates:D", "T", None, "The last date of the quarter."),
    "ENDOFYEAR": ("dates:D, year_end_date:S?", "T", None, "The last date of the year."),
    "OPENINGBALANCEMONTH": (
        "expression:S, dates:D, filter:F?",
        "S",
        "arg0",
        "The value at the end of the previous month.",
    ),
    "OPENINGBALANCEQUARTER": (
        "expression:S, dates:D, filter:F?",
        "S",
        "arg0",
        "The value at the end of the previous quarter.",
    ),
    "OPENINGBALANCEYEAR": (
        "expression:S, dates:D, filter:F?, year_end_date:S?",
        "S",
        "arg0",
        "The value at the end of the previous year.",
    ),
    "CLOSINGBALANCEMONTH": (
        "expression:S, dates:D, filter:F?",
        "S",
        "arg0",
        "The value at the last date of the month.",
    ),
    "CLOSINGBALANCEQUARTER": (
        "expression:S, dates:D, filter:F?",
        "S",
        "arg0",
        "The value at the last date of the quarter.",
    ),
    "CLOSINGBALANCEYEAR": (
        "expression:S, dates:D, filter:F?, year_end_date:S?",
        "S",
        "arg0",
        "The value at the last date of the year.",
    ),
}

SPECS: dict[str, Spec] = {
    name: Spec(
        name=name,
        params=_parse(entry[0]),
        returns=entry[1],
        type_rule=entry[2],
        summary=entry[3],
        numeric_column=len(entry) > 4 and bool(entry[4]),
    )
    for name, entry in _TABLE.items()
}

# Hand-written in _special.py, with their DAX signatures for documentation.
SPECIAL_SIGNATURES: dict[str, tuple[str, str]] = {
    "MIN": ("MIN(<column>) or MIN(<value1>, <value2>)", "The smallest value."),
    "MAX": ("MAX(<column>) or MAX(<value1>, <value2>)", "The largest value."),
    "SWITCH": (
        "SWITCH(<expression>, <value>, <result>[, <value>, <result>]...[, <else>])",
        "The result for the first value that matches the expression.",
    ),
    "ADDCOLUMNS": (
        "ADDCOLUMNS(<table>, <name>, <expression>[, <name>, <expression>]...)",
        "Adds calculated columns to a table.",
    ),
    "SELECTCOLUMNS": (
        "SELECTCOLUMNS(<table>, [<name>], <expression>[, [<name>], <expression>]...)",
        "Picks and calculates the columns of a new table.",
    ),
    "ROW": (
        "ROW(<name>, <expression>[, <name>, <expression>]...)",
        "A one-row table with the given columns.",
    ),
    "SUMMARIZE": (
        "SUMMARIZE(<table>, <groupBy_column>[, <groupBy_column>]...[, <name>, <expression>]...)",
        "Groups a table by columns.",
    ),
    "SUMMARIZECOLUMNS": (
        "SUMMARIZECOLUMNS(<groupBy_column>...[, <filter_table>]...[, <name>, <expression>]...)",
        "Groups by columns and adds measures, removing rows where every measure is BLANK.",
    ),
    "GROUPBY": (
        "GROUPBY(<table>, <groupBy_column>...[, <name>, <expression>]...)",
        "Groups a table, with CURRENTGROUP() in the expressions.",
    ),
    "TOPN": (
        "TOPN(<n_value>, <table>[, <orderBy_expression>[, <order>]]...)",
        "The top N rows of a table.",
    ),
    "CONCATENATEX": (
        "CONCATENATEX(<table>, <expression>[, <delimiter>[, <orderBy_expression>[, <order>]]...])",
        "Joins an expression's values over a table into text.",
    ),
    "LOOKUPVALUE": (
        "LOOKUPVALUE(<result_column>, <search_column>, <search_value>[, ...][, <alternate>])",
        "The value of a column in the row matching the search values.",
    ),
    "CONTAINS": (
        "CONTAINS(<table>, <column>, <value>[, <column>, <value>]...)",
        "TRUE when a row has all the column values.",
    ),
    "DATATABLE": (
        "DATATABLE(<name>, <type>[, <name>, <type>]..., {{<value>, ...}, ...})",
        "A table written out with its column types and rows.",
    ),
    "ORDERBY": (
        "ORDERBY(<expression>[, <order>[, <expression>[, <order>]]...])",
        "The sort order of a window function.",
    ),
    "IN": ("<value> IN <table>", "TRUE when the value is in the table. Same as x.IN(...)."),
    "VAR": ("VAR <name> = <expression> ... RETURN <expression>", "Defines variables."),
}
