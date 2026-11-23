"""Time intelligence (Phase 7): date tables, the date-set functions and the functions
built on them (TOTALYTD, OPENINGBALANCEMONTH, ...).

The rules here were checked against Power BI Desktop
(``tests/golden/cases/time-intelligence``). In short:

* A date-set function reads the dates visible in the current context (with context
  transition, as ``CALCULATETABLE(DISTINCT(<dates>))``) and returns dates drawn from
  all the values of the column, whatever filters are on it.
* Results keep the lineage of the dates column, so they filter it when used in
  CALCULATE. A filter on a date column that is the one side of a relationship (or the
  date column of a marked date table) also removes the filters from the rest of its
  table; that part lives in ``Evaluator._filter_argument``.

Where Power BI's answer wasn't checked, the functions raise NotSupportedError.
"""

from __future__ import annotations

import calendar
import datetime as dt
import itertools
import re
from collections.abc import Callable, Sequence
from typing import Any

from ._evaluate import (
    Context,
    Evaluator,
    Frame,
    ResultColumn,
    TableValue,
    Value,
)
from .errors import DaxError, NotSupportedError
from .expr import Literal_, TableCall, _Omitted, is_column
from .objects import Column
from .types import DaxType

Date = dt.date
DateSet = list[Date]  # sorted, distinct dates

_DATE_SET_FUNCTIONS = {
    "DATESYTD", "DATESQTD", "DATESMTD", "SAMEPERIODLASTYEAR", "DATEADD",
    "PARALLELPERIOD", "DATESINPERIOD", "DATESBETWEEN", "PREVIOUSDAY", "PREVIOUSMONTH",
    "PREVIOUSQUARTER", "PREVIOUSYEAR", "NEXTDAY", "NEXTMONTH", "NEXTQUARTER", "NEXTYEAR",
    "FIRSTDATE", "LASTDATE", "STARTOFMONTH", "STARTOFQUARTER", "STARTOFYEAR",
    "ENDOFMONTH", "ENDOFQUARTER", "ENDOFYEAR",
}  # fmt: skip

_MONTHS = {"MONTH": 1, "QUARTER": 3, "YEAR": 12}


# --- calendar arithmetic ----------------------------------------------------------------


def _last_day(year: int, month: int) -> int:
    return calendar.monthrange(year, month)[1]


def _is_month_end(d: Date) -> bool:
    return d.day == _last_day(d.year, d.month)


def _add_months(d: Date, months: int, *, month_end: bool = False) -> Date:
    """EDATE: the same day ``months`` later, clamped to the end of a shorter month.
    With ``month_end``, the last day of the target month (EOMONTH)."""
    index = d.year * 12 + d.month - 1 + months
    year, month = divmod(index, 12)
    month += 1
    if not 1900 <= year <= 9999:
        raise NotSupportedError("time intelligence outside the supported range of dates")
    last = _last_day(year, month)
    return Date(year, month, last if month_end else min(d.day, last))


def _add_days(d: Date, days: int) -> Date:
    try:
        return d + dt.timedelta(days=days)
    except OverflowError:
        raise NotSupportedError("time intelligence outside the supported range of dates") from None


YearEnd = tuple[int, int]  # (month, day)


def _period(d: Date, months: int, year_end: YearEnd = (12, 31)) -> tuple[Date, Date]:
    """The month (1), quarter (3) or year (12) containing ``d``, as (first, last) dates.
    Years end on ``year_end``."""
    if months == 12:
        end_month = year_end[0]
        end_year = d.year if (d.month, d.day) <= year_end else d.year + 1
        last = Date(end_year, end_month, _last_day(end_year, end_month))
        first = _add_days(_add_months(last, -12, month_end=True), 1)
        return first, last
    first_month = (d.month - 1) // months * months + 1
    first = Date(d.year, first_month, 1)
    return first, _add_months(first, months - 1, month_end=True)


def _shift_period(period: tuple[Date, Date], months: int) -> tuple[Date, Date]:
    first, last = period
    return _add_months(first, months), _add_months(last, months, month_end=True)


# --- the dates argument -----------------------------------------------------------------


class _Dates:
    """The dates argument of a time intelligence function: the column, all its dates,
    and the dates visible in the current context."""

    def __init__(self, column: Column, all_dates: DateSet, visible: DateSet) -> None:
        self.column = column
        self.all = all_dates
        self.visible = visible

    def within(self, first: Date, last: Date) -> DateSet:
        """All the column's dates from ``first`` to ``last``."""
        return [d for d in self.all if first <= d <= last]

    def result(self, dates: Sequence[Date]) -> Frame:
        values: list[list[Value]] = [[dt.datetime(d.year, d.month, d.day)] for d in dates]
        column = ResultColumn(
            f"{self.column.table.name}[{self.column.name}]", DaxType.DATETIME, self.column
        )
        return Frame([column], values)


def _to_dates(values: Sequence[Value], what: str) -> DateSet:
    dates = set()
    for v in values:
        if v is None:
            raise NotSupportedError(f"time intelligence over {what} that include BLANK")
        if not isinstance(v, dt.datetime):
            raise NotSupportedError(f"time intelligence over {what} that aren't dates")
        if v.time() != dt.time():
            raise NotSupportedError(f"time intelligence over {what} with times of day")
        dates.add(v.date())
    return sorted(dates)


def _dates(ev: Evaluator, node: Any, ctx: Context, fn: str) -> _Dates:
    if is_column(node):
        column: Column = node
        if column.data_type is not DaxType.DATETIME:
            raise DaxError(
                f"Function '{fn}' only works with columns of type Date/Time; "
                f"{column.ref} isn't one."
            )
        all_values = ev.storage.ext(column.table)[column.name].filter(
            ev.storage.real_rows(column.table)
        )
        all_dates = _to_dates(all_values.drop_nulls().unique().to_list(), "dates")
        if all_values.null_count():
            raise NotSupportedError(f"{fn} over a date column with blanks")
        # The column means CALCULATETABLE(DISTINCT(<column>)): row contexts become filters.
        inner = ev.calculate_context([], ctx)
        # Time intelligence ignores the blank row that unmatched keys add to the dates.
        visible = ev.column_distinct(column, inner, blank=False).to_list()
        return _Dates(column, all_dates, _to_dates(visible, "dates"))
    if isinstance(node, TableCall) and node.name in _DATE_SET_FUNCTIONS:
        # A date-set function's result, as in PREVIOUSDAY(STARTOFMONTH(<dates>)).
        frame = ev.to_frame(ev.table(node, ctx))
        inner_column = frame.columns[0].column
        assert inner_column is not None
        base = _dates(ev, _innermost(node), ctx, fn)
        return _Dates(inner_column, base.all, _to_dates([r[0] for r in frame.rows], "dates"))
    raise NotSupportedError(f"{fn} with a table expression for its dates")


def _innermost(node: Any) -> Any:
    while isinstance(node, TableCall):
        node = node.args[0]
    return node


def _given(args: list[Any], i: int) -> bool:
    return i < len(args) and not isinstance(args[i], _Omitted)


def _filter_and_year_end(args: list[Any]) -> tuple[Any | None, Any | None]:
    """The filter and year end date arguments of TOTALYTD, OPENINGBALANCEYEAR and
    CLOSINGBALANCEYEAR. As in DAX, a text constant in the filter's place is the year
    end date."""
    third = args[2] if _given(args, 2) else None
    fourth = args[3] if _given(args, 3) else None
    if fourth is None and isinstance(third, Literal_) and isinstance(third.value, str):
        return None, third
    return third, fourth


def _year_end(ev: Evaluator, args: list[Any], i: int, ctx: Context, fn: str) -> YearEnd:
    """The year_end_date argument: "6/30" style text naming the last day of a month."""
    if not _given(args, i):
        return (12, 31)
    value = ev.scalar(args[i], ctx)
    match = re.fullmatch(r"\s*(\d{1,2})/(\d{1,2})\s*", value) if isinstance(value, str) else None
    if match is None:
        raise NotSupportedError(f'{fn} with a year end date that isn\'t text like "6/30"')
    month, day = int(match[1]), int(match[2])
    if not 1 <= month <= 12 or month == 2 or day != _last_day(2001, month):
        raise NotSupportedError(f"{fn} with a year end that isn't the last day of a month")
    return (month, day)


def _whole(value: Value, fn: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise NotSupportedError(f"{fn} with a number of intervals that isn't a whole number")
    return value


def _date_value(value: Value, fn: str) -> Date | None:
    if value is None:
        return None
    if not isinstance(value, dt.datetime):
        raise NotSupportedError(f"{fn} with a date argument that isn't a date")
    if value.time() != dt.time():
        raise NotSupportedError(f"{fn} with a date argument that has a time of day")
    return value.date()


# --- the date-set functions -------------------------------------------------------------

DateSetHandler = Callable[[Evaluator, list[Any], Context], TableValue]


def _to_date(months: int, fiscal: bool) -> DateSetHandler:
    """DATESYTD, DATESQTD and DATESMTD: from the start of the period of the last visible
    date to that date."""
    fn = {12: "DATESYTD", 3: "DATESQTD", 1: "DATESMTD"}[months]

    def handler(ev: Evaluator, args: list[Any], ctx: Context) -> TableValue:
        dates = _dates(ev, args[0], ctx, fn)
        year_end = _year_end(ev, args, 1, ctx, fn) if fiscal else (12, 31)
        if not dates.visible:
            return dates.result([])
        last = dates.visible[-1]
        return dates.result(dates.within(_period(last, months, year_end)[0], last))

    return handler


def _shift(dates: _Dates, number: int, interval: str) -> DateSet:
    """DATEADD: shift the visible dates by a number of intervals.

    A set with no gaps among the column's dates shifts as a range: its first and last
    dates move like EDATE, and when the last date ends its month (and the set has more
    than one date), the range runs to the end of the target month. A set with gaps
    shifts date by date, which was checked only when no two of its dates are adjacent.
    """
    visible = dates.visible
    if not visible:
        return []
    months = _MONTHS.get(interval)

    def move(d: Date, *, month_end: bool = False) -> Date:
        if months is None:
            return _add_days(d, number)
        return _add_months(d, number * months, month_end=month_end)

    index = {d: i for i, d in enumerate(dates.all)}
    positions = [index[d] for d in visible]
    if positions[-1] - positions[0] + 1 == len(positions):
        first = move(visible[0])
        last = move(visible[-1], month_end=len(visible) > 1 and _is_month_end(visible[-1]))
        return dates.within(first, last)
    if any(b - a == 1 for a, b in itertools.pairwise(positions)):
        raise NotSupportedError("DATEADD of a set of dates with gaps and adjacent dates")
    shifted = {move(d) for d in visible}
    return [d for d in dates.all if d in shifted]


def _interval(node: Any, fn: str) -> str:
    name: str = node.name
    if name not in ("DAY", "MONTH", "QUARTER", "YEAR"):
        raise NotSupportedError(f"{fn} with interval {name}")
    return name


def _dateadd(ev: Evaluator, args: list[Any], ctx: Context) -> TableValue:
    dates = _dates(ev, args[0], ctx, "DATEADD")
    number = _whole(ev.scalar(args[1], ctx), "DATEADD")
    return dates.result(_shift(dates, number, _interval(args[2], "DATEADD")))


def _sameperiodlastyear(ev: Evaluator, args: list[Any], ctx: Context) -> TableValue:
    dates = _dates(ev, args[0], ctx, "SAMEPERIODLASTYEAR")
    return dates.result(_shift(dates, -1, "YEAR"))


def _parallelperiod(ev: Evaluator, args: list[Any], ctx: Context) -> TableValue:
    """Whole periods, from the period of the first visible date to that of the last,
    shifted by a number of intervals."""
    dates = _dates(ev, args[0], ctx, "PARALLELPERIOD")
    number = _whole(ev.scalar(args[1], ctx), "PARALLELPERIOD")
    months = _MONTHS[_interval(args[2], "PARALLELPERIOD")]
    if not dates.visible:
        return dates.result([])
    first = _shift_period(_period(dates.visible[0], months), number * months)[0]
    last = _shift_period(_period(dates.visible[-1], months), number * months)[1]
    return dates.result(dates.within(first, last))


def _datesinperiod(ev: Evaluator, args: list[Any], ctx: Context) -> TableValue:
    """The dates from a start date over a number of intervals, forwards or backwards.
    A start date at the end of a month counts in whole months."""
    dates = _dates(ev, args[0], ctx, "DATESINPERIOD")
    start = _date_value(ev.scalar(args[1], ctx), "DATESINPERIOD")
    number = _whole(ev.scalar(args[2], ctx), "DATESINPERIOD")
    interval = _interval(args[3], "DATESINPERIOD")
    if start is None:
        raise NotSupportedError("DATESINPERIOD with a BLANK start date")
    if number == 0:
        raise NotSupportedError("DATESINPERIOD over zero intervals")
    months = _MONTHS.get(interval)
    if months is None:
        anchor = _add_days(start, number)
    else:
        anchor = _add_months(start, number * months, month_end=_is_month_end(start))
    if number < 0:
        return dates.result(dates.within(_add_days(anchor, 1), start))
    return dates.result(dates.within(start, _add_days(anchor, -1)))


def _datesbetween(ev: Evaluator, args: list[Any], ctx: Context) -> TableValue:
    """The column's dates in a range. A BLANK bound is the first or last date."""
    dates = _dates(ev, args[0], ctx, "DATESBETWEEN")
    first = _date_value(ev.scalar(args[1], ctx), "DATESBETWEEN")
    last = _date_value(ev.scalar(args[2], ctx), "DATESBETWEEN")
    if not dates.all:
        return dates.result([])
    return dates.result(dates.within(first or dates.all[0], last or dates.all[-1]))


def _adjacent(months: int | None, step: int, fiscal: bool = False) -> DateSetHandler:
    """PREVIOUSDAY/MONTH/QUARTER/YEAR (``step`` -1, before the first visible date) and
    NEXTDAY/... (``step`` 1, after the last visible date): the whole adjacent period."""
    unit = {None: "DAY", 1: "MONTH", 3: "QUARTER", 12: "YEAR"}[months]
    fn = ("PREVIOUS" if step < 0 else "NEXT") + unit

    def handler(ev: Evaluator, args: list[Any], ctx: Context) -> TableValue:
        dates = _dates(ev, args[0], ctx, fn)
        if fiscal and _year_end(ev, args, 1, ctx, fn) != (12, 31):
            raise NotSupportedError(f"{fn} with a year end date")
        if not dates.visible:
            return dates.result([])
        anchor = dates.visible[0] if step < 0 else dates.visible[-1]
        if months is None:
            day = _add_days(anchor, step)
            return dates.result(dates.within(day, day))
        first, last = _shift_period(_period(anchor, months), step * months)
        return dates.result(dates.within(first, last))

    return handler


def _edge(last: bool) -> DateSetHandler:
    """FIRSTDATE and LASTDATE: the first or last visible date."""
    fn = "LASTDATE" if last else "FIRSTDATE"

    def handler(ev: Evaluator, args: list[Any], ctx: Context) -> TableValue:
        dates = _dates(ev, args[0], ctx, fn)
        if not dates.visible:
            return dates.result([])
        return dates.result([dates.visible[-1] if last else dates.visible[0]])

    return handler


def _bound(months: int, end: bool, fiscal: bool = False) -> DateSetHandler:
    """STARTOFMONTH/QUARTER/YEAR: the column's first date in the period of the first
    visible date. ENDOF...: its last date in the period of the last visible date."""
    unit = {1: "MONTH", 3: "QUARTER", 12: "YEAR"}[months]
    fn = ("ENDOF" if end else "STARTOF") + unit

    def handler(ev: Evaluator, args: list[Any], ctx: Context) -> TableValue:
        dates = _dates(ev, args[0], ctx, fn)
        year_end = _year_end(ev, args, 1, ctx, fn) if fiscal else (12, 31)
        if not end and year_end != (12, 31):
            raise NotSupportedError(f"{fn} with a year end date")
        if not dates.visible:
            return dates.result([])
        anchor = dates.visible[-1] if end else dates.visible[0]
        found = dates.within(*_period(anchor, months, year_end))
        return dates.result(found[-1:] if end else found[:1])

    return handler


# --- functions built on the date sets ---------------------------------------------------


def _with_dates(set_name: str, fiscal: bool) -> Callable[..., Value]:
    """TOTALYTD, TOTALQTD and TOTALMTD: CALCULATE(<expression>, DATESxTD(<dates>),
    <filter>)."""

    def as_calculate(args: list[Any]) -> tuple[Any, list[Any]]:
        if fiscal:
            extra, year_end = _filter_and_year_end(args)
        else:
            extra, year_end = (args[2] if _given(args, 2) else None), None
        set_args = [args[1]] if year_end is None else [args[1], year_end]
        filters: list[Any] = [TableCall(set_name, set_args)]
        if extra is not None:
            filters.append(extra)
        return args[0], filters

    return _calculate_like(as_calculate)


def _calculate_like(
    as_calculate: Callable[[list[Any]], tuple[Any, list[Any]]],
) -> Callable[..., Value]:
    """A handler for a function that is CALCULATE(<expression>, <filters>...). The
    handler's ``as_calculate`` gives that form (SUMMARIZECOLUMNS uses it to evaluate
    every group at once)."""

    def handler(ev: Evaluator, args: list[Any], ctx: Context) -> Value:
        expression, filters = as_calculate(args)
        return ev.scalar(expression, ev.calculate_context(filters, ctx))

    handler.as_calculate = as_calculate  # type: ignore[attr-defined]
    return handler


def _balance(months: int, closing: bool) -> Callable[..., Value]:
    """OPENINGBALANCExxx: CALCULATE(<expression>, PREVIOUSDAY(STARTOFxxx(<dates>))).
    CLOSINGBALANCExxx: CALCULATE(<expression>, ENDOFxxx(<dates>))."""
    unit = {1: "MONTH", 3: "QUARTER", 12: "YEAR"}[months]

    def as_calculate(args: list[Any]) -> tuple[Any, list[Any]]:
        if months == 12:
            extra, year_end = _filter_and_year_end(args)
        else:
            extra, year_end = (args[2] if _given(args, 2) else None), None
        if closing:
            set_args = [args[1]] if year_end is None else [args[1], year_end]
            dates: Any = TableCall(f"ENDOF{unit}", set_args)
        else:
            if year_end is not None:
                raise NotSupportedError(f"OPENINGBALANCE{unit} with a year end date")
            dates = TableCall("PREVIOUSDAY", [TableCall(f"STARTOF{unit}", [args[1]])])
        filters: list[Any] = [dates]
        if extra is not None:
            filters.append(extra)
        return args[0], filters

    return _calculate_like(as_calculate)


# --- date tables ------------------------------------------------------------------------


def _calendar_frame(first: Date, last: Date) -> Frame:
    days = (last - first).days + 1
    rows: list[list[Value]] = []
    for i in range(days):
        d = first + dt.timedelta(days=i)
        rows.append([dt.datetime(d.year, d.month, d.day)])
    return Frame([ResultColumn("[Date]", DaxType.DATETIME)], rows)


def _calendar(ev: Evaluator, args: list[Any], ctx: Context) -> TableValue:
    first = _date_value(ev.scalar(args[0], ctx), "CALENDAR")
    last = _date_value(ev.scalar(args[1], ctx), "CALENDAR")
    if first is None or last is None:
        raise NotSupportedError("CALENDAR with a BLANK date")
    if first > last:
        raise NotSupportedError("CALENDAR with a start date after the end date")
    return _calendar_frame(first, last)


def _calendarauto(ev: Evaluator, args: list[Any], ctx: Context) -> TableValue:
    """Whole (fiscal) years covering the dates in the model's columns, leaving out
    calculated columns, as Microsoft documents it."""
    end_month = 12
    if _given(args, 0):
        value = ev.scalar(args[0], ctx)
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 12:
            raise NotSupportedError("CALENDARAUTO with a fiscal year end month outside 1 to 12")
        end_month = value
    found: list[dt.datetime] = []
    for table in ev.model.tables:
        for column in table.columns:
            if column.data_type is not DaxType.DATETIME or column.expression is not None:
                continue
            values = column.values.drop_nulls()
            if values.len():
                found.extend([values.min(), values.max()])  # type: ignore[list-item]
    if not found:
        raise NotSupportedError("CALENDARAUTO in a model with no dates")
    year_end = (end_month, _last_day(2001, end_month)) if end_month != 2 else None
    if year_end is None:
        raise NotSupportedError("CALENDARAUTO with a fiscal year ending in February")
    first = _period(min(found).date(), 12, year_end)[0]
    last = _period(max(found).date(), 12, year_end)[1]
    return _calendar_frame(first, last)


TABLES: dict[str, DateSetHandler] = {
    "CALENDAR": _calendar,
    "CALENDARAUTO": _calendarauto,
    "DATESYTD": _to_date(12, True),
    "DATESQTD": _to_date(3, False),
    "DATESMTD": _to_date(1, False),
    "SAMEPERIODLASTYEAR": _sameperiodlastyear,
    "DATEADD": _dateadd,
    "PARALLELPERIOD": _parallelperiod,
    "DATESINPERIOD": _datesinperiod,
    "DATESBETWEEN": _datesbetween,
    "PREVIOUSDAY": _adjacent(None, -1),
    "PREVIOUSMONTH": _adjacent(1, -1),
    "PREVIOUSQUARTER": _adjacent(3, -1),
    "PREVIOUSYEAR": _adjacent(12, -1, fiscal=True),
    "NEXTDAY": _adjacent(None, 1),
    "NEXTMONTH": _adjacent(1, 1),
    "NEXTQUARTER": _adjacent(3, 1),
    "NEXTYEAR": _adjacent(12, 1, fiscal=True),
    "FIRSTDATE": _edge(last=False),
    "LASTDATE": _edge(last=True),
    "STARTOFMONTH": _bound(1, end=False),
    "STARTOFQUARTER": _bound(3, end=False),
    "STARTOFYEAR": _bound(12, end=False, fiscal=True),
    "ENDOFMONTH": _bound(1, end=True),
    "ENDOFQUARTER": _bound(3, end=True),
    "ENDOFYEAR": _bound(12, end=True, fiscal=True),
}

SCALAR: dict[str, Callable[..., Value]] = {
    "TOTALYTD": _with_dates("DATESYTD", fiscal=True),
    "TOTALQTD": _with_dates("DATESQTD", fiscal=False),
    "TOTALMTD": _with_dates("DATESMTD", fiscal=False),
    "OPENINGBALANCEMONTH": _balance(1, closing=False),
    "OPENINGBALANCEQUARTER": _balance(3, closing=False),
    "OPENINGBALANCEYEAR": _balance(12, closing=False),
    "CLOSINGBALANCEMONTH": _balance(1, closing=True),
    "CLOSINGBALANCEQUARTER": _balance(3, closing=True),
    "CLOSINGBALANCEYEAR": _balance(12, closing=True),
}
