"""DAX format strings as Streamlit format strings.

Streamlit formats numbers in the browser with a printf-style string (its fork of
sprintf-js) and dates with a moment.js string. Both are used the same way by
``st.column_config.NumberColumn`` / ``DateColumn`` / ``DatetimeColumn`` / ``TimeColumn``
and by ``st.metric``. This module turns a DAX format string into one of those, and
prepares values so the browser shows exactly the text Power BI shows.

Two things printf can't do on its own, so :func:`streamlit_values` does them first:

* scaling: ``%`` multiplies by 100 and a comma after the last digit divides by 1,000;
* rounding: the browser rounds the binary double (``1.005`` shows as ``1.00``), where
  Power BI rounds the 15-digit decimal half away from zero (``1.01``), and a negative
  number that rounds to zero keeps its minus sign (``-0``), where Power BI drops it.

:func:`streamlit_values` also checks every value: it formats the value the way Power BI
does (``FORMAT``) and the way Streamlit will, and raises ``NotSupportedError`` if they
differ. The Streamlit side is a copy of Streamlit's own code (``render_printf`` and
``render_moment``), which ``tests/test_display.py`` checks against the real thing.

Nothing here imports Streamlit, and nothing here draws anything.
"""

from __future__ import annotations

import datetime as dt
import re
from collections.abc import Iterable
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, overload

import polars as pl

from . import _format
from .errors import NotSupportedError

__all__ = ["streamlit_format", "streamlit_values"]


# --- converting the format string ---------------------------------------------------------


@dataclass(frozen=True)
class _Number:
    """A number format Streamlit can show: printf text plus how to prepare values."""

    printf: str
    decimals: int
    scale: Decimal  # values are multiplied by this before rounding


@dataclass(frozen=True)
class _Date:
    moment: str


_NAMED_NUMBER = {
    "fixed": "0.00",
    "standard": "#,##0.00",
    "percent": "0.00%",
    "currency": "$#,##0.00;($#,##0.00)",
}
_NAMED_DATE = {"short date": "m/d/yyyy", "long date": "dddd, mmmm d, yyyy"}


def streamlit_format(format_string: str, *, nonnegative: bool = False) -> str:
    """The Streamlit format string that shows values the way ``format_string`` does in
    Power BI.

    Number formats become printf strings (``"#,0.00"`` gives ``"%,.2f"``) for the
    ``format`` argument of ``st.column_config.NumberColumn`` and ``st.metric``. Date
    formats become moment.js strings (``"mmm yyyy"`` gives ``"MMM YYYY"``) for
    ``st.column_config.DateColumn``, ``DatetimeColumn`` and ``TimeColumn``.

    Pass the values through :func:`streamlit_values` with the same format string before
    you show them. It applies the scaling (``%``, thousands commas) and the rounding
    printf can't, and checks each value.

    Formats Streamlit can't show exactly raise ``NotSupportedError``: optional digits
    (``0.##``), scientific notation, zero padding (``000``), text-only sections, a zero
    section that differs, General Number and General Date. A negative section that
    shows negatives other than with a minus sign right before the digits raises too:
    ``0;(0)``, the named Currency format, and text before the number such as
    ``"$#,0"`` (Power BI shows ``-$5``, Streamlit ``$-5``). Pass ``nonnegative=True``
    to promise there are no negative values and convert those anyway.
    """
    spec = _convert(format_string, nonnegative)
    return spec.printf if isinstance(spec, _Number) else spec.moment


def _convert(fmt: str, nonnegative: bool) -> _Number | _Date:
    name = fmt.strip().lower()
    if name in ("", "general number", "general date"):
        raise NotSupportedError(
            f"showing the format {fmt!r} in Streamlit (what it shows depends on the value)"
        )
    if name == "scientific":
        raise NotSupportedError("showing scientific notation in Streamlit")
    if name in _NAMED_DATE:
        return _date(_NAMED_DATE[name])
    if name in _NAMED_NUMBER:
        return _number(_NAMED_NUMBER[name], fmt, nonnegative)
    if _format._is_date_format(fmt):
        return _date(fmt)
    return _number(fmt, fmt, nonnegative)


def _number(pattern: str, original: str, nonnegative: bool) -> _Number:
    _format.format_value(1.0, pattern)  # the engine's own checks of the format
    sections = _format._sections(pattern)
    positive = _section(sections[0], original)
    if not nonnegative:
        # Streamlit shows a negative number as prefix, minus sign, digits. Power BI
        # shows the negative section, or else a minus sign before the prefix.
        prefix, rest = _split(positive.printf)
        if len(sections) > 1 and sections[1]:
            negative = _section(sections[1], original)
        else:
            negative = _Number("-" + prefix + rest, positive.decimals, positive.scale)
        if negative != _Number(prefix + "-" + rest, positive.decimals, positive.scale):
            raise NotSupportedError(
                f"showing negative numbers with {original!r} in Streamlit (it can only put "
                "the minus sign right before the digits; pass nonnegative=True if the "
                "values are never negative)"
            )
    if len(sections) > 2 and _section(sections[2], original) != positive:
        raise NotSupportedError(f"showing the zero section of {original!r} in Streamlit")
    return positive


def _split(printf: str) -> tuple[str, str]:
    """A section's printf text as (prefix, placeholder and suffix)."""
    at = re.search(r"(?<!%)(?:%%)*%[,.]", printf)
    assert at is not None
    start = at.end() - 2
    return printf[:start], printf[start:]


def _section(tokens: list[tuple[str, str]], original: str) -> _Number:
    section = _format.number_section(tokens)
    if section is None:
        raise NotSupportedError(f"showing a text-only section of {original!r} in Streamlit")
    if section.exponent is not None:
        raise NotSupportedError("showing scientific notation in Streamlit")
    int_part, _, frac_part = section.pattern.partition(".")
    scale_commas = len(int_part) - len(int_part.rstrip(","))
    int_part = int_part.rstrip(",")
    grouping = "," in int_part
    int_part = int_part.replace(",", "")
    frac_part = frac_part.replace(",", "")
    if int_part.count("0") != 1 or not int_part.endswith("0"):
        if int_part.count("0") == 0:
            raise NotSupportedError(
                f"showing {original!r} in Streamlit (it always shows a digit before the point)"
            )
        raise NotSupportedError(f"showing the zero padding of {original!r} in Streamlit")
    if "#" in frac_part:
        raise NotSupportedError(
            f"showing the optional decimal digits (#) of {original!r} in Streamlit"
        )
    decimals = len(frac_part)
    placeholder = f"%{',' if grouping else ''}.{decimals}f"
    printf = _escape(section.prefix) + placeholder + _escape(section.suffix)
    scale = Decimal(100) ** section.percent / Decimal(1000) ** scale_commas
    return _Number(printf, decimals, scale)


def _escape(text: str) -> str:
    return text.replace("%", "%%")


_MOMENT = {
    "yyyy": "YYYY", "yy": "YY", "mmmm": "MMMM", "mmm": "MMM", "mm": "MM", "m": "M",
    "dddd": "dddd", "ddd": "ddd", "dd": "DD", "d": "D", "nn": "mm", "n": "m",
    "ss": "ss", "s": "s", "q": "Q", "AM/PM": "A", "am/pm": "a",
}  # fmt: skip


def _date(fmt: str) -> _Date:
    tokens, twelve = _format.date_tokens(fmt)
    out: list[str] = []
    for i, (kind, text) in enumerate(tokens):
        if kind == "lit":
            if not text:
                continue
            if "[" in text or "]" in text:
                raise NotSupportedError(f"showing the text {text!r} of a date format in Streamlit")
            out.append(text if all(ch in " /-:,." for ch in text) else f"[{text}]")
            continue
        low = text.lower()
        if low in ("m", "mm") and _format._is_minute(tokens, i):
            out.append(low)
        elif low in ("h", "hh"):
            out.append(low if twelve else low.upper())
        elif low == "am/pm":
            if text not in _MOMENT:
                raise NotSupportedError(f"the date format token {text!r}")
            out.append(_MOMENT[text])
        else:
            out.append(_MOMENT[low])
    return _Date("".join(out))


# --- preparing values ---------------------------------------------------------------------


@overload
def streamlit_values(
    values: pl.Series, format_string: str, *, nonnegative: bool = ...
) -> pl.Series: ...
@overload
def streamlit_values(
    values: int | float | Decimal | None, format_string: str, *, nonnegative: bool = ...
) -> float | None: ...
@overload
def streamlit_values(
    values: dt.date, format_string: str, *, nonnegative: bool = ...
) -> dt.date: ...
@overload
def streamlit_values(
    values: list[Any] | tuple[Any, ...], format_string: str, *, nonnegative: bool = ...
) -> list[Any]: ...
def streamlit_values(values: Any, format_string: str, *, nonnegative: bool = False) -> Any:
    """Values ready to show in Streamlit with ``streamlit_format(format_string)``.

    For a number format, each number is scaled (``%`` multiplies by 100, a comma after
    the last digit divides by 1,000) and rounded the way Power BI rounds it, and returned
    as a float. For a date format, the dates are returned as they are. BLANK (None)
    stays None.

    Every value is checked: if Streamlit would show it differently from Power BI's
    ``FORMAT(value, format_string)``, this raises ``NotSupportedError``. That catches,
    for example, a negative value when you passed ``nonnegative=True``.

    Takes one value (for ``st.metric``), a polars Series (returned as a Series, for a
    dataframe column), or any other iterable (returned as a list). Pandas Series are
    accepted and come back as polars.
    """
    spec = _convert(format_string, nonnegative)
    if isinstance(values, pl.Series):
        out = [_prepare(v, spec, format_string) for v in values.to_list()]
        dtype = pl.Float64 if isinstance(spec, _Number) else values.dtype
        return pl.Series(values.name, out, dtype=dtype)
    if _is_pandas_series(values):
        series = pl.Series(values.name, pl.from_pandas(values))
        return streamlit_values(series, format_string, nonnegative=nonnegative)
    if isinstance(values, Iterable) and not isinstance(values, str | bytes):
        return [_prepare(v, spec, format_string) for v in values]
    return _prepare(values, spec, format_string)


def _is_pandas_series(values: Any) -> bool:
    return type(values).__name__ == "Series" and type(values).__module__.startswith("pandas")


def _prepare(value: Any, spec: _Number | _Date, fmt: str) -> Any:
    if value is None:
        return None
    if isinstance(spec, _Date):
        if isinstance(value, dt.datetime):
            moment_value = value
        elif isinstance(value, dt.date):
            moment_value = dt.datetime(value.year, value.month, value.day)
        elif isinstance(value, dt.time):
            moment_value = dt.datetime.combine(dt.date(1899, 12, 30), value)
        else:
            raise NotSupportedError(f"showing {value!r} with the date format {fmt!r}")
        _check(render_moment(spec.moment, moment_value), moment_value, fmt, value)
        return value
    if isinstance(value, bool) or not isinstance(value, int | float | Decimal):
        raise NotSupportedError(f"showing {value!r} with the number format {fmt!r}")
    number = _format._decimal(value, "15") * spec.scale
    rounded = number.quantize(Decimal(1).scaleb(-spec.decimals), rounding=ROUND_HALF_UP)
    shown = float(rounded) if rounded != 0 else 0.0
    _check(render_printf(spec.printf, shown), value, fmt, value)
    return shown


def _check(streamlit_text: str, dax_value: Any, fmt: str, original: Any) -> None:
    power_bi_text = _format.format_value(dax_value, fmt)
    if streamlit_text != power_bi_text:
        raise NotSupportedError(
            f"showing {original!r} with {fmt!r} in Streamlit (Power BI shows "
            f"{power_bi_text!r}, Streamlit would show {streamlit_text!r})"
        )


# --- what Streamlit shows ---------------------------------------------------------------
# Copies of the parts of Streamlit's frontend this module relies on (Streamlit 1.67):
# its sprintf-js fork for printf number formats, and moment.js (English) for dates.

_PLACEHOLDER = re.compile(r"%(,)?\.(\d+)f")


def render_printf(printf: str, value: float) -> str:
    """What Streamlit shows for ``value`` with a printf format made by this module."""
    out: list[str] = []
    i = 0
    while i < len(printf):
        if printf.startswith("%%", i):
            out.append("%")
            i += 2
            continue
        match = _PLACEHOLDER.match(printf, i)
        if printf[i] == "%":
            if match is None:
                raise ValueError(f"not a printf format from streamlit_format: {printf!r}")
            out.append(_printf_number(value, match.group(1) == ",", int(match.group(2))))
            i = match.end()
            continue
        out.append(printf[i])
        i += 1
    return "".join(out)


def _printf_number(value: float, grouping: bool, decimals: int) -> str:
    text = _to_fixed(value, decimals)
    sign = ""
    if not value >= 0:  # -0.0 >= 0 in JavaScript too
        sign = "-"
        text = text.lstrip("-")
    if grouping:
        whole, point, frac = text.partition(".")
        text = f"{int(whole):,}" + point + frac
    return sign + text


def _to_fixed(value: float, decimals: int) -> str:
    """JavaScript's ``Number.prototype.toFixed``: rounds the exact binary value, and
    takes the larger magnitude on an exact tie."""
    if abs(value) >= 1e21:
        raise NotSupportedError("showing a number of 1E21 or more in Streamlit")
    negative = value < 0
    exact = Decimal(abs(value)).quantize(Decimal(1).scaleb(-decimals), rounding=ROUND_HALF_UP)
    return ("-" if negative else "") + format(exact, "f")


_MONTHS = _format._MONTHS
_DAYS = _format._DAYS
_MOMENT_TOKEN = re.compile(
    r"\[[^\[]*\]|YYYY|YY|MMMM|MMM|MM|M|dddd|ddd|DD|D|HH|H|hh|h|mm|m|ss|s|Q|A|a"
)


def render_moment(moment: str, value: dt.datetime) -> str:
    """What moment.js shows for ``value`` with a format made by this module."""

    def token(match: re.Match[str]) -> str:
        t = match.group(0)
        hour12 = value.hour % 12 or 12
        table = {
            "YYYY": f"{value.year:04d}",
            "YY": f"{value.year % 100:02d}",
            "MMMM": _MONTHS[value.month - 1],
            "MMM": _MONTHS[value.month - 1][:3],
            "MM": f"{value.month:02d}",
            "M": str(value.month),
            "dddd": _DAYS[value.weekday()],
            "ddd": _DAYS[value.weekday()][:3],
            "DD": f"{value.day:02d}",
            "D": str(value.day),
            "HH": f"{value.hour:02d}",
            "H": str(value.hour),
            "hh": f"{hour12:02d}",
            "h": str(hour12),
            "mm": f"{value.minute:02d}",
            "m": str(value.minute),
            "ss": f"{value.second:02d}",
            "s": str(value.second),
            "Q": str((value.month - 1) // 3 + 1),
            "A": "AM" if value.hour < 12 else "PM",
            "a": "am" if value.hour < 12 else "pm",
        }
        return t[1:-1] if t.startswith("[") else table[t]

    return _MOMENT_TOKEN.sub(token, moment)
