"""Turning values into text: DAX's implicit conversion and FORMAT (Phase 6).

Only the en-US culture is supported. Rules were checked against Power BI Desktop
(``tests/golden/cases/library``); anything outside them raises NotSupportedError.
"""

from __future__ import annotations

import datetime as dt
import math
import re
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

from . import _scalar as sc
from .errors import NotSupportedError

_MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]  # fmt: skip
_DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


# --- implicit conversion to text ------------------------------------------------------


def to_text(value: Any) -> str:
    """A value as text, the way DAX converts it (``value & ""``). BLANK is empty."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return _general_number(value)
    if isinstance(value, Decimal):
        return _plain(value)
    if isinstance(value, dt.datetime):
        return _general_date(value)
    raise TypeError(f"not a DAX value: {value!r}")


def _plain(d: Decimal) -> str:
    """A decimal without trailing zeros or an exponent."""
    text = format(d.normalize(), "f")
    return "0" if text in ("-0", "") else text


def _general_number(x: float) -> str:
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    if math.isnan(x):
        return "-nan(ind)"  # what Power BI shows for 0/0
    if x == 0:
        return "0"
    if abs(x) >= 1e15:
        mantissa, _, exponent = f"{x:.14e}".partition("e")
        digits = mantissa.rstrip("0").rstrip(".")
        if len(digits.lstrip("-")) > 1:
            # Only 1E+15, 1E+20 and the like were checked against Power BI.
            raise NotSupportedError("converting a large number with several digits to text")
        return f"{digits}E+{int(exponent):02d}"
    if abs(x) < 1e-6:
        raise NotSupportedError("converting a very small number to text")
    return _plain(Decimal(f"{x:.15g}"))


def _general_date(value: dt.datetime) -> str:
    date = f"{value.month}/{value.day}/{value.year}"
    if value.time() == dt.time():
        return date
    if value.microsecond:
        raise NotSupportedError("converting a date/time with fractions of a second to text")
    hour = value.hour % 12 or 12
    ampm = "AM" if value.hour < 12 else "PM"
    return f"{date} {hour}:{value.minute:02d}:{value.second:02d} {ampm}"


# --- FORMAT -----------------------------------------------------------------------------

_NAMED_NUMBER = {
    "fixed": "0.00",
    "standard": "#,##0.00",
    "percent": "0.00%",
    "scientific": "0.00E+00",
}
_NAMED_DATE = {
    "short date": "m/d/yyyy",
    "long date": "dddd, mmmm d, yyyy",
}


def format_value(value: Any, fmt: str, locale: str | None = None) -> str:
    """DAX ``FORMAT(value, format_string[, locale_name])``."""
    if locale is not None and locale.lower() != "en-us":
        raise NotSupportedError(f"FORMAT with the locale {locale!r} (only en-US)")
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        value = -1 if value else 0
    name = fmt.strip().lower()
    if name in ("", "general number", "general date"):
        if name == "general number" and isinstance(value, dt.datetime):
            raise NotSupportedError("FORMAT of a date with General Number")
        if name == "general date" and not isinstance(value, dt.datetime):
            raise NotSupportedError("FORMAT of a number with General Date")
        return to_text(value)
    if name == "currency":
        return _format_number(value, "$#,##0.00;($#,##0.00)")
    if name in _NAMED_NUMBER:
        return _format_number(value, _NAMED_NUMBER[name])
    if name in _NAMED_DATE:
        return _format_date(_as_date(value), _NAMED_DATE[name])
    if _is_date_format(fmt):
        return _format_date(_as_date(value), fmt)
    if isinstance(value, dt.datetime):
        raise NotSupportedError(f"FORMAT of a date with the format {fmt!r}")
    return _format_number(value, fmt)


def _as_date(value: Any) -> dt.datetime:
    if isinstance(value, dt.datetime):
        return value
    if isinstance(value, int | float | Decimal):
        return sc.from_serial(float(value))
    raise NotSupportedError("FORMAT of this value as a date")


def _tokens(fmt: str) -> list[tuple[str, str]]:
    """The format split into ("lit", text) for quoted or escaped text, and ("code", ch)."""
    result: list[tuple[str, str]] = []
    i = 0
    while i < len(fmt):
        ch = fmt[i]
        if ch == '"':
            end = fmt.find('"', i + 1)
            if end < 0:
                raise NotSupportedError("a format string with an unclosed quote")
            result.append(("lit", fmt[i + 1 : end]))
            i = end + 1
        elif ch == "\\":
            if i + 1 >= len(fmt):
                raise NotSupportedError("a format string ending in a backslash")
            result.append(("lit", fmt[i + 1]))
            i += 2
        else:
            result.append(("code", ch))
            i += 1
    return result


def _is_date_format(fmt: str) -> bool:
    codes = "".join(t for kind, t in _tokens(fmt) if kind == "code").lower()
    return not re.search(r"[0#]", codes) and bool(re.search(r"[ydhnsqm]", codes))


# --- numbers ----------------------------------------------------------------------------


def _sections(fmt: str) -> list[list[tuple[str, str]]]:
    sections: list[list[tuple[str, str]]] = [[]]
    for token in _tokens(fmt):
        if token == ("code", ";"):
            sections.append([])
        else:
            sections[-1].append(token)
    if len(sections) > 3:
        raise NotSupportedError("a format string with more than three sections")
    return sections


def _decimal(value: Any, digits: str) -> Decimal:
    if isinstance(value, Decimal):
        return value
    if isinstance(value, int):
        return Decimal(value)
    if math.isnan(value) or math.isinf(value):
        raise NotSupportedError("FORMAT of a number that isn't finite")
    if abs(value) >= 1e15:
        raise NotSupportedError("FORMAT of a number of 1E15 or more")
    return Decimal(f"{value:.15g}" if digits == "15" else repr(value))


def _format_number(value: Any, fmt: str) -> str:
    # Power BI's ROUND reads a double to 15 significant digits. FORMAT agrees on every
    # checked case, which can't tell 15 digits from the shortest exact text; refuse when
    # the two readings give different text.
    text = _format_decimal(_decimal(value, "15"), fmt)
    if isinstance(value, float) and _format_decimal(_decimal(value, "repr"), fmt) != text:
        raise NotSupportedError("FORMAT of a number that rounds differently at 15 digits")
    return text


def _format_decimal(number: Decimal, fmt: str) -> str:
    sections = _sections(fmt)
    section = sections[0]
    negative_section = False
    if number < 0 and len(sections) > 1 and sections[1]:
        section, negative_section = sections[1], True
    elif number == 0 and len(sections) > 2:
        section = sections[2]
    text = _apply_number(abs(number) if negative_section else number, section)
    return text


_LITERAL_CODES = set("$-+()/: ")


@dataclass(frozen=True)
class NumberSection:
    """One section of a custom number format, split around its digit pattern."""

    prefix: str  # text before the first digit placeholder
    pattern: str  # the placeholders, separators and point, such as "#,##0.00"
    exponent: tuple[str, int] | None  # ("+" or "-", exponent digits) for E+00 formats
    suffix: str  # text after the pattern (and the exponent)
    percent: int  # how many % signs (each multiplies by 100)


def number_section(tokens: list[tuple[str, str]]) -> NumberSection | None:
    """The parts of a number format section. None when the section is literal text only."""
    codes = [t for kind, t in tokens if kind == "code"]
    if not any(c in "0#" for c in codes):
        # No digit placeholders: the section is all literal text.
        if any(c not in _LITERAL_CODES for c in codes):
            raise NotSupportedError("a number format without digit placeholders")
        return None
    percent = codes.count("%")
    # Find the number pattern: from the first to the last digit placeholder, plus
    # separators inside it and scientific notation after it.
    first = next(i for i, (k, t) in enumerate(tokens) if k == "code" and t in "0#")
    e_at = next(
        (i for i, (k, t) in enumerate(tokens) if k == "code" and t in "Ee" and i > first),
        len(tokens),
    )
    last = max(i for i, (k, t) in enumerate(tokens[:e_at]) if k == "code" and t in "0#.,")
    exponent: tuple[str, int] | None = None
    j = last + 1
    if (
        j + 1 < len(tokens)
        and tokens[j][0] == "code"
        and tokens[j][1] in "Ee"
        and tokens[j + 1] in (("code", "+"), ("code", "-"))
    ):
        k = j + 2
        while k < len(tokens) and tokens[k] == ("code", "0"):
            k += 1
        if k == j + 2:
            raise NotSupportedError("a scientific format without exponent digits")
        exponent = (tokens[j + 1][1], k - (j + 2))
        tail = tokens[k:]
    else:
        tail = tokens[last + 1 :]
    pattern = "".join(t for _, t in tokens[first : last + 1])
    if any(k == "lit" for k, _ in tokens[first : last + 1]):
        raise NotSupportedError("literal text inside a number format")
    if re.search(r"[^0#.,]", pattern):
        raise NotSupportedError(f"the number format {pattern!r}")
    for kind, t in tokens[:first] + tail:
        if kind == "code" and t not in _LITERAL_CODES and t != "%":
            raise NotSupportedError(f"the character {t!r} in a number format")
    prefix = "".join(t for _, t in tokens[:first])
    suffix = "".join(t for _, t in tail)
    return NumberSection(prefix, pattern, exponent, suffix, percent)


def _apply_number(number: Decimal, tokens: list[tuple[str, str]]) -> str:
    section = number_section(tokens)
    if section is None:
        return "".join(t for _, t in tokens)
    number *= Decimal(100) ** section.percent
    body = _digits(number, section.pattern, section.exponent)
    if body.startswith("-"):
        # The minus sign goes before any text: FORMAT(-5, "$#,0") is "-$5".
        return "-" + section.prefix + body[1:] + section.suffix
    return section.prefix + body + section.suffix


def _digits(number: Decimal, pattern: str, exponent: tuple[str, int] | None) -> str:
    int_part, _, frac_part = pattern.partition(".")
    if "." in frac_part:
        raise NotSupportedError("a number format with two decimal points")
    # Commas right after the last integer placeholder divide by 1,000 each.
    scale = len(int_part) - len(int_part.rstrip(","))
    int_part = int_part.rstrip(",")
    grouping = "," in int_part
    int_part = int_part.replace(",", "")
    frac_part = frac_part.replace(",", "")  # commas after the point are ignored
    number /= Decimal(1000) ** scale
    min_int = int_part.count("0")
    min_frac = len(frac_part.rstrip("#"))
    max_frac = len(frac_part)
    if min_int and "#" in int_part.lstrip("#"):
        raise NotSupportedError("a number format with # after 0 before the decimal point")
    if "0" in frac_part[min_frac:]:
        raise NotSupportedError("a number format with 0 after # after the decimal point")

    sign = "-" if number < 0 else ""
    number = abs(number)
    exp_text = ""
    if exponent is not None:
        if grouping:
            raise NotSupportedError("thousands separators in a scientific format")
        power = 0
        if number != 0:
            power = number.adjusted() - (max(min_int, 1) - 1)
            number = number.scaleb(-power)
            rounded = number.quantize(Decimal(1).scaleb(-max_frac), rounding=ROUND_HALF_UP)
            if rounded >= Decimal(10) ** max(min_int, 1):
                power += 1
                number = number.scaleb(-1)
        sign_char, width = exponent
        exp_sign = "-" if power < 0 else ("+" if sign_char == "+" else "")
        exp_text = f"E{exp_sign}{abs(power):0{width}d}"

    rounded = number.quantize(Decimal(1).scaleb(-max_frac), rounding=ROUND_HALF_UP)
    if rounded == 0:
        sign = ""  # FORMAT(-0.4, "0") is "0"
    whole, _, frac = format(rounded, "f").partition(".")
    frac = frac.ljust(max_frac, "0")
    while len(frac) > min_frac and frac.endswith("0"):
        frac = frac[:-1]
    whole = whole.lstrip("0")
    whole = whole.rjust(min_int, "0")
    if grouping and whole:
        whole = f"{int(whole):,}".rjust(len(whole), "0") if min_int <= len(whole) else whole
    text = whole + ("." + frac if "." in pattern else "")
    return sign + text + exp_text


# --- dates ------------------------------------------------------------------------------

_DATE_TOKEN = re.compile(
    r"yyyy|yy|mmmm|mmm|mm|m|dddd|ddd|dd|d|hh|h|nn|n|ss|s|q|am/pm|AM/PM", re.IGNORECASE
)


def date_tokens(fmt: str) -> tuple[list[tuple[str, str]], bool]:
    """A date format as ("lit", text) and ("tok", code) pieces, and whether it shows a
    12-hour clock (has AM/PM)."""
    pieces: list[tuple[str, str]] = []
    for kind, text in _tokens(fmt):
        if kind == "lit":
            pieces.append(("lit", text))
            continue
        if pieces and pieces[-1][0] == "code":
            pieces[-1] = ("code", pieces[-1][1] + text)
        else:
            pieces.append(("code", text))
    twelve = any(k == "code" and "am/pm" in t.lower() for k, t in pieces)
    tokens: list[tuple[str, str]] = []
    for kind, text in pieces:
        if kind == "lit":
            tokens.append(("lit", text))
            continue
        pos = 0
        for m in _DATE_TOKEN.finditer(text):
            for ch in text[pos : m.start()]:
                if ch not in " /-:,.'":
                    raise NotSupportedError(f"the character {ch!r} in a date format")
                tokens.append(("lit", ch))
            tokens.append(("tok", m.group(0)))
            pos = m.end()
        for ch in text[pos:]:
            if ch not in " /-:,.'":
                raise NotSupportedError(f"the character {ch!r} in a date format")
            tokens.append(("lit", ch))
    return tokens, twelve


def _format_date(value: dt.datetime, fmt: str) -> str:
    out: list[str] = []
    tokens, twelve = date_tokens(fmt)
    for i, (kind, text) in enumerate(tokens):
        if kind == "lit":
            out.append(text)
            continue
        low = text.lower()
        if low in ("m", "mm") and _is_minute(tokens, i):
            out.append(f"{value.minute:02d}" if low == "mm" else str(value.minute))
        elif low == "yyyy":
            out.append(f"{value.year:04d}")
        elif low == "yy":
            out.append(f"{value.year % 100:02d}")
        elif low == "mmmm":
            out.append(_MONTHS[value.month - 1])
        elif low == "mmm":
            out.append(_MONTHS[value.month - 1][:3])
        elif low == "mm":
            out.append(f"{value.month:02d}")
        elif low == "m":
            out.append(str(value.month))
        elif low == "dddd":
            out.append(_DAYS[value.weekday()])
        elif low == "ddd":
            out.append(_DAYS[value.weekday()][:3])
        elif low == "dd":
            out.append(f"{value.day:02d}")
        elif low == "d":
            out.append(str(value.day))
        elif low in ("h", "hh"):
            hour = (value.hour % 12 or 12) if twelve else value.hour
            out.append(f"{hour:02d}" if low == "hh" else str(hour))
        elif low in ("n", "nn"):
            out.append(f"{value.minute:02d}" if low == "nn" else str(value.minute))
        elif low in ("s", "ss"):
            out.append(f"{value.second:02d}" if low == "ss" else str(value.second))
        elif low == "q":
            out.append(str((value.month - 1) // 3 + 1))
        elif low == "am/pm":
            if text not in ("AM/PM", "am/pm"):
                raise NotSupportedError(f"the date format token {text!r}")
            ampm = "AM" if value.hour < 12 else "PM"
            out.append(ampm if text == "AM/PM" else ampm.lower())
    return "".join(out)


def _is_minute(tokens: list[tuple[str, str]], i: int) -> bool:
    """``m`` or ``mm`` right after an hour or right before seconds means minutes."""
    for j in range(i - 1, -1, -1):
        if tokens[j][0] == "tok":
            return tokens[j][1].lower() in ("h", "hh")
    for j in range(i + 1, len(tokens)):
        if tokens[j][0] == "tok":
            return tokens[j][1].lower() in ("s", "ss")
    return False
