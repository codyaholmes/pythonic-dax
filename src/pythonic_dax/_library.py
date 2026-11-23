"""The scalar function library (Phase 6): logic, information, math, text, conversion and
date/time functions, and the INFO.VIEW functions.

Every rule here was checked against Power BI Desktop (``tests/golden/cases/library``).
Where Power BI's answer wasn't checked, the functions raise NotSupportedError.
"""

from __future__ import annotations

import calendar
import datetime as dt
import math
import re
from collections.abc import Callable
from decimal import (
    ROUND_CEILING,
    ROUND_DOWN,
    ROUND_FLOOR,
    ROUND_HALF_UP,
    ROUND_UP,
    Decimal,
    localcontext,
)
from decimal import (
    Context as DecimalContext,
)
from typing import TYPE_CHECKING, Any

from . import _format
from . import _scalar as sc
from ._evaluate import (
    BLANK_TYPE,
    Context,
    Evaluator,
    Frame,
    ResultColumn,
    StaticType,
    TableValue,
    Value,
    arithmetic_type,
    convert,
    infer_type,
    unify,
)
from .errors import CalculationError, DaxError, NotSupportedError
from .expr import Call, Keyword, _Omitted, is_column
from .objects import Cardinality, CrossFilter, SummarizeBy
from .types import DaxType

if TYPE_CHECKING:
    from .model import Model

_NUMERIC = (DaxType.INTEGER, DaxType.DECIMAL, DaxType.CURRENCY)
_DECIMALS = DecimalContext(prec=60)


# --- argument helpers -----------------------------------------------------------------


def _given(args: list[Any], i: int) -> bool:
    return i < len(args) and not isinstance(args[i], _Omitted)


def _arg(ev: Evaluator, args: list[Any], i: int, ctx: Context) -> Value:
    """Argument ``i``, or BLANK when it was left out."""
    return ev.scalar(args[i], ctx) if _given(args, i) else None


def _kind(value: Value) -> str:
    return sc.type_label(sc.type_of(value))


def _number(value: Value, fn: str) -> int | float | Decimal | None:
    """A numeric argument. BLANK stays None; other types aren't verified."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float | Decimal):
        raise NotSupportedError(f"{fn} of a {_kind(value)} value")
    if isinstance(value, float) and not math.isfinite(value):
        raise NotSupportedError(f"{fn} of a number that isn't finite")
    return value


def _required(value: Value, fn: str) -> int | float | Decimal:
    number = _number(value, fn)
    if number is None:
        raise NotSupportedError(f"{fn} of BLANK")
    return number


def _whole(value: Value, fn: str, what: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise NotSupportedError(f"{fn} with a {what} that isn't a whole number")
    if minimum is not None and value < minimum:
        raise NotSupportedError(f"{fn} with a {what} below {minimum}")
    return value


def _dec(number: int | float | Decimal) -> Decimal:
    """A number as a decimal. Doubles are read to 15 significant digits, as Power BI's
    rounding functions do: ROUND(1.004999999999999, 2) is 1.01."""
    if isinstance(number, float):
        if abs(number) >= 1e15:
            raise NotSupportedError("rounding a number of 1E15 or more")
        return Decimal(f"{number:.15g}")
    return Decimal(number)


def _text(value: Value) -> str | None:
    return None if value is None else _format.to_text(value)


def _date(value: Value, fn: str) -> dt.datetime | None:
    if value is None or isinstance(value, dt.datetime):
        return value
    raise NotSupportedError(f"{fn} of a {_kind(value)} value")


def _non_finite(value: Value) -> bool:
    return isinstance(value, float) and not math.isfinite(value)


# --- logical and information ----------------------------------------------------------


def _divide(ev: Evaluator, args: list[Any], ctx: Context) -> Value:
    numerator = _number(ev.scalar(args[0], ctx), "DIVIDE")
    denominator = _number(ev.scalar(args[1], ctx), "DIVIDE")
    if numerator is None:
        value = None
    elif denominator is None or denominator == 0:
        value = _arg(ev, args, 2, ctx)
    else:
        value = sc.arithmetic("/", numerator, denominator)
    return convert(value, infer_type(Call("DIVIDE", args)))


def _divide_type(args: list[Any]) -> StaticType:
    base = arithmetic_type("/", infer_type(args[0]), infer_type(args[1]))
    alternate = infer_type(args[2]) if _given(args, 2) else BLANK_TYPE
    return unify([base, alternate])


def _switch_results(args: list[Any]) -> list[Any]:
    rest = args[1:]
    results = rest[1::2][: len(rest) // 2]
    if len(rest) % 2 == 1:
        results.append(rest[-1])
    return results


def _switch(ev: Evaluator, args: list[Any], ctx: Context) -> Value:
    value = ev.scalar(args[0], ctx)
    rest = args[1:]
    result = None
    for i in range(0, len(rest) - 1, 2):
        if sc.compare("==", value, ev.scalar(rest[i], ctx)):
            result = ev.scalar(rest[i + 1], ctx)
            break
    else:
        if len(rest) % 2 == 1:
            result = ev.scalar(rest[-1], ctx)
    return convert(result, infer_type(Call("SWITCH", args)))


def _switch_type(args: list[Any]) -> StaticType:
    types = [infer_type(r) for r in _switch_results(args)]
    if (len(args) - 1) % 2 == 0:
        types.append(BLANK_TYPE)
    return unify(types)


def _coalesce(ev: Evaluator, args: list[Any], ctx: Context) -> Value:
    result = None
    for arg in args:
        result = ev.scalar(arg, ctx)
        if result is not None:
            break
    return convert(result, infer_type(Call("COALESCE", args)))


def _attempt(ev: Evaluator, node: Any, ctx: Context, fn: str) -> tuple[bool, Value]:
    """(is_error, value). Errors are calculation errors and results that aren't finite."""
    try:
        value = ev.scalar(node, ctx)
    except CalculationError:
        return True, None
    except DaxError as error:
        raise NotSupportedError(
            f"{fn} around this error, which Power BI may not catch: {error}"
        ) from None
    return _non_finite(value), value


def _iferror(ev: Evaluator, args: list[Any], ctx: Context) -> Value:
    failed, value = _attempt(ev, args[0], ctx, "IFERROR")
    if failed:
        value = ev.scalar(args[1], ctx)
    return convert(value, infer_type(Call("IFERROR", args)))


def _iserror(ev: Evaluator, args: list[Any], ctx: Context) -> Value:
    return _attempt(ev, args[0], ctx, "ISERROR")[0]


def _isblank(ev: Evaluator, args: list[Any], ctx: Context) -> Value:
    return ev.scalar(args[0], ctx) is None


def _isnumber(ev: Evaluator, args: list[Any], ctx: Context) -> Value:
    value = ev.scalar(args[0], ctx)
    return not isinstance(value, bool) and isinstance(value, int | float | Decimal | dt.datetime)


def _istext(ev: Evaluator, args: list[Any], ctx: Context) -> Value:
    return isinstance(ev.scalar(args[0], ctx), str)


def _islogical(ev: Evaluator, args: list[Any], ctx: Context) -> Value:
    return isinstance(ev.scalar(args[0], ctx), bool)


def _isempty(ev: Evaluator, args: list[Any], ctx: Context) -> Value:
    return not ev.rows_of(ev.table(args[0], ctx))


def _contains(ev: Evaluator, args: list[Any], ctx: Context) -> Value:
    frame = ev.to_frame(ev.table(args[0], ctx))
    tests = []
    for column, node in zip(args[1::2], args[2::2], strict=True):
        i = frame.find(column)
        if i is None:
            raise DaxError(f"CONTAINS: the column {column.ref} isn't in the table.")
        value = ev.scalar(node, ctx)
        if value is None or value in (0, "", False):
            raise NotSupportedError("CONTAINS searching for BLANK, 0, empty text or FALSE")
        tests.append((i, value))
    return any(all(sc.compare("==", row[i], v) for i, v in tests) for row in frame.rows)


def _containsrow(ev: Evaluator, args: list[Any], ctx: Context) -> Value:
    frame = ev.to_frame(ev.table(args[0], ctx))
    values = [ev.scalar(node, ctx) for node in args[1:]]
    if len(frame.columns) != len(values):
        raise DaxError(
            "The number of columns in the table must match the number of values in CONTAINSROW."
        )
    return any(
        all(sc.compare("==", a, b) for a, b in zip(row, values, strict=True)) for row in frame.rows
    )


# --- math -------------------------------------------------------------------------------


def _abs(ev: Evaluator, args: list[Any], ctx: Context) -> Value:
    x = _number(ev.scalar(args[0], ctx), "ABS")
    if x is None:
        return None
    return sc.check_int(abs(x)) if isinstance(x, int) else abs(x)


def _same_type(args: list[Any]) -> StaticType:
    t = infer_type(args[0])
    if t == BLANK_TYPE:
        return DaxType.INTEGER
    return t if t in _NUMERIC else None


def _rounder(name: str, mode: str, default_digits: int | None = None) -> Callable[..., Value]:
    def handler(ev: Evaluator, args: list[Any], ctx: Context) -> Value:
        x = _number(ev.scalar(args[0], ctx), name)
        if _given(args, 1):
            digits = _whole(ev.scalar(args[1], ctx), name, "number of digits")
        elif default_digits is not None:
            digits = default_digits
        else:  # pragma: no cover - the API requires the argument
            raise NotSupportedError(f"{name} without a number of digits")
        if x is None:
            return None
        return _round(x, digits, mode, name)

    return handler


def _round(x: int | float | Decimal, digits: int, mode: str, name: str) -> Value:
    if digits > 15:
        raise NotSupportedError(f"{name} to more than 15 digits")
    step = Decimal(1).scaleb(-digits)
    if isinstance(x, int):
        if digits >= 0:
            return x
        with localcontext(_DECIMALS):
            return sc.check_int(int(Decimal(x).quantize(step, rounding=mode)))
    with localcontext(_DECIMALS):
        rounded = _dec(x).quantize(step, rounding=mode)
    if isinstance(x, Decimal):
        return rounded
    return float(rounded)


def _int(ev: Evaluator, args: list[Any], ctx: Context) -> Value:
    value = ev.scalar(args[0], ctx)
    if isinstance(value, bool):  # INT(TRUE()) is 1
        return int(value)
    if isinstance(value, str):  # INT("202611") converts the text, as VALUE does
        value = parse_number(value)
    x = _number(value, "INT")
    if x is None:
        return None
    if isinstance(x, int):
        return x
    floored = int(_dec(x).to_integral_value(rounding=ROUND_FLOOR))
    if isinstance(x, float) and floored != math.floor(x):
        raise NotSupportedError("INT of a number this close to a whole number")
    return sc.check_int(floored)


def _mod(ev: Evaluator, args: list[Any], ctx: Context) -> Value:
    n = _number(ev.scalar(args[0], ctx), "MOD")
    d = _number(ev.scalar(args[1], ctx), "MOD")
    if n is None:
        return None
    if d is None or d == 0:
        raise CalculationError("MOD: division by zero.")
    if isinstance(n, Decimal) or isinstance(d, Decimal):
        raise NotSupportedError("MOD of a Currency value")
    if isinstance(n, int) and isinstance(d, int):
        return n % d
    with localcontext(_DECIMALS):
        dn, dd = _dec(n), _dec(d)
        return float(dn - dd * (dn / dd).to_integral_value(rounding=ROUND_FLOOR))


def _mod_type(args: list[Any]) -> StaticType:
    kinds = set()
    for t in (infer_type(args[0]), infer_type(args[1])):
        kinds.add(DaxType.INTEGER if t == BLANK_TYPE else t)
    if kinds == {DaxType.INTEGER}:
        return DaxType.INTEGER
    if kinds <= {DaxType.INTEGER, DaxType.DECIMAL}:
        return DaxType.DECIMAL
    return None


def _power(ev: Evaluator, args: list[Any], ctx: Context) -> Value:
    x = _number(ev.scalar(args[0], ctx), "POWER")
    y = _number(ev.scalar(args[1], ctx), "POWER")
    if x is None:
        return None
    if y is None:
        raise NotSupportedError("POWER with a BLANK power")
    return sc.arithmetic("^", x, y)


def _sqrt(ev: Evaluator, args: list[Any], ctx: Context) -> Value:
    x = float(_required(ev.scalar(args[0], ctx), "SQRT"))
    if x < 0:
        raise CalculationError("SQRT of a negative number.")
    return math.sqrt(x)


def _exp(ev: Evaluator, args: list[Any], ctx: Context) -> Value:
    x = float(_required(ev.scalar(args[0], ctx), "EXP"))
    try:
        return math.exp(x)
    except OverflowError:
        raise NotSupportedError("EXP of a result too large for a number") from None


def _logarithm(name: str) -> Callable[..., Value]:
    def handler(ev: Evaluator, args: list[Any], ctx: Context) -> Value:
        x = float(_required(ev.scalar(args[0], ctx), name))
        base = 10.0
        if name == "LN":
            base = math.e
        elif name == "LOG" and _given(args, 1):
            base = float(_required(ev.scalar(args[1], ctx), "LOG"))
            if base <= 0 or base == 1:
                raise NotSupportedError("LOG with a base that is 1 or below 0")
        if x <= 0:
            raise CalculationError(f"{name} of a number that isn't above zero.")
        if base == 10.0:
            return math.log10(x)
        if name == "LN":
            return math.log(x)
        return math.log(x) / math.log(base)

    return handler


def _sign(ev: Evaluator, args: list[Any], ctx: Context) -> Value:
    x = _number(ev.scalar(args[0], ctx), "SIGN")
    if x is None:
        return None
    return (x > 0) - (x < 0)


def _multiple(name: str, mode: str) -> Callable[..., Value]:
    """CEILING, FLOOR and MROUND: round to a multiple of a significance."""

    def handler(ev: Evaluator, args: list[Any], ctx: Context) -> Value:
        x = _number(ev.scalar(args[0], ctx), name)
        step = _number(ev.scalar(args[1], ctx), name)
        if step is None or isinstance(step, Decimal):
            raise NotSupportedError(f"{name} with a BLANK or Currency significance")
        if step <= 0 and name != "MROUND":
            raise NotSupportedError(f"{name} with a significance that isn't above zero")
        if step == 0:
            raise NotSupportedError("MROUND to a multiple of zero")
        if x is None:
            return None
        if isinstance(x, Decimal) and (name == "MROUND" or not isinstance(step, int)):
            raise NotSupportedError(f"{name} of a Currency value")
        if name == "MROUND" and x != 0 and (x < 0) != (step < 0):
            raise NotSupportedError("MROUND with a number and multiple of different signs")
        with localcontext(_DECIMALS):
            ds = _dec(step)
            result = (_dec(x) / ds).to_integral_value(rounding=mode) * ds
        if isinstance(x, Decimal):
            return sc.to_currency(result)
        if isinstance(step, int):
            return sc.check_int(int(result))
        return float(result)

    return handler


def _multiple_type(args: list[Any]) -> StaticType:
    if infer_type(args[0]) is DaxType.CURRENCY:
        return DaxType.CURRENCY
    t = infer_type(args[1])
    return t if t in (DaxType.INTEGER, DaxType.DECIMAL) else None


def _quotient(ev: Evaluator, args: list[Any], ctx: Context) -> Value:
    n = _number(ev.scalar(args[0], ctx), "QUOTIENT")
    d = _number(ev.scalar(args[1], ctx), "QUOTIENT")
    if n is None:
        return None
    if d is None or d == 0:
        raise NotSupportedError("QUOTIENT by zero or BLANK")
    with localcontext(_DECIMALS):
        return sc.check_int(int((_dec(n) / _dec(d)).to_integral_value(rounding=ROUND_DOWN)))


# --- text -------------------------------------------------------------------------------


def _concatenate(ev: Evaluator, args: list[Any], ctx: Context) -> Value:
    return sc.concatenate(ev.scalar(args[0], ctx), ev.scalar(args[1], ctx))


def _count_arg(ev: Evaluator, args: list[Any], i: int, ctx: Context, fn: str, what: str) -> int:
    return _whole(ev.scalar(args[i], ctx), fn, what, 0)


def _left(ev: Evaluator, args: list[Any], ctx: Context) -> Value:
    text = _text(ev.scalar(args[0], ctx))
    n = _count_arg(ev, args, 1, ctx, "LEFT", "number of characters") if _given(args, 1) else 1
    return None if text is None else text[:n]


def _right(ev: Evaluator, args: list[Any], ctx: Context) -> Value:
    text = _text(ev.scalar(args[0], ctx))
    n = _count_arg(ev, args, 1, ctx, "RIGHT", "number of characters") if _given(args, 1) else 1
    if text is None:
        return None
    return text[len(text) - n :] if n else ""


def _mid(ev: Evaluator, args: list[Any], ctx: Context) -> Value:
    text = _text(ev.scalar(args[0], ctx))
    start = _whole(ev.scalar(args[1], ctx), "MID", "start position", 1)
    n = _count_arg(ev, args, 2, ctx, "MID", "number of characters")
    return None if text is None else text[start - 1 : start - 1 + n]


def _len(ev: Evaluator, args: list[Any], ctx: Context) -> Value:
    text = _text(ev.scalar(args[0], ctx))
    return None if text is None else len(text)


def _case(name: str) -> Callable[..., Value]:
    def handler(ev: Evaluator, args: list[Any], ctx: Context) -> Value:
        text = _text(ev.scalar(args[0], ctx))
        if text is None:
            return None
        if not text.isascii():
            raise NotSupportedError(f"{name} of text with characters outside ASCII")
        return text.upper() if name == "UPPER" else text.lower()

    return handler


def _trim(ev: Evaluator, args: list[Any], ctx: Context) -> Value:
    text = _text(ev.scalar(args[0], ctx))
    return None if text is None else re.sub(" +", " ", text.strip(" "))


def _needed_text(value: Value, fn: str, what: str) -> str:
    if value is None:
        raise NotSupportedError(f"{fn} with a BLANK {what}")
    return _format.to_text(value)


def _substitute(ev: Evaluator, args: list[Any], ctx: Context) -> Value:
    text = _text(ev.scalar(args[0], ctx))
    old = _needed_text(ev.scalar(args[1], ctx), "SUBSTITUTE", "old text")
    new = _needed_text(ev.scalar(args[2], ctx), "SUBSTITUTE", "new text")
    instance = (
        _whole(ev.scalar(args[3], ctx), "SUBSTITUTE", "instance", 1) if _given(args, 3) else None
    )
    if old == "":
        raise NotSupportedError("SUBSTITUTE of empty text")
    if text is None:
        return None
    if instance is None:
        return text.replace(old, new)
    pos = -1
    for _ in range(instance):
        pos = text.find(old, pos + 1 if pos < 0 else pos + len(old))
        if pos < 0:
            return text
    return text[:pos] + new + text[pos + len(old) :]


def _replace(ev: Evaluator, args: list[Any], ctx: Context) -> Value:
    text = _text(ev.scalar(args[0], ctx)) or ""
    start = _whole(ev.scalar(args[1], ctx), "REPLACE", "start position", 1)
    n = _count_arg(ev, args, 2, ctx, "REPLACE", "number of characters")
    new = _needed_text(ev.scalar(args[3], ctx), "REPLACE", "new text")
    return text[: start - 1] + new + text[start - 1 + n :]


def _wildcards(pattern: str) -> str:
    """A SEARCH pattern as a regular expression: ? is one character, * any, ~ escapes."""
    out = []
    i = 0
    while i < len(pattern):
        ch = pattern[i]
        if ch == "~" and i + 1 < len(pattern) and pattern[i + 1] in "?*~":
            out.append(re.escape(pattern[i + 1]))
            i += 2
            continue
        out.append("." if ch == "?" else ".*?" if ch == "*" else re.escape(ch))
        i += 1
    return "".join(out)


def _finder(name: str) -> Callable[..., Value]:
    def handler(ev: Evaluator, args: list[Any], ctx: Context) -> Value:
        find = _needed_text(ev.scalar(args[0], ctx), name, "text to find")
        within = _needed_text(ev.scalar(args[1], ctx), name, "text to search")
        start = 1
        if _given(args, 2):
            start = _whole(ev.scalar(args[2], ctx), name, "start position", 1)
        if start > len(within) + 1:
            raise NotSupportedError(f"{name} starting past the end of the text")
        if name == "SEARCH":
            match = re.compile(_wildcards(find), re.IGNORECASE | re.DOTALL).search(
                within, start - 1
            )
            position = None if match is None else match.start() + 1
        else:
            found = within.find(find, start - 1)
            position = None if found < 0 else found + 1
        if position is None:
            if not _given(args, 3):
                raise CalculationError(f"{name}: the text wasn't found.")
            value = ev.scalar(args[3], ctx)
        else:
            value = position
        return convert(value, infer_type(Call(name, args)))

    return handler


def _finder_type(args: list[Any]) -> StaticType:
    if _given(args, 3):
        return unify([DaxType.INTEGER, infer_type(args[3])])
    return DaxType.INTEGER


def _rept(ev: Evaluator, args: list[Any], ctx: Context) -> Value:
    text = _text(ev.scalar(args[0], ctx))
    n = _count_arg(ev, args, 1, ctx, "REPT", "number of times")
    return None if text is None else text * n


def _exact(ev: Evaluator, args: list[Any], ctx: Context) -> Value:
    a, b = ev.scalar(args[0], ctx), ev.scalar(args[1], ctx)
    return _format.to_text(a) == _format.to_text(b)


def _containsstring(ev: Evaluator, args: list[Any], ctx: Context) -> Value:
    found = ev.scalar(args[1], ctx)
    find = "" if found is None else _format.to_text(found)  # BLANK finds like ""
    unchecked = False
    if is_column(args[0]) and not find.endswith(" ") and not any(c in find for c in "?*~"):
        # The search ignores case and can't end in trailing spaces, so it finds the same
        # in every spelling Power BI might keep of a text value, unless the value is
        # only spaces (then "" could be kept, where nothing is found).
        raw = ev._column_value(args[0], ctx, check=False)
        unchecked = bool(find) or not (isinstance(raw, str) and raw.strip(" ") == "")
    within = _text(raw if unchecked else ev.scalar(args[0], ctx))
    if not within:  # Power BI: nothing is found in empty text, not even ""
        return False
    return re.search(_wildcards(find), within, re.IGNORECASE | re.DOTALL) is not None


def _containsstringexact(ev: Evaluator, args: list[Any], ctx: Context) -> Value:
    found = ev.scalar(args[1], ctx)
    find = "" if found is None else _format.to_text(found)
    if find == "":  # Power BI: empty or BLANK text is found in anything, even BLANK
        return True
    within = _needed_text(ev.scalar(args[0], ctx), "CONTAINSSTRINGEXACT", "text to search")
    if any(ch in find for ch in "?*~"):
        raise NotSupportedError("CONTAINSSTRINGEXACT with wildcard characters")
    return find in within


def _format_fn(ev: Evaluator, args: list[Any], ctx: Context) -> Value:
    value = ev.scalar(args[0], ctx)
    fmt = ev.scalar(args[1], ctx)
    if not isinstance(fmt, str):
        raise NotSupportedError("FORMAT with a format string that isn't text")
    locale = _arg(ev, args, 2, ctx)
    if locale is not None and not isinstance(locale, str):
        raise NotSupportedError("FORMAT with a locale that isn't text")
    return _format.format_value(value, fmt, locale)


# --- conversion -------------------------------------------------------------------------

_NUMBER_TEXT = re.compile(r"^(?:\$|-?)(?:\d{1,3}(?:,\d{3})+|\d+)?(?:\.\d+)?(?:[eE][+-]?\d+)?$")


def parse_number(text: str) -> float:
    """Text to a number, as VALUE converts it."""
    s = text.strip(" ")
    if "%" in s or not any(ch.isdigit() for ch in s):
        raise CalculationError(f"Cannot convert value '{text}' of type Text to type Number.")
    if not _NUMBER_TEXT.match(s) or not re.search(r"\d", s.split("e")[0].split("E")[0]):
        raise NotSupportedError(f"converting the text {text!r} to a number")
    return float(s.replace("$", "").replace(",", ""))


def _value(ev: Evaluator, args: list[Any], ctx: Context) -> Value:
    value = ev.scalar(args[0], ctx)
    if value is None:
        return None
    if isinstance(value, str):
        return parse_number(value)
    return float(_required(value, "VALUE"))


def _fixed(ev: Evaluator, args: list[Any], ctx: Context) -> Value:
    x = _number(ev.scalar(args[0], ctx), "FIXED")
    decimals = 2
    if _given(args, 1):
        decimals = _whole(ev.scalar(args[1], ctx), "FIXED", "number of decimals")
    commas = not sc.to_bool(_arg(ev, args, 2, ctx))
    if decimals > 15:
        raise NotSupportedError("FIXED with more than 15 decimals")
    number = _dec(x if x is not None else 0)
    with localcontext(_DECIMALS):
        rounded = number.scaleb(decimals).quantize(Decimal(1), rounding=ROUND_HALF_UP)
        rounded = rounded.scaleb(-decimals)
    if number < 0 and rounded == 0:
        raise NotSupportedError("FIXED of a negative number that rounds to zero")
    pattern = ("#,##0" if commas else "0") + ("." + "0" * decimals if decimals > 0 else "")
    return _format.format_value(rounded, pattern)


_CONVERT_TYPES = {
    "INTEGER": DaxType.INTEGER,
    "DOUBLE": DaxType.DECIMAL,
    "STRING": DaxType.STRING,
    "BOOLEAN": DaxType.BOOLEAN,
    "CURRENCY": DaxType.CURRENCY,
    "DATETIME": DaxType.DATETIME,
}


def _convert(ev: Evaluator, args: list[Any], ctx: Context) -> Value:
    value = ev.scalar(args[0], ctx)
    target = args[1].name if isinstance(args[1], Keyword) else str(args[1])
    if value is None:
        if target in ("INTEGER", "DOUBLE", "CURRENCY"):
            return None
        raise NotSupportedError(f"CONVERT of BLANK to {target}")
    if target == "STRING":
        return _format.to_text(value)
    if target == "DOUBLE" and isinstance(value, str):
        return parse_number(value)
    if target == "BOOLEAN" and isinstance(value, bool):
        return value
    if target == "DATETIME" and isinstance(value, dt.datetime):
        return value
    number = _required(value, f"CONVERT to {target}")
    if target == "INTEGER":
        exact = Decimal(number)  # the exact value, for detecting ties
        rounded = exact.to_integral_value(rounding=ROUND_HALF_UP)
        if isinstance(number, float) and rounded != _dec(number).to_integral_value(
            rounding=ROUND_HALF_UP
        ):
            raise NotSupportedError("CONVERT to INTEGER of a number this close to a half")
        return sc.check_int(int(rounded))
    if target == "DOUBLE":
        return float(number)
    if target == "BOOLEAN":
        return number != 0
    if target == "CURRENCY":
        return sc.to_currency(number)
    return sc.from_serial(float(number))


def _convert_type(args: list[Any]) -> StaticType:
    target = args[1].name if isinstance(args[1], Keyword) else None
    return _CONVERT_TYPES.get(target or "")


def _currency(ev: Evaluator, args: list[Any], ctx: Context) -> Value:
    value = _number(ev.scalar(args[0], ctx), "CURRENCY")
    return None if value is None else sc.to_currency(value)


# --- dates and times --------------------------------------------------------------------

_DATE_ERROR = (
    "An argument of function 'DATE' has the wrong data type or the result is too large or "
    "too small."
)


def _make_date(year: int, month: int, day: int, fn: str) -> dt.datetime:
    year += (month - 1) // 12
    month = (month - 1) % 12 + 1
    if not 1900 <= year <= 9999:
        raise NotSupportedError(f"{fn} with a year before 1900 or after 9999")
    try:
        return dt.datetime(year, month, 1) + dt.timedelta(days=day - 1)
    except OverflowError:
        raise NotSupportedError(f"{fn} outside the supported range of dates") from None


def _date_fn(ev: Evaluator, args: list[Any], ctx: Context) -> Value:
    parts = [ev.scalar(a, ctx) for a in args]
    year, month, day = (_whole(p, "DATE", "part") for p in parts)
    if month < 1 or day < 1:
        raise DaxError(_DATE_ERROR)
    return _make_date(year, month, day, "DATE")


def _time(ev: Evaluator, args: list[Any], ctx: Context) -> Value:
    h, m, s = (_whole(ev.scalar(a, ctx), "TIME", "part", 0) for a in args)
    return sc.BASE_DATE + dt.timedelta(seconds=(h * 3600 + m * 60 + s) % 86400)


_ISO_DATE = re.compile(r"^(\d{4})-(\d{1,2})-(\d{1,2})$")
_US_DATE = re.compile(r"^(\d{1,2})/(\d{1,2})/(\d{4})$")
_TIME_TEXT = re.compile(r"^(\d{1,2}):(\d{2})(?::(\d{2}))?(?:\s*(AM|PM|am|pm))?$")


def _datevalue(ev: Evaluator, args: list[Any], ctx: Context) -> Value:
    text = ev.scalar(args[0], ctx)
    if not isinstance(text, str):
        raise NotSupportedError(f"DATEVALUE of a {_kind(text)} value")
    s = text.strip()
    if m := _ISO_DATE.match(s):
        year, month, day = int(m[1]), int(m[2]), int(m[3])
    elif m := _US_DATE.match(s):
        month, day, year = int(m[1]), int(m[2]), int(m[3])
    else:
        raise NotSupportedError(f"DATEVALUE of the text {text!r} (only yyyy-mm-dd and m/d/yyyy)")
    try:
        return dt.datetime(year, month, day)
    except ValueError:
        raise NotSupportedError(f"DATEVALUE of the text {text!r}") from None


def _timevalue(ev: Evaluator, args: list[Any], ctx: Context) -> Value:
    text = ev.scalar(args[0], ctx)
    if not isinstance(text, str):
        raise NotSupportedError(f"TIMEVALUE of a {_kind(text)} value")
    m = _TIME_TEXT.match(text.strip())
    if m is None:
        raise NotSupportedError(f"TIMEVALUE of the text {text!r}")
    hour, minute, second = int(m[1]), int(m[2]), int(m[3] or 0)
    if m[4]:
        if not 1 <= hour <= 12:
            raise NotSupportedError(f"TIMEVALUE of the text {text!r}")
        hour = hour % 12 + (12 if m[4].upper() == "PM" else 0)
    if hour > 23 or minute > 59 or second > 59:
        raise NotSupportedError(f"TIMEVALUE of the text {text!r}")
    return sc.BASE_DATE + dt.timedelta(hours=hour, minutes=minute, seconds=second)


def _part(name: str, get: Callable[[dt.datetime], int]) -> Callable[..., Value]:
    def handler(ev: Evaluator, args: list[Any], ctx: Context) -> Value:
        value = _date(ev.scalar(args[0], ctx), name)
        return None if value is None else get(value)

    return handler


def _weekday(ev: Evaluator, args: list[Any], ctx: Context) -> Value:
    value = _date(ev.scalar(args[0], ctx), "WEEKDAY")
    kind = _whole(ev.scalar(args[1], ctx), "WEEKDAY", "return type") if _given(args, 1) else 1
    if kind not in (1, 2, 3):
        raise NotSupportedError(f"WEEKDAY with return type {kind}")
    if value is None:
        return None
    monday0 = value.weekday()
    return {1: (monday0 + 1) % 7 + 1, 2: monday0 + 1, 3: monday0}[kind]


def _weeknum(ev: Evaluator, args: list[Any], ctx: Context) -> Value:
    value = _date(ev.scalar(args[0], ctx), "WEEKNUM")
    kind = _whole(ev.scalar(args[1], ctx), "WEEKNUM", "return type") if _given(args, 1) else 1
    if kind not in (1, 2, 21):
        raise NotSupportedError(f"WEEKNUM with return type {kind}")
    if value is None:
        return None
    if kind == 21:
        return value.isocalendar().week
    jan1 = dt.date(value.year, 1, 1).weekday()  # Monday is 0
    offset = (jan1 + 1) % 7 if kind == 1 else jan1
    return (value.timetuple().tm_yday - 1 + offset) // 7 + 1


def _months(
    ev: Evaluator, args: list[Any], ctx: Context, fn: str
) -> tuple[dt.datetime | None, int]:
    start = _date(ev.scalar(args[0], ctx), fn)
    months = _whole(ev.scalar(args[1], ctx), fn, "number of months")
    return start, months


def _shift(start: dt.datetime, months: int, fn: str) -> tuple[int, int]:
    index = start.year * 12 + start.month - 1 + months
    year, month = divmod(index, 12)
    if not 1900 <= year <= 9999:
        raise NotSupportedError(f"{fn} outside the supported range of dates")
    return year, month + 1


def _edate(ev: Evaluator, args: list[Any], ctx: Context) -> Value:
    start, months = _months(ev, args, ctx, "EDATE")
    if start is None:
        return None
    year, month = _shift(start, months, "EDATE")
    return dt.datetime(year, month, min(start.day, calendar.monthrange(year, month)[1]))


def _eomonth(ev: Evaluator, args: list[Any], ctx: Context) -> Value:
    start, months = _months(ev, args, ctx, "EOMONTH")
    if start is None:
        return None
    year, month = _shift(start, months, "EOMONTH")
    return dt.datetime(year, month, calendar.monthrange(year, month)[1])


def _datediff(ev: Evaluator, args: list[Any], ctx: Context) -> Value:
    a = _date(ev.scalar(args[0], ctx), "DATEDIFF")
    b = _date(ev.scalar(args[1], ctx), "DATEDIFF")
    interval = args[2].name
    if a is None or b is None:
        return None
    return _boundary(b, interval) - _boundary(a, interval)


def _boundary(value: dt.datetime, interval: str) -> int:
    """The number of interval boundaries from 30 December 1899 to the value."""
    delta = value - sc.BASE_DATE
    seconds = delta.days * 86400 + delta.seconds
    if interval == "SECOND":
        return seconds
    if interval == "MINUTE":
        return seconds // 60
    if interval == "HOUR":
        return seconds // 3600
    if interval == "DAY":
        return delta.days
    if interval == "WEEK":
        return (delta.days + 6) // 7  # weeks start on Sunday; 30 December 1899 is a Saturday
    if interval == "MONTH":
        return value.year * 12 + value.month
    if interval == "QUARTER":
        return value.year * 4 + (value.month - 1) // 3
    return value.year


def _today(ev: Evaluator, args: list[Any], ctx: Context) -> Value:
    today = dt.date.today()
    return dt.datetime(today.year, today.month, today.day)


def _now(ev: Evaluator, args: list[Any], ctx: Context) -> Value:
    return dt.datetime.now().replace(microsecond=0)


# --- INFO.VIEW functions ------------------------------------------------------------------

S, N, B = DaxType.STRING, DaxType.INTEGER, DaxType.BOOLEAN


def _info_frame(columns: list[tuple[str, DaxType]], rows: list[list[Value]]) -> Frame:
    return Frame([ResultColumn(f"[{n}]", t) for n, t in columns], rows, filterable=False)


def _info_tables(model: Model) -> Frame:
    columns = [
        ("ID", N), ("Name", S), ("Model", S), ("DataCategory", S), ("Description", S),
        ("IsHidden", B), ("StorageMode", S), ("TableStorage", S), ("Expression", S),
        ("ShowAsVariationOnly", B), ("IsPrivate", B), ("CalculationGroupPrecedence", N),
        ("LineageTag", S),
    ]  # fmt: skip
    rows = [
        [
            i,
            t.name,
            None,
            "Time" if t.is_date_table else "Regular",
            t.description,
            t.is_hidden,
            "Import",
            None,
            None,
            False,
            False,
            None,
            None,
        ]
        for i, t in enumerate(model.tables, 1)
    ]
    return _info_frame(columns, rows)


_SUMMARIZE = {
    SummarizeBy.NONE: "None", SummarizeBy.SUM: "Sum", SummarizeBy.MIN: "Min",
    SummarizeBy.MAX: "Max", SummarizeBy.COUNT: "Count", SummarizeBy.AVERAGE: "Average",
    SummarizeBy.DISTINCT_COUNT: "DistinctCount", SummarizeBy.DEFAULT: "Default",
}  # fmt: skip


def _info_columns(model: Model) -> Frame:
    columns = [
        ("ID", N), ("Name", S), ("Table", S), ("DataType", S), ("DataCategory", S),
        ("Description", S), ("IsHidden", B), ("IsUnique", B), ("IsKey", B), ("IsNullable", B),
        ("Alignment", S), ("SummarizeBy", S), ("ColumnStorage", S), ("Type", S),
        ("SourceColumn", S), ("Expression", S), ("FormatString", S), ("IsAvailableInMDX", B),
        ("SortByColumn", S), ("GroupingBehavior", S), ("SourceProviderType", S),
        ("DisplayFolder", S), ("AlternateOf", S), ("LineageTag", S),
    ]  # fmt: skip
    rows: list[list[Value]] = []
    for t in model.tables:
        for c in t.columns:
            default = SummarizeBy.SUM if c.data_type.is_numeric else SummarizeBy.NONE
            calculated = c.is_calculated
            rows.append(
                [
                    len(rows) + 1,
                    c.name,
                    t.name,
                    sc.type_label(c.data_type),
                    c.data_category or "Regular",
                    c.description,
                    c.is_hidden,
                    False,
                    c.is_key,
                    c.is_nullable,
                    "Default",
                    "Default" if c.summarize_by == default else _SUMMARIZE[c.summarize_by],
                    None,
                    "Calculated" if calculated else "Data",
                    None if calculated else c.name,
                    c.expression.to_dax() if calculated else None,
                    c.format_string,
                    True,
                    c.sort_by_column.name if c.sort_by_column is not None else None,
                    "GroupOnValue",
                    None,
                    c.display_folder,
                    None,
                    None,
                ]
            )
    return _info_frame(columns, rows)


def _info_measures(model: Model) -> Frame:
    columns = [
        ("ID", N), ("Name", S), ("Table", S), ("Description", S), ("DataType", S),
        ("Expression", S), ("FormatString", S), ("IsHidden", B), ("State", S), ("KPIID", N),
        ("IsSimpleMeasure", B), ("DisplayFolder", S), ("DetailRowsDefinition", S),
        ("DataCategory", S), ("FormatStringDefinition", S), ("LineageTag", S),
    ]  # fmt: skip
    rows: list[list[Value]] = []
    for i, m in enumerate(model.measures, 1):
        t = infer_type(m.expression)
        label = sc.type_label(t) if isinstance(t, DaxType) else None
        rows.append(
            [
                i,
                m.name,
                m.table.name,
                m.description,
                label,
                m.expression.to_dax(),
                m.format_string,
                m.is_hidden,
                "Valid",
                None,
                False,
                m.display_folder,
                None,
                m.data_category,
                None,
                None,
            ]
        )
    return _info_frame(columns, rows)


def _info_relationships(model: Model) -> Frame:
    columns = [
        ("ID", N), ("Name", S), ("Relationship", S), ("Model", S), ("IsActive", B),
        ("CrossFilteringBehavior", S), ("RelyOnReferentialIntegrity", B), ("FromTable", S),
        ("FromColumn", S), ("FromCardinality", S), ("ToTable", S), ("ToColumn", S),
        ("ToCardinality", S), ("State", S), ("SecurityFilteringBehavior", S),
    ]  # fmt: skip
    rows: list[list[Value]] = []
    for i, r in enumerate(model.relationships, 1):
        from_card, to_card = {
            Cardinality.MANY_TO_ONE: ("Many", "One"),
            Cardinality.ONE_TO_ONE: ("One", "One"),
            Cardinality.MANY_TO_MANY: ("Many", "Many"),
        }[r.cardinality]
        both = r.cross_filter is CrossFilter.BOTH
        symbol = {"Many": "*", "One": "1"}
        text = (
            f"'{r.from_column.table.name}'[{r.from_column.name}] {symbol[from_card]}"
            f"[{'<->' if both else '<-'}]{symbol[to_card]} "
            f"'{r.to_column.table.name}'[{r.to_column.name}]"
        )
        rows.append(
            [
                i,
                r.name,
                text,
                None,
                r.is_active,
                "BothDirections" if both else "OneDirection",
                r.rely_on_referential_integrity,
                r.from_column.table.name,
                r.from_column.name,
                from_card,
                r.to_column.table.name,
                r.to_column.name,
                to_card,
                "Ready",
                "Single",
            ]
        )
    return _info_frame(columns, rows)


def _info(build: Callable[[Model], Frame]) -> Callable[..., TableValue]:
    def handler(ev: Evaluator, args: list[Any], ctx: Context) -> TableValue:
        return build(ev.model)

    return handler


# --- registration -------------------------------------------------------------------------

SCALAR: dict[str, Callable[..., Value]] = {
    "DIVIDE": _divide,
    "SWITCH": _switch,
    "COALESCE": _coalesce,
    "IFERROR": _iferror,
    "ISERROR": _iserror,
    "ISBLANK": _isblank,
    "ISNUMBER": _isnumber,
    "ISTEXT": _istext,
    "ISLOGICAL": _islogical,
    "ISEMPTY": _isempty,
    "CONTAINS": _contains,
    "CONTAINSROW": _containsrow,
    "ABS": _abs,
    "ROUND": _rounder("ROUND", ROUND_HALF_UP),
    "ROUNDUP": _rounder("ROUNDUP", ROUND_UP),
    "ROUNDDOWN": _rounder("ROUNDDOWN", ROUND_DOWN),
    "TRUNC": _rounder("TRUNC", ROUND_DOWN, 0),
    "INT": _int,
    "MOD": _mod,
    "POWER": _power,
    "SQRT": _sqrt,
    "EXP": _exp,
    "LN": _logarithm("LN"),
    "LOG": _logarithm("LOG"),
    "LOG10": _logarithm("LOG10"),
    "SIGN": _sign,
    "CEILING": _multiple("CEILING", ROUND_CEILING),
    "FLOOR": _multiple("FLOOR", ROUND_FLOOR),
    "MROUND": _multiple("MROUND", ROUND_HALF_UP),
    "QUOTIENT": _quotient,
    "CONCATENATE": _concatenate,
    "LEFT": _left,
    "RIGHT": _right,
    "MID": _mid,
    "LEN": _len,
    "UPPER": _case("UPPER"),
    "LOWER": _case("LOWER"),
    "TRIM": _trim,
    "SUBSTITUTE": _substitute,
    "REPLACE": _replace,
    "SEARCH": _finder("SEARCH"),
    "FIND": _finder("FIND"),
    "REPT": _rept,
    "EXACT": _exact,
    "CONTAINSSTRING": _containsstring,
    "CONTAINSSTRINGEXACT": _containsstringexact,
    "FORMAT": _format_fn,
    "VALUE": _value,
    "FIXED": _fixed,
    "CONVERT": _convert,
    "CURRENCY": _currency,
    "DATE": _date_fn,
    "TIME": _time,
    "DATEVALUE": _datevalue,
    "TIMEVALUE": _timevalue,
    "YEAR": _part("YEAR", lambda d: d.year),
    "MONTH": _part("MONTH", lambda d: d.month),
    "DAY": _part("DAY", lambda d: d.day),
    "HOUR": _part("HOUR", lambda d: d.hour),
    "MINUTE": _part("MINUTE", lambda d: d.minute),
    "SECOND": _part("SECOND", lambda d: d.second),
    "QUARTER": _part("QUARTER", lambda d: (d.month - 1) // 3 + 1),
    "WEEKDAY": _weekday,
    "WEEKNUM": _weeknum,
    "TODAY": _today,
    "NOW": _now,
    "EDATE": _edate,
    "EOMONTH": _eomonth,
    "DATEDIFF": _datediff,
}

TABLES: dict[str, Callable[..., TableValue]] = {
    "INFO.VIEW.TABLES": _info(_info_tables),
    "INFO.VIEW.COLUMNS": _info(_info_columns),
    "INFO.VIEW.MEASURES": _info(_info_measures),
    "INFO.VIEW.RELATIONSHIPS": _info(_info_relationships),
}

TYPES: dict[str, Callable[[list[Any]], StaticType]] = {
    "DIVIDE": _divide_type,
    "SWITCH": _switch_type,
    "COALESCE": lambda args: unify([infer_type(a) for a in args]),
    "IFERROR": lambda args: unify([infer_type(args[0]), infer_type(args[1])]),
    "ABS": _same_type,
    "ROUND": _same_type,
    "ROUNDUP": _same_type,
    "ROUNDDOWN": _same_type,
    "TRUNC": _same_type,
    "MOD": _mod_type,
    "CEILING": _multiple_type,
    "FLOOR": _multiple_type,
    "MROUND": _multiple_type,
    "SEARCH": _finder_type,
    "FIND": _finder_type,
    "CONVERT": _convert_type,
}
