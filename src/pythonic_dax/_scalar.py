"""Scalar values and operators, with DAX's rules for BLANK, types and conversions.

Values are plain Python objects: ``None`` is BLANK, ``int`` is a whole number,
``float`` a decimal number, ``Decimal`` a fixed decimal (currency), ``datetime`` a
date/time, plus ``str`` and ``bool``.

The rules here were checked against Power BI Desktop (see the golden cases under
``tests/golden/cases/operators``). Where Power BI's answer isn't known yet, these
functions raise :class:`NotSupportedError` instead of guessing.
"""

from __future__ import annotations

import datetime as dt
import math
import re
from decimal import Decimal
from typing import Any

from .errors import DaxError, NotSupportedError
from .types import DaxType

Value = Any  # None | bool | int | float | Decimal | str | dt.datetime

BASE_DATE = dt.datetime(1899, 12, 30)
_DAY = dt.timedelta(days=1)
_INT64_MIN, _INT64_MAX = -(2**63), 2**63 - 1
_CURRENCY_STEP = Decimal("0.0001")
_POWER_ERROR = (
    "An argument of function 'POWER' has the wrong data type or the result is too large or "
    "too small."
)
_ORDERABLE_TEXT = re.compile(r"^[A-Za-z0-9 ]*$")


def type_of(value: Value) -> DaxType | None:
    """The DAX type of a value, or None for BLANK."""
    if value is None:
        return None
    if isinstance(value, bool):
        return DaxType.BOOLEAN
    if isinstance(value, int):
        return DaxType.INTEGER
    if isinstance(value, float):
        return DaxType.DECIMAL
    if isinstance(value, Decimal):
        return DaxType.CURRENCY
    if isinstance(value, str):
        return DaxType.STRING
    if isinstance(value, dt.datetime):
        return DaxType.DATETIME
    raise TypeError(f"not a DAX value: {value!r}")


def type_label(dax_type: DaxType | None) -> str:
    """The type name Power BI uses in error messages."""
    return {
        None: "Blank",
        DaxType.INTEGER: "Integer",
        DaxType.DECIMAL: "Number",
        DaxType.CURRENCY: "Currency",
        DaxType.STRING: "Text",
        DaxType.BOOLEAN: "True/False",
        DaxType.DATETIME: "Date",
    }[dax_type]


# --- conversions ------------------------------------------------------------------


def to_serial(value: dt.datetime) -> float:
    """A date/time as DAX's number of days since 30 December 1899."""
    return (value - BASE_DATE) / _DAY


def from_serial(days: float) -> dt.datetime:
    if math.isnan(days) or math.isinf(days):
        raise NotSupportedError("a date/time that is not a finite number")
    try:
        return BASE_DATE + dt.timedelta(microseconds=round(days * 86_400_000_000))
    except OverflowError:
        raise NotSupportedError("a date/time outside the supported range") from None


def to_currency(number: float | int | Decimal) -> Decimal:
    """Round to a fixed decimal with four places, as Power BI does."""
    if isinstance(number, Decimal):
        exact = number
    else:
        if isinstance(number, float) and (math.isnan(number) or math.isinf(number)):
            raise NotSupportedError("a fixed decimal result that is not a finite number")
        exact = Decimal(number)
    scaled = exact / _CURRENCY_STEP
    fraction = abs(scaled - scaled.to_integral_value(rounding="ROUND_FLOOR"))
    if isinstance(number, float) and abs(fraction - Decimal("0.5")) < Decimal("1e-6"):
        # Power BI rounds half away from zero, but on the double it computed, whose last
        # bits we can't be sure we reproduce. Don't guess at a tie.
        raise NotSupportedError("rounding a fixed decimal result that falls on a tie")
    return exact.quantize(_CURRENCY_STEP, rounding="ROUND_HALF_UP")


def check_int(value: int) -> int:
    if not _INT64_MIN <= value <= _INT64_MAX:
        raise NotSupportedError("whole-number arithmetic that overflows 64 bits")
    return value


def _number(value: Value, op: str) -> int | float | Decimal | dt.datetime:
    """An operand of an arithmetic operator, after DAX's implicit conversions."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int | float | Decimal | dt.datetime):
        return value
    if isinstance(value, str):
        raise NotSupportedError(f"converting text to a number for the {op} operator", phase=6)
    raise TypeError(f"not a DAX value: {value!r}")


def to_bool(value: Value) -> bool:
    """DAX's conversion to TRUE/FALSE: BLANK is FALSE, numbers are TRUE unless 0."""
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float | Decimal):
        if isinstance(value, float) and math.isnan(value):
            raise NotSupportedError("converting NaN to TRUE/FALSE")
        return value != 0
    if isinstance(value, str):
        raise DaxError(f"Cannot convert value '{value}' of type Text to type True/False.")
    raise NotSupportedError("converting a date/time to TRUE/FALSE")


# --- arithmetic ----------------------------------------------------------------------


def arithmetic(op: str, a: Value, b: Value) -> Value:
    """``a op b`` for ``+ - * / ^``."""
    if op == "^":
        return _power(a, b)
    if op in ("+", "-"):
        if a is None and b is None:
            return None
        x = _number(a, op) if a is not None else None
        y = _number(b, op) if b is not None else None
        if x is None:
            x = _zero_like(y)
        if y is None:
            y = _zero_like(x)
        return _add(op, x, y)
    if op == "*":
        if a is None or b is None:
            return None
        return _multiply(_number(a, op), _number(b, op))
    if op == "/":
        if a is None:
            return None
        x = _number(a, op)
        y = _number(b, op) if b is not None else 0
        return _divide(x, y)
    raise ValueError(op)


def _zero_like(value: Any) -> Any:
    return 0


def _kind(value: Any) -> str:
    if isinstance(value, dt.datetime):
        return "date"
    if isinstance(value, Decimal):
        return "cur"
    if isinstance(value, float):
        return "dec"
    return "int"


def _as_float(value: Any) -> float:
    if isinstance(value, dt.datetime):
        return to_serial(value)
    return float(value)


def _add(op: str, x: Any, y: Any) -> Value:
    kx, ky = _kind(x), _kind(y)
    if "date" in (kx, ky):
        a, b = _as_float(x), _as_float(y)
        return from_serial(a + b if op == "+" else a - b)
    if kx == ky == "int":
        return check_int(x + y if op == "+" else x - y)
    if "dec" not in (kx, ky):  # currency with currency or a whole number
        da, db = Decimal(x), Decimal(y)
        return to_currency(da + db if op == "+" else da - db)
    a, b = float(x), float(y)
    return a + b if op == "+" else a - b


def _multiply(x: Any, y: Any) -> Value:
    kx, ky = _kind(x), _kind(y)
    if "date" in (kx, ky):
        return _as_float(x) * _as_float(y)
    if kx == ky == "int":
        return check_int(x * y)
    if kx == ky == "cur":
        return float(x) * float(y)
    if "cur" in (kx, ky):
        if "dec" in (kx, ky):
            return to_currency(float(x) * float(y))
        return to_currency(Decimal(x) * Decimal(y))
    return float(x) * float(y)


def _divide(x: Any, y: Any) -> Value:
    kx, ky = _kind(x), _kind(y)
    if kx == "cur" and ky in ("int", "dec"):
        if y == 0:
            raise NotSupportedError("dividing a fixed decimal number by zero")
        return to_currency(float(x) / float(y))
    a, b = _as_float(x), _as_float(y)
    if b == 0:
        if a == 0 or math.isnan(a):
            return math.nan
        return math.copysign(math.inf, a)
    return a / b


def _power(a: Value, b: Value) -> Value:
    if a is None and b is None:
        raise DaxError(_POWER_ERROR)
    if a is None:
        return None
    x = _as_float(_number(a, "^"))
    y = _as_float(_number(b, "^")) if b is not None else 0.0
    if x == 0 and y <= 0:
        raise DaxError(_POWER_ERROR)
    if x < 0 and not y.is_integer():
        raise DaxError(_POWER_ERROR)
    try:
        result = math.pow(x, y)
    except OverflowError:
        raise DaxError(_POWER_ERROR) from None
    if math.isinf(result):
        raise DaxError(_POWER_ERROR)
    return result


def negate(value: Value) -> Value:
    """Unary minus. ``-BLANK()`` is BLANK."""
    if value is None:
        return None
    x = _number(value, "-")
    if isinstance(x, dt.datetime):
        raise NotSupportedError("negating a date/time")
    if isinstance(x, int):
        return check_int(-x)
    return -x


# --- text --------------------------------------------------------------------------


def concatenate(a: Value, b: Value) -> Value:
    """``a & b``. BLANK & BLANK is BLANK; otherwise BLANK becomes empty text."""
    if a is None and b is None:
        return None
    return _text(a) + _text(b)


def _text(value: Value) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    raise NotSupportedError(f"converting a {type_label(type_of(value))} value to text", phase=6)


# --- comparisons -------------------------------------------------------------------


def compare(op: str, a: Value, b: Value) -> bool:
    """``= <> < <= > >=``, and ``==`` (strict: BLANK only equals BLANK)."""
    if op == "==":
        if a is None or b is None:
            return a is None and b is None
        return _order(a, b, "==") == 0
    if a is None and b is None:
        result = 0
    else:
        if a is None:
            a = _blank_as(b)
        if b is None:
            b = _blank_as(a)
        result = _order(a, b, op)
    return {
        "=": result == 0,
        "<>": result != 0,
        "<": result < 0,
        "<=": result <= 0,
        ">": result > 0,
        ">=": result >= 0,
    }[op]


def _blank_as(other: Value) -> Value:
    """What BLANK compares as next to a value of another type."""
    if isinstance(other, str):
        return ""
    if isinstance(other, bool):
        return False
    if isinstance(other, dt.datetime):
        return BASE_DATE
    return 0


def _order(a: Value, b: Value, op: str) -> int:
    ka, kb = _compare_kind(a), _compare_kind(b)
    if ka != kb:
        if "text" in (ka, kb):
            ta = type_label(type_of(a))
            tb = type_label(type_of(b))
            raise DaxError(
                f"DAX comparison operations do not support comparing values of type {ta} "
                f"with values of type {tb}. Consider using the VALUE or FORMAT function to "
                "convert one of the values."
            )
        raise NotSupportedError(
            f"comparing a {type_label(type_of(a))} value with a {type_label(type_of(b))} value"
        )
    if ka == "text":
        x, y = a.lower(), b.lower()
        if op not in ("=", "<>", "==") and not (
            _ORDERABLE_TEXT.match(a) and _ORDERABLE_TEXT.match(b)
        ):
            raise NotSupportedError(
                "ordering text that isn't plain letters, digits and spaces (Power BI sorts "
                "text by the model's collation)"
            )
        return int(x > y) - int(x < y)
    if ka == "number":
        x, y = float(a), float(b)
        if isinstance(a, int) and isinstance(b, int):
            return int(a > b) - int(a < b)
        if math.isnan(x) or math.isnan(y):
            raise NotSupportedError("comparing NaN")
        return (x > y) - (x < y)
    return int(a > b) - int(a < b)


def _compare_kind(value: Value) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int | float | Decimal):
        return "number"
    if isinstance(value, str):
        return "text"
    return "date"


def scalar_min_max(name: str, a: Value, b: Value) -> Value:
    """Two-argument MIN and MAX. BLANK compares as 0, and is returned if it wins."""
    if a is None and b is None:
        return None
    for v in (a, b):
        if v is None:
            continue
        if isinstance(v, bool) or not isinstance(v, int | float | Decimal | dt.datetime):
            raise NotSupportedError(f"{name} of {type_label(type_of(v))} values")
    x = a if a is not None else _blank_as(b)
    y = b if b is not None else _blank_as(a)
    order = _order(x, y, "<")
    if order == 0:
        if (a is None) != (b is None):
            raise NotSupportedError(f"{name} of BLANK and zero")
        return a
    first = order < 0 if name == "MIN" else order > 0
    winner = a if first else b
    other = b if first else a
    if (isinstance(winner, int) and isinstance(other, float | Decimal)) or (
        isinstance(winner, Decimal) and isinstance(other, float)
    ):
        return float(winner)
    return winner
