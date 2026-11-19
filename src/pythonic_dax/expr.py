"""Expression objects: the Python form of DAX expressions.

Calling a DAX function such as ``SUM(Sales["Amount"])`` doesn't compute anything. It
builds an expression tree that the model evaluates later in a filter context, exactly
as DAX does. Every expression can print itself as DAX with :meth:`Expr.to_dax`.

Python operators map to DAX operators:

=========================  ===================
Python                     DAX
=========================  ===================
``+ - * /``, ``**``        ``+ - * /``, ``^``
``==``, ``!=``             ``=``, ``<>``
``< <= > >=``              ``< <= > >=``
``&``, ``|``, ``~``        ``&&``, ``||``, ``NOT``
``x.IN([1, 2])``           ``x IN {1, 2}``
``x.concat(y)``            ``x & y``
=========================  ===================

Python's ``and``, ``or``, ``not``, ``if`` and ``in`` can't be redefined. Using an
expression where Python needs ``True`` or ``False`` raises a ``TypeError`` that points
to the DAX way of writing it.
"""

from __future__ import annotations

import datetime as dt
import math
import re
from collections.abc import Callable, Iterable, Iterator, Sequence
from decimal import Decimal
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from .types import DaxType

if TYPE_CHECKING:
    from .objects import Column, Measure

Kind = Literal["scalar", "table", "modifier"]

# Operator precedence, highest binds tightest (from the DAX operator reference).
PREC_ATOM = 100
PREC_POWER = 90
PREC_SIGN = 80
PREC_MUL = 70
PREC_ADD = 60
PREC_CONCAT = 50
PREC_COMPARE = 40
PREC_AND = 20
PREC_OR = 10

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class _ExprBase:
    """Behavior shared by scalar and table expressions."""

    kind: ClassVar[Kind] = "scalar"

    # Explicit, because defining __eq__ below would otherwise remove hashing.
    __hash__ = object.__hash__

    def children(self) -> Iterable[Node]:
        """Sub-expressions, for walking the tree."""
        return ()

    def _dax(self) -> tuple[str, int]:
        """DAX text and its precedence."""
        raise NotImplementedError

    def to_dax(self) -> str:
        """The equivalent DAX expression, ready to paste into Power BI."""
        return self._dax()[0]

    @property
    def dax_type(self) -> DaxType | None:
        """The DAX type this expression returns, when it is known before evaluation."""
        return None

    def walk(self) -> Iterator[Node]:
        """This node and every node below it."""
        yield self  # type: ignore[misc]
        for child in self.children():
            yield from child.walk()

    def __bool__(self) -> bool:
        raise TypeError(
            "A DAX expression has no True/False value in Python. Use IF() instead of "
            "`if`, & and | instead of `and` and `or`, ~ instead of `not`, and .IN() "
            "instead of `in`. Comparisons combined with & or | need parentheses: "
            "(a == 1) & (b > 2)."
        )

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.to_dax()}>"


class Expr(_ExprBase):
    """A scalar DAX expression. Supports DAX operators through Python operators."""

    kind: ClassVar[Kind] = "scalar"

    # --- arithmetic ---------------------------------------------------------

    def __add__(self, other: Any) -> Expr:
        return BinaryOp("+", self, coerce(other))

    def __radd__(self, other: Any) -> Expr:
        return BinaryOp("+", coerce(other), self)

    def __sub__(self, other: Any) -> Expr:
        return BinaryOp("-", self, coerce(other))

    def __rsub__(self, other: Any) -> Expr:
        return BinaryOp("-", coerce(other), self)

    def __mul__(self, other: Any) -> Expr:
        return BinaryOp("*", self, coerce(other))

    def __rmul__(self, other: Any) -> Expr:
        return BinaryOp("*", coerce(other), self)

    def __truediv__(self, other: Any) -> Expr:
        return BinaryOp("/", self, coerce(other))

    def __rtruediv__(self, other: Any) -> Expr:
        return BinaryOp("/", coerce(other), self)

    def __pow__(self, other: Any) -> Expr:
        return BinaryOp("^", self, coerce(other))

    def __rpow__(self, other: Any) -> Expr:
        return BinaryOp("^", coerce(other), self)

    def __neg__(self) -> Expr:
        return UnaryOp("-", self)

    def __pos__(self) -> Expr:
        return UnaryOp("+", self)

    def __floordiv__(self, other: Any) -> Expr:
        raise TypeError("DAX has no // operator. Use QUOTIENT(a, b) or INT(a / b).")

    def __mod__(self, other: Any) -> Expr:
        raise TypeError("DAX has no % operator. Use MOD(a, b).")

    # --- comparison ---------------------------------------------------------

    def __eq__(self, other: Any) -> Expr:  # type: ignore[override]
        return BinaryOp("=", self, coerce(other))

    def __ne__(self, other: Any) -> Expr:  # type: ignore[override]
        return BinaryOp("<>", self, coerce(other))

    def __lt__(self, other: Any) -> Expr:
        return BinaryOp("<", self, coerce(other))

    def __le__(self, other: Any) -> Expr:
        return BinaryOp("<=", self, coerce(other))

    def __gt__(self, other: Any) -> Expr:
        return BinaryOp(">", self, coerce(other))

    def __ge__(self, other: Any) -> Expr:
        return BinaryOp(">=", self, coerce(other))

    def strict_eq(self, other: Any) -> Expr:
        """DAX ``==``: equal without converting BLANK (``BLANK() == 0`` is FALSE)."""
        return BinaryOp("==", self, coerce(other))

    def IN(self, values: Any) -> Expr:
        """DAX ``IN``: ``Product["Color"].IN(["Red", "Blue"])``."""
        return BinaryOp("IN", self, coerce_table(values, "IN"))

    # --- logic and text -----------------------------------------------------

    def __and__(self, other: Any) -> Expr:
        return BinaryOp("&&", self, coerce(other))

    def __rand__(self, other: Any) -> Expr:
        return BinaryOp("&&", coerce(other), self)

    def __or__(self, other: Any) -> Expr:
        return BinaryOp("||", self, coerce(other))

    def __ror__(self, other: Any) -> Expr:
        return BinaryOp("||", coerce(other), self)

    def __invert__(self) -> Expr:
        return Call("NOT", [self], DaxType.BOOLEAN)

    def concat(self, other: Any) -> Expr:
        """DAX ``&`` (text concatenation): ``Customer["First"].concat(" ").concat(...)``."""
        return BinaryOp("&", self, coerce(other))


class TableExpr(_ExprBase):
    """An expression that returns a table, such as ``FILTER(...)`` or a model table."""

    kind: ClassVar[Kind] = "table"

    def __getitem__(self, name: str) -> ColumnName:
        """A column this table expression adds, by name: DAX ``[name]``.

        ``t = ADDCOLUMNS(Product, {"Sales": total})`` then ``FILTER(t, t["Sales"] > 10)``.
        """
        if not isinstance(name, str):
            raise TypeError(f"column names are text, got {name!r}")
        return ColumnName(name)


class ModifierExpr(_ExprBase):
    """A CALCULATE modifier such as ``USERELATIONSHIP`` or ``REMOVEFILTERS``.

    Modifiers are only valid as filter arguments of CALCULATE and CALCULATETABLE.
    """

    kind: ClassVar[Kind] = "modifier"


Node = Expr | TableExpr | ModifierExpr


# --- leaves -------------------------------------------------------------------


class Literal_(Expr):
    """A constant: a number, text, TRUE/FALSE, BLANK, or a date."""

    def __init__(self, value: Any) -> None:
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            raise ValueError("DAX has no literal for NaN or infinity; write it as 0/0 or 1/0")
        self.value = value

    @property
    def dax_type(self) -> DaxType | None:
        v = self.value
        if v is None:
            return None
        if isinstance(v, bool):
            return DaxType.BOOLEAN
        if isinstance(v, int):
            return DaxType.INTEGER
        if isinstance(v, float | Decimal):
            return DaxType.DECIMAL
        if isinstance(v, str):
            return DaxType.STRING
        if isinstance(v, dt.date):
            return DaxType.DATETIME
        return None

    def _dax(self) -> tuple[str, int]:
        v = self.value
        if v is None:
            return "BLANK()", PREC_ATOM
        if v is True:
            return "TRUE()", PREC_ATOM
        if v is False:
            return "FALSE()", PREC_ATOM
        if isinstance(v, int):
            return (str(v), PREC_ATOM) if v >= 0 else (str(v), PREC_SIGN)
        if isinstance(v, float):
            text = repr(v)
            return (text, PREC_ATOM) if v >= 0 else (text, PREC_SIGN)
        if isinstance(v, Decimal):
            text = format(v, "f")
            return (text, PREC_ATOM) if v >= 0 else (text, PREC_SIGN)
        if isinstance(v, str):
            return '"' + v.replace('"', '""') + '"', PREC_ATOM
        if isinstance(v, dt.datetime):
            date = f"DATE({v.year}, {v.month}, {v.day})"
            if v.time() == dt.time():
                return date, PREC_ATOM
            return f"{date} + TIME({v.hour}, {v.minute}, {v.second})", PREC_ADD
        if isinstance(v, dt.date):
            return f"DATE({v.year}, {v.month}, {v.day})", PREC_ATOM
        raise TypeError(f"can't write {v!r} as DAX")


class Keyword:
    """A DAX keyword argument, such as ``ASC``, ``YEAR`` in ``DATEADD`` or ``BOTH``."""

    __slots__ = ("name",)

    def __init__(self, name: str) -> None:
        self.name = name.upper()

    def __repr__(self) -> str:
        return self.name

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Keyword) and other.name == self.name

    def __hash__(self) -> int:
        return hash(self.name)


class _Omitted:
    """An optional argument left out in the middle of an argument list."""

    _instance: _Omitted | None = None

    def __new__(cls) -> _Omitted:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "OMITTED"

    def __bool__(self) -> bool:
        return False


OMITTED: _Omitted = _Omitted()
"""Placeholder for an optional argument that isn't given."""

Arg = Expr | TableExpr | ModifierExpr | Keyword | _Omitted


# --- operators ----------------------------------------------------------------

_BINARY_PREC = {
    "^": PREC_POWER,
    "*": PREC_MUL,
    "/": PREC_MUL,
    "+": PREC_ADD,
    "-": PREC_ADD,
    "&": PREC_CONCAT,
    "=": PREC_COMPARE,
    "==": PREC_COMPARE,
    "<>": PREC_COMPARE,
    "<": PREC_COMPARE,
    "<=": PREC_COMPARE,
    ">": PREC_COMPARE,
    ">=": PREC_COMPARE,
    "IN": PREC_COMPARE,
    "&&": PREC_AND,
    "||": PREC_OR,
}


class BinaryOp(Expr):
    def __init__(self, op: str, left: Node, right: Node) -> None:
        if left.kind == "modifier" or right.kind == "modifier":
            raise TypeError(f"a CALCULATE modifier can't be used with the {op} operator")
        if op != "IN" and right.kind == "table" and isinstance(right, TableConstructor):
            raise TypeError(f"a list can't be used with {op}; did you mean .IN([...])?")
        self.op = op
        self.left = left
        self.right = right

    def children(self) -> Iterable[Node]:
        return (self.left, self.right)

    @property
    def dax_type(self) -> DaxType | None:
        op = self.op
        if op in ("=", "==", "<>", "<", "<=", ">", ">=", "IN", "&&", "||"):
            return DaxType.BOOLEAN
        if op == "&":
            return DaxType.STRING
        a, b = self.left.dax_type, self.right.dax_type
        if op == "^":
            return DaxType.DECIMAL
        if a is None or b is None:
            return None
        if DaxType.DATETIME in (a, b) and op in ("+", "-"):
            return DaxType.DATETIME
        if op == "/":
            return (
                DaxType.CURRENCY
                if a is DaxType.CURRENCY and b is DaxType.INTEGER
                else (DaxType.DECIMAL)
            )
        if a is DaxType.INTEGER and b is DaxType.INTEGER:
            return DaxType.INTEGER
        if DaxType.CURRENCY in (a, b) and DaxType.DECIMAL not in (a, b):
            return DaxType.CURRENCY
        if a.is_numeric and b.is_numeric:
            return DaxType.DECIMAL
        return None

    def _dax(self) -> tuple[str, int]:
        prec = _BINARY_PREC[self.op]
        left, lp = self.left._dax()
        right, rp = self.right._dax()
        if lp < prec:
            left = f"({left})"
        if rp <= prec:
            right = f"({right})"
        return f"{left} {self.op} {right}", prec


class UnaryOp(Expr):
    def __init__(self, op: str, operand: Node) -> None:
        if operand.kind != "scalar":
            raise TypeError(f"the {op} sign needs a scalar value")
        self.op = op
        self.operand = operand

    def children(self) -> Iterable[Node]:
        return (self.operand,)

    @property
    def dax_type(self) -> DaxType | None:
        return self.operand.dax_type

    def _dax(self) -> tuple[str, int]:
        text, prec = self.operand._dax()
        if prec < PREC_SIGN:
            text = f"({text})"
        return f"{self.op}{text}", PREC_SIGN


# --- function calls -------------------------------------------------------------


class _CallMixin:
    name: str
    args: list[Arg]

    def children(self) -> Iterable[Node]:
        return [a for a in self.args if isinstance(a, _ExprBase)]

    def _dax(self) -> tuple[str, int]:
        parts: list[str] = []
        for a in self.args:
            if isinstance(a, _Omitted):
                parts.append("")
            elif isinstance(a, Keyword):
                parts.append(a.name)
            else:
                parts.append(a.to_dax())
        while parts and parts[-1] == "":
            parts.pop()
        return f"{self.name}({', '.join(parts)})", PREC_ATOM


class Call(_CallMixin, Expr):
    """A call to a DAX function that returns a scalar."""

    def __init__(self, name: str, args: list[Arg], dax_type: DaxType | None = None) -> None:
        self.name = name
        self.args = args
        self._type = dax_type

    @property
    def dax_type(self) -> DaxType | None:
        return self._type

    @staticmethod
    def raw(name: str, args: list[Arg], kind: Kind, dax_type: DaxType | None = None) -> Node:
        """Build a call node of the right kind without argument checks."""
        if kind == "table":
            return TableCall(name, args)
        if kind == "modifier":
            return ModifierCall(name, args)
        return Call(name, args, dax_type)


class TableCall(_CallMixin, TableExpr):
    """A call to a DAX function that returns a table."""

    def __init__(self, name: str, args: list[Arg]) -> None:
        self.name = name
        self.args = args


class ModifierCall(_CallMixin, ModifierExpr):
    """A call to a CALCULATE modifier."""

    def __init__(self, name: str, args: list[Arg]) -> None:
        self.name = name
        self.args = args


# --- tables written inline -------------------------------------------------------


class TableConstructor(TableExpr):
    """DAX ``{ ... }``, written in Python as a list: ``["Red", "Blue"]`` or ``[(1, "a")]``."""

    def __init__(self, rows: Sequence[Sequence[Expr]]) -> None:
        widths = {len(r) for r in rows}
        if len(widths) > 1:
            raise ValueError("every row of a table constructor needs the same number of values")
        self.rows = [list(r) for r in rows]

    def children(self) -> Iterable[Node]:
        return [v for row in self.rows for v in row]

    def _dax(self) -> tuple[str, int]:
        def row(values: list[Expr]) -> str:
            if len(values) == 1:
                return values[0].to_dax()
            return "(" + ", ".join(v.to_dax() for v in values) + ")"

        return "{" + ", ".join(row(r) for r in self.rows) + "}", PREC_ATOM


class ColumnName(Expr):
    """A column of a table expression that isn't a model column, such as one added by
    ADDCOLUMNS. DAX writes it ``[name]``. It reads the innermost row context that has a
    column with this name."""

    def __init__(self, name: str) -> None:
        self.name = name

    def _dax(self) -> tuple[str, int]:
        return "[" + self.name.replace("]", "]]") + "]", PREC_ATOM


# --- variables ------------------------------------------------------------------


class VarRef(Expr):
    """A reference to a DAX variable defined with :func:`VAR`."""

    def __init__(self, name: str, value: Node) -> None:
        self.name = name
        self.value = value

    @property
    def dax_type(self) -> DaxType | None:
        return self.value.dax_type

    def _dax(self) -> tuple[str, int]:
        return self.name, PREC_ATOM


class TableVarRef(TableExpr):
    """A reference to a table-valued DAX variable."""

    def __init__(self, name: str, value: Node) -> None:
        self.name = name
        self.value = value

    def _dax(self) -> tuple[str, int]:
        return self.name, PREC_ATOM


class _VarBlockMixin:
    variables: list[tuple[str, Node]]
    body: Node

    def children(self) -> Iterable[Node]:
        return [*(v for _, v in self.variables), self.body]

    def _dax(self) -> tuple[str, int]:
        lines = [f"VAR {name} = {value.to_dax()}" for name, value in self.variables]
        return " ".join([*lines, f"RETURN {self.body.to_dax()}"]), PREC_OR - 1


class VarBlock(_VarBlockMixin, Expr):
    """``VAR ... RETURN ...`` returning a scalar."""

    def __init__(self, variables: list[tuple[str, Node]], body: Node) -> None:
        self.variables = variables
        self.body = body

    @property
    def dax_type(self) -> DaxType | None:
        return self.body.dax_type


class TableVarBlock(_VarBlockMixin, TableExpr):
    """``VAR ... RETURN ...`` returning a table."""

    def __init__(self, variables: list[tuple[str, Node]], body: Node) -> None:
        self.variables = variables
        self.body = body


class Vars:
    """The variables defined so far, passed to the callables given to :func:`VAR`."""

    def __init__(self, refs: dict[str, Node]) -> None:
        self._refs = refs

    def __getattr__(self, name: str) -> Node:
        try:
            return self._refs[name]
        except KeyError:
            raise AttributeError(f"no variable named {name!r} is defined yet") from None

    def __getitem__(self, name: str) -> Node:
        return self._refs[name]


class VarBuilder:
    """The result of ``VAR(...)``. Call :meth:`RETURN` to finish it."""

    def __init__(self, variables: list[tuple[str, Node]], refs: dict[str, Node]) -> None:
        self._variables = variables
        self._refs = refs

    def RETURN(self, body: Any) -> Node:
        """The expression the block returns. Pass a callable to use the variables."""
        value = body(Vars(self._refs)) if callable(body) else body
        node = coerce_any(value)
        if node.kind == "modifier":
            raise TypeError("RETURN can't return a CALCULATE modifier")
        if node.kind == "table":
            return TableVarBlock(self._variables, node)
        return VarBlock(self._variables, node)

    def __repr__(self) -> str:
        names = ", ".join(n for n, _ in self._variables)
        return f"<VAR {names} (call .RETURN(...) to finish)>"


def VAR(**variables: Any) -> VarBuilder:
    """DAX ``VAR ... RETURN``. Each variable is evaluated once, where it is defined.

    Later variables can use earlier ones through a callable::

        VAR(
            Total=SUM(Sales["Amount"]),
            Average=lambda v: DIVIDE(v.Total, COUNTROWS(Sales)),
        ).RETURN(lambda v: IF(v.Total > 1000, v.Average))

    A plain Python variable holding an expression would be re-evaluated in every place
    it's used, so it would give different answers inside CALCULATE. Use VAR when an
    expression must be evaluated once, as in DAX.
    """
    from ._coverage import PLANNED

    if not variables:
        raise TypeError("VAR needs at least one variable, such as VAR(Total=...)")
    defined: list[tuple[str, Node]] = []
    refs: dict[str, Node] = {}
    for name, raw in variables.items():
        if not _IDENTIFIER.match(name) or name.upper() in PLANNED or name.upper() in KEYWORDS:
            raise ValueError(f"{name!r} can't be a DAX variable name")
        value = coerce_any(raw(Vars(refs)) if callable(raw) else raw)
        if value.kind == "modifier":
            raise TypeError("a variable can't hold a CALCULATE modifier")
        defined.append((name, value))
        refs[name] = VarRef(name, value) if value.kind == "scalar" else TableVarRef(name, value)
    return VarBuilder(defined, refs)


# --- conversion of Python values ----------------------------------------------------

KEYWORDS = frozenset(
    {
        "ASC", "DESC", "BOTH", "NONE", "ONEWAY", "ONEWAY_LEFTFILTERSRIGHT",
        "ONEWAY_RIGHTFILTERSLEFT", "DAY", "WEEK", "MONTH", "QUARTER", "YEAR", "HOUR",
        "MINUTE", "SECOND", "SKIP", "DENSE", "INTEGER", "DOUBLE", "STRING", "BOOLEAN",
        "CURRENCY", "DATETIME", "ABS", "REL", "DEFAULT", "FIRST", "LAST", "KEEP",
        "PARTITIONS",
    }
)  # fmt: skip

_SCALARS = (int, float, str, bool, Decimal, dt.date, dt.datetime)


def coerce(value: Any) -> Expr:
    """Turn a Python value into a scalar expression (numbers, text, dates, None)."""
    if isinstance(value, Expr):
        return value
    if value is None or isinstance(value, _SCALARS):
        return Literal_(value)
    if isinstance(value, TableExpr | ModifierExpr):
        raise TypeError(f"expected a scalar value, got a {value.kind} expression")
    if isinstance(value, list | tuple):
        raise TypeError("a list is a table in DAX; use .IN([...]) to test membership")
    raise TypeError(f"can't use a {type(value).__name__} in a DAX expression")


def coerce_table(value: Any, function: str) -> TableExpr:
    """Turn a table expression or a Python list into a table expression."""
    if isinstance(value, TableExpr):
        return value
    if isinstance(value, list | tuple):
        rows = [
            [coerce(v) for v in item] if isinstance(item, tuple) else [coerce(item)]
            for item in value
        ]
        if not rows:
            raise ValueError("an empty list can't be a DAX table constructor")
        return TableConstructor(rows)
    raise TypeError(f"{function} needs a table here, got {_describe(value)}")


def coerce_any(value: Any) -> Node:
    """A scalar, table or modifier expression from any supported Python value."""
    if isinstance(value, _ExprBase):
        return value  # type: ignore[return-value]
    if isinstance(value, list | tuple):
        return coerce_table(value, "a table expression")
    return coerce(value)


def keyword(value: Any, allowed: Iterable[str], function: str, param: str) -> Keyword:
    """A DAX keyword from a string, a Keyword, or a DAX function object such as YEAR."""
    if isinstance(value, Keyword):
        name = value.name
    elif isinstance(value, str):
        name = value.upper()
    elif callable(value) and hasattr(value, "_dax_name"):
        name = value._dax_name
    else:
        raise TypeError(f"{function}: {param} must be one of {', '.join(allowed)}")
    allowed = tuple(allowed)
    if name not in allowed:
        raise ValueError(f"{function}: {param} must be one of {', '.join(allowed)}, not {name}")
    return Keyword(name)


def _describe(value: Any) -> str:
    if isinstance(value, _ExprBase):
        return f"a {value.kind} expression ({value.to_dax()})"
    return f"a {type(value).__name__}"


def is_column(node: object) -> bool:
    return getattr(node, "is_column_ref", False) is True


def is_measure(node: object) -> bool:
    return getattr(node, "is_measure_ref", False) is True


def columns_in(node: Node) -> list[Column]:
    """Every column referenced anywhere in the expression."""
    return [n for n in node.walk() if is_column(n)]  # type: ignore[misc]


def measures_in(node: Node) -> list[Measure]:
    """Every measure referenced anywhere in the expression, directly or not."""
    seen: dict[int, Measure] = {}

    def visit(n: Node) -> None:
        for sub in n.walk():
            if is_measure(sub) and id(sub) not in seen:
                m: Measure = sub  # type: ignore[assignment]
                seen[id(m)] = m
                if isinstance(m.expression, _ExprBase):
                    visit(m.expression)  # type: ignore[arg-type]

    visit(node)
    return list(seen.values())


def quote_name(name: str) -> str:
    """A table name as DAX writes it: bare when simple, otherwise in single quotes."""
    from ._coverage import PLANNED

    if _IDENTIFIER.match(name) and name.upper() not in PLANNED and name.upper() not in KEYWORDS:
        return name
    return "'" + name.replace("'", "''") + "'"


Builder = Callable[..., Node]
