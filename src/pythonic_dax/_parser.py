"""A DAX text parser that builds the library's own expression objects.

DAX text is turned into exactly what the Python API builds: ``SUM(Sales[Amount])`` in
text becomes the result of calling :func:`pythonic_dax.SUM` with the column
``Sales["Amount"]``. Every check the Python functions make (argument kinds, Power BI's
build-time errors) applies to parsed text too, and evaluation goes through the same
engine. Syntax the engine doesn't reproduce raises :class:`NotSupportedError`.

The parser has two stages: :func:`tokenize` and :class:`_Parser` read the text into a
small syntax tree, then :class:`Converter` calls the Python functions. Names (tables,
columns, measures) are resolved through a :class:`Resolver`, so an importer can build
measures and calculated columns on demand, in dependency order.
"""

from __future__ import annotations

import datetime as dt
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from .errors import DaxError, ModelError, NotSupportedError
from .expr import (
    _IDENTIFIER,
    KEYWORDS,
    OMITTED,
    ColumnName,
    Expr,
    Keyword,
    Node,
    TableConstructor,
    TableVarBlock,
    TableVarRef,
    VarBlock,
    VarRef,
    coerce,
    coerce_any,
)

if TYPE_CHECKING:
    from .objects import Column, Measure, Table


# --- tokens ------------------------------------------------------------------------


@dataclass(frozen=True)
class Token:
    kind: str  # NUM, STR, DATE, IDENT, QUOTED, BRACKET, OP, EOF
    text: str
    pos: int
    value: Any = None


_OPERATORS = ("&&", "||", "==", "<>", "<=", ">=", "=", "<", ">", "+", "-", "*", "/", "^",
              "&", "(", ")", "{", "}", ",", ".")  # fmt: skip
_NUMBER = re.compile(r"(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?")
_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*")


def tokenize(text: str) -> list[Token]:
    """Split DAX text into tokens, skipping whitespace and comments."""
    tokens: list[Token] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch.isspace():
            i += 1
            continue
        if text.startswith(("//", "--"), i):
            end = text.find("\n", i)
            i = n if end < 0 else end + 1
            continue
        if text.startswith("/*", i):
            end = text.find("*/", i + 2)
            if end < 0:
                raise _syntax(text, i, "a /* comment is never closed")
            i = end + 2
            continue
        if ch == '"':
            value, j = _quoted(text, i, '"')
            tokens.append(Token("STR", text[i:j], i, value))
            i = j
            continue
        if ch == "'":
            value, j = _quoted(text, i, "'")
            tokens.append(Token("QUOTED", text[i:j], i, value))
            i = j
            continue
        if ch == "[":
            value, j = _bracket(text, i)
            tokens.append(Token("BRACKET", text[i:j], i, value))
            i = j
            continue
        if ch.isdigit() or (ch == "." and i + 1 < n and text[i + 1].isdigit()):
            m = _NUMBER.match(text, i)
            assert m is not None
            raw = m.group(0)
            if "." in raw or "e" in raw.lower():
                tokens.append(Token("NUM", raw, i, float(raw)))
            else:
                tokens.append(Token("NUM", raw, i, int(raw)))
            i = m.end()
            continue
        if ch.isalpha() or ch == "_":
            m = _WORD.match(text, i)
            assert m is not None
            word = m.group(0)
            j = m.end()
            if word.lower() == "dt" and j < n and text[j] == '"':
                value, k = _quoted(text, j, '"')
                tokens.append(Token("DATE", text[i:k], i, _date_literal(text, i, value)))
                i = k
                continue
            tokens.append(Token("IDENT", word, i, word))
            i = j
            continue
        for op in _OPERATORS:
            if text.startswith(op, i):
                tokens.append(Token("OP", op, i, op))
                i += len(op)
                break
        else:
            raise _syntax(text, i, f"unexpected character {ch!r}")
    tokens.append(Token("EOF", "", n))
    return tokens


def _quoted(text: str, start: int, quote: str) -> tuple[str, int]:
    out: list[str] = []
    i = start + 1
    while i < len(text):
        ch = text[i]
        if ch == quote:
            if text[i + 1 : i + 2] == quote:
                out.append(quote)
                i += 2
                continue
            return "".join(out), i + 1
        out.append(ch)
        i += 1
    raise _syntax(text, start, f"a {quote} is never closed")


def _bracket(text: str, start: int) -> tuple[str, int]:
    out: list[str] = []
    i = start + 1
    while i < len(text):
        ch = text[i]
        if ch == "]":
            if text[i + 1 : i + 2] == "]":
                out.append("]")
                i += 2
                continue
            return "".join(out), i + 1
        out.append(ch)
        i += 1
    raise _syntax(text, start, "a [ is never closed")


def _date_literal(text: str, pos: int, value: str) -> dt.datetime:
    try:
        return dt.datetime.fromisoformat(value.strip())
    except ValueError:
        raise _syntax(text, pos, f"dt{value!r} is not a date") from None


def _syntax(text: str, pos: int, message: str) -> DaxError:
    line = text.count("\n", 0, pos) + 1
    col = pos - (text.rfind("\n", 0, pos) + 1) + 1
    return DaxError(f"The syntax is incorrect (line {line}, column {col}): {message}.")


# --- syntax tree -------------------------------------------------------------------


@dataclass
class Ast:
    kind: str  # lit, date, ident, table, col, bracket, call, binop, unop, ctor, var, omitted
    pos: int
    value: Any = None
    args: list[Ast] = field(default_factory=list)


_COMPARE = ("=", "==", "<>", "<", "<=", ">", ">=")


class _Parser:
    def __init__(self, text: str) -> None:
        self.text = text
        self.tokens = tokenize(text)
        self.i = 0

    # helpers
    @property
    def tok(self) -> Token:
        return self.tokens[self.i]

    def next(self) -> Token:
        tok = self.tokens[self.i]
        self.i += 1
        return tok

    def at_op(self, *ops: str) -> bool:
        return self.tok.kind == "OP" and self.tok.text in ops

    def at_word(self, *words: str) -> bool:
        return self.tok.kind == "IDENT" and self.tok.text.upper() in words

    def expect_op(self, op: str) -> Token:
        if not self.at_op(op):
            raise self.error(f"expected {op!r}")
        return self.next()

    def error(self, message: str) -> DaxError:
        tok = self.tok
        found = "the end of the expression" if tok.kind == "EOF" else repr(tok.text)
        return _syntax(self.text, tok.pos, f"{message}, found {found}")

    # grammar
    def expression(self) -> Ast:
        if self.at_word("VAR"):
            return self.var_block()
        return self.or_expr()

    def var_block(self) -> Ast:
        start = self.tok.pos
        names: list[Ast] = []
        while self.at_word("VAR"):
            self.next()
            tok = self.next()
            if tok.kind != "IDENT" or "." in tok.text:
                raise _syntax(self.text, tok.pos, "expected a variable name")
            self.expect_op("=")
            value = self.expression()
            names.append(Ast("vardef", tok.pos, tok.text, [value]))
        if not self.at_word("RETURN"):
            raise self.error("expected RETURN")
        self.next()
        body = self.expression()
        return Ast("var", start, None, [*names, body])

    def or_expr(self) -> Ast:
        left = self.and_expr()
        while self.at_op("||"):
            tok = self.next()
            left = Ast("binop", tok.pos, "||", [left, self.and_expr()])
        return left

    def and_expr(self) -> Ast:
        left = self.not_expr()
        while self.at_op("&&"):
            tok = self.next()
            left = Ast("binop", tok.pos, "&&", [left, self.not_expr()])
        return left

    def not_expr(self) -> Ast:
        if self.at_word("NOT"):
            tok = self.next()
            return Ast("unop", tok.pos, "NOT", [self.not_expr()])
        return self.comparison()

    def comparison(self) -> Ast:
        left = self.concat()
        while True:
            if self.at_op(*_COMPARE):
                tok = self.next()
                left = Ast("binop", tok.pos, tok.text, [left, self.concat()])
            elif self.at_word("IN"):
                tok = self.next()
                left = Ast("binop", tok.pos, "IN", [left, self.concat()])
            else:
                return left

    def concat(self) -> Ast:
        left = self.additive()
        while self.at_op("&"):
            tok = self.next()
            left = Ast("binop", tok.pos, "&", [left, self.additive()])
        return left

    def additive(self) -> Ast:
        left = self.multiplicative()
        while self.at_op("+", "-"):
            tok = self.next()
            left = Ast("binop", tok.pos, tok.text, [left, self.multiplicative()])
        return left

    def multiplicative(self) -> Ast:
        left = self.unary()
        while self.at_op("*", "/"):
            tok = self.next()
            left = Ast("binop", tok.pos, tok.text, [left, self.unary()])
        return left

    def unary(self) -> Ast:
        if self.at_op("-", "+"):
            tok = self.next()
            return Ast("unop", tok.pos, tok.text, [self.unary()])
        return self.power()

    def power(self) -> Ast:
        left = self.primary()
        while self.at_op("^"):
            tok = self.next()
            right = self.signed_primary()
            left = Ast("binop", tok.pos, "^", [left, right])
        return left

    def signed_primary(self) -> Ast:
        if self.at_op("-", "+"):
            tok = self.next()
            return Ast("unop", tok.pos, tok.text, [self.signed_primary()])
        return self.primary()

    def primary(self) -> Ast:
        tok = self.tok
        if tok.kind in ("NUM", "STR"):
            self.next()
            return Ast("lit", tok.pos, tok.value)
        if tok.kind == "DATE":
            self.next()
            return Ast("lit", tok.pos, tok.value)
        if self.at_op("("):
            self.next()
            inner = self.expression()
            if self.at_op(","):
                items = [inner]
                while self.at_op(","):
                    self.next()
                    items.append(self.expression())
                self.expect_op(")")
                return Ast("row", tok.pos, None, items)
            self.expect_op(")")
            return Ast("paren", tok.pos, None, [inner])
        if self.at_op("{"):
            return self.constructor()
        if tok.kind == "BRACKET":
            self.next()
            node = Ast("bracket", tok.pos, tok.value)
            return self.variation(node)
        if tok.kind == "QUOTED":
            self.next()
            if self.tok.kind == "BRACKET":
                col = self.next()
                return self.variation(Ast("col", tok.pos, (tok.value, col.value)))
            return Ast("table", tok.pos, tok.value)
        if tok.kind == "IDENT":
            self.next()
            if self.at_op("("):
                return self.call(tok)
            if self.tok.kind == "BRACKET" and "." not in tok.text:
                col = self.next()
                return self.variation(Ast("col", tok.pos, (tok.text, col.value)))
            return Ast("ident", tok.pos, tok.text)
        raise self.error("expected a value")

    def variation(self, node: Ast) -> Ast:
        if self.at_op(".") and self.tokens[self.i + 1].kind == "BRACKET":
            raise NotSupportedError("column variations such as Sales[OrderDate].[Year]")
        return node

    def call(self, name: Token) -> Ast:
        self.expect_op("(")
        args: list[Ast] = []
        if not self.at_op(")"):
            while True:
                if self.at_op(",", ")"):
                    args.append(Ast("omitted", self.tok.pos))
                else:
                    args.append(self.expression())
                if self.at_op(","):
                    self.next()
                    if self.at_op(")"):
                        args.append(Ast("omitted", self.tok.pos))
                        break
                    continue
                break
        self.expect_op(")")
        return Ast("call", name.pos, name.text.upper(), args)

    def constructor(self) -> Ast:
        start = self.expect_op("{")
        rows: list[Ast] = []
        if not self.at_op("}"):
            while True:
                if self.at_op("{"):
                    rows.append(self.constructor())  # DATATABLE's rows
                else:
                    rows.append(self.expression())
                if self.at_op(","):
                    self.next()
                    continue
                break
        self.expect_op("}")
        return Ast("ctor", start.pos, None, rows)

    def end(self) -> None:
        if self.tok.kind != "EOF":
            raise self.error("expected the end of the expression")


def parse_ast(text: str) -> Ast:
    """Parse a DAX expression into a syntax tree."""
    parser = _Parser(text)
    tree = parser.expression()
    parser.end()
    return tree


# --- name resolution ------------------------------------------------------------------


class Resolver(Protocol):
    """Finds the model objects a DAX expression names."""

    def table(self, name: str) -> Table | None: ...

    def column(self, table: Table, name: str) -> Column | None: ...

    def measure(self, name: str) -> Measure | None: ...


class ModelResolver:
    """Resolves names against a model's existing tables, columns and measures."""

    def __init__(self, model: Any) -> None:
        self.model = model

    def table(self, name: str) -> Table | None:
        return self.model._tables.get(name)  # type: ignore[no-any-return]

    def column(self, table: Table, name: str) -> Column | None:
        return table._columns.get(name)

    def measure(self, name: str) -> Measure | None:
        return self.model._measures.get(name)  # type: ignore[no-any-return]


# --- conversion to expressions --------------------------------------------------------

_DATATABLE_TYPES = ("INTEGER", "DOUBLE", "STRING", "BOOLEAN", "CURRENCY", "DATETIME")


class Converter:
    """Turns a syntax tree into expression objects by calling the DAX functions."""

    def __init__(
        self,
        text: str,
        resolver: Resolver,
        home: Table | None = None,
        functions: Callable[[str], Any] | None = None,
    ) -> None:
        self.text = text
        self.resolver = resolver
        self.home = home  # a calculated column's table: [Name] can mean its columns
        self.scopes: list[dict[str, Node]] = []
        self._functions = functions or _function

    def error(self, node: Ast, message: str) -> DaxError:
        return _syntax(self.text, node.pos, message)

    # Values passed to a function stay plain Python values where the Python API
    # expects them: text for names in ADDCOLUMNS or SUMMARIZECOLUMNS, numbers for
    # counts. Everywhere else they become literals.
    def arg(self, node: Ast) -> Any:
        if node.kind == "lit":
            return node.value
        if node.kind == "omitted":
            return OMITTED
        if node.kind == "unop" and node.value == "-" and node.args[0].kind == "lit":
            value = node.args[0].value
            if isinstance(value, int | float) and not isinstance(value, bool):
                return -value
        if (
            node.kind == "ident"
            and self.lookup(node.value) is None
            and node.value.upper() in KEYWORDS
            and self.resolver.table(node.value) is None
        ):
            return Keyword(node.value)
        return self.node(node)

    def scalar(self, node: Ast) -> Expr:
        value = self.node(node)
        if not isinstance(value, Expr):
            raise DaxError(f"expected a single value, found a {value.kind} expression")
        return value

    def node(self, node: Ast) -> Node:
        method = getattr(self, "_" + node.kind)
        result: Node = method(node)
        return result

    def _lit(self, node: Ast) -> Node:
        return coerce(node.value)

    def _paren(self, node: Ast) -> Node:
        return self.node(node.args[0])

    def _omitted(self, node: Ast) -> Node:
        raise self.error(node, "an argument is missing")

    def _row(self, node: Ast) -> Node:
        raise NotSupportedError("a row of values such as (1, 2) outside a table constructor")

    def lookup(self, name: str) -> Node | None:
        key = name.casefold()
        for scope in reversed(self.scopes):
            if key in scope:
                return scope[key]
        return None

    def _ident(self, node: Ast) -> Node:
        name: str = node.value
        var = self.lookup(name)
        if var is not None:
            if var is _QUERY_VAR:
                raise NotSupportedError(
                    f"the query variable {name} inside a DEFINE MEASURE or COLUMN"
                )
            return var
        upper = name.upper()
        if upper in ("TRUE", "FALSE"):
            return self._functions(upper)()  # type: ignore[no-any-return]
        table = self.resolver.table(name)
        if table is not None:
            return table
        if upper in KEYWORDS:
            raise self.error(node, f"the keyword {upper} can't be used here")
        raise DaxError(
            f"Failed to resolve name '{name}'. It is not a valid table, variable, or function name."
        )

    def _table(self, node: Ast) -> Node:
        table = self.resolver.table(node.value)
        if table is None:
            raise DaxError(f"Cannot find table '{node.value}'.")
        return table

    def _col(self, node: Ast) -> Node:
        table_name, col_name = node.value
        table = self.resolver.table(table_name)
        if table is None:
            raise DaxError(f"Cannot find table '{table_name}'.")
        column = self.resolver.column(table, col_name)
        if column is not None:
            return column
        measure = self.resolver.measure(col_name)
        if measure is not None and measure.table is table:
            return measure  # Table[Measure] is a valid measure reference
        raise DaxError(
            f"Column '{col_name}' in table '{table.name}' cannot be found or may not be "
            "used in this expression."
        )

    def _bracket(self, node: Ast) -> Node:
        name: str = node.value
        measure = self.resolver.measure(name)
        if measure is not None:
            return measure
        if self.home is not None:
            column = self.resolver.column(self.home, name)
            if column is not None:
                return column
        return ColumnName(name)  # a column of a table expression, such as ADDCOLUMNS

    def _ctor(self, node: Ast) -> Node:
        rows: list[list[Expr]] = []
        for item in node.args:
            if item.kind == "row":
                rows.append([self.scalar(v) for v in item.args])
            elif item.kind == "ctor":
                raise self.error(item, "a table constructor can't contain another one")
            else:
                rows.append([self.scalar(item)])
        if not rows:
            raise self.error(node, "a table constructor needs at least one value")
        return TableConstructor(rows)

    def _unop(self, node: Ast) -> Node:
        op = node.value
        if op == "NOT":
            return self._functions("NOT")(self.arg(node.args[0]))  # type: ignore[no-any-return]
        operand = node.args[0]
        number = operand.kind == "lit" and isinstance(operand.value, int | float)
        if number and op == "-" and not isinstance(operand.value, bool):
            return coerce(-operand.value)
        value = self.scalar(operand)
        return -value if op == "-" else +value

    def _binop(self, node: Ast) -> Node:
        op = node.value
        left_ast, right_ast = node.args
        if op == "IN":
            if left_ast.kind == "row":
                raise NotSupportedError("IN with a row of several values on the left")
            right = self.node(right_ast)
            return self.scalar(left_ast).IN(right)
        left = self.scalar(left_ast)
        right = self.scalar(right_ast)
        if op == "+":
            return left + right
        if op == "-":
            return left - right
        if op == "*":
            return left * right
        if op == "/":
            return left / right
        if op == "^":
            return left**right
        if op == "&":
            return left.concat(right)
        if op == "&&":
            return left & right
        if op == "||":
            return left | right
        if op == "=":
            return left == right
        if op == "==":
            return left.strict_eq(right)
        if op == "<>":
            return left != right
        if op == "<":
            return left < right
        if op == "<=":
            return left <= right
        if op == ">":
            return left > right
        if op == ">=":
            return left >= right
        raise AssertionError(op)

    def _var(self, node: Ast) -> Node:
        from ._coverage import PLANNED

        *defs, body_ast = node.args
        scope: dict[str, Node] = {}
        self.scopes.append(scope)
        try:
            defined: list[tuple[str, Node]] = []
            for d in defs:
                name: str = d.value
                if (
                    not _IDENTIFIER.match(name)
                    or name.upper() in PLANNED
                    or name.upper() in KEYWORDS
                ):
                    raise self.error(d, f"{name!r} can't be a variable name")
                value = coerce_any(self.node(d.args[0]))
                if value.kind == "modifier":
                    raise DaxError("a variable can't hold a CALCULATE modifier")
                defined.append((name, value))
                scope[name.casefold()] = (
                    VarRef(name, value) if value.kind == "scalar" else TableVarRef(name, value)
                )
            body = coerce_any(self.node(body_ast))
        finally:
            self.scopes.pop()
        if body.kind == "modifier":
            raise DaxError("RETURN can't return a CALCULATE modifier")
        if body.kind == "table":
            return TableVarBlock(defined, body)
        return VarBlock(defined, body)

    def _call(self, node: Ast) -> Node:
        name: str = node.value
        if name == "DATATABLE":
            return self._datatable(node)
        if name.startswith("INFO."):
            return self._info(node)
        if name == "VAR":
            raise self.error(node, "VAR must be followed by a name and =")
        fn = self._functions(name)
        if fn is None:
            raise NotSupportedError(f"the DAX function {name}")
        args = [self.arg(a) for a in node.args]
        if name in _YEAR_END_THIRD and len(args) == 3 and isinstance(args[2], str):
            args = [args[0], args[1], OMITTED, args[2]]  # a text third argument is the year end
        try:
            result: Node = fn(*args)
        except TypeError as exc:
            if "positional argument" in str(exc) or "missing" in str(exc):
                raise DaxError(
                    f"Too few or too many arguments were passed to {name}: {exc}"
                ) from exc
            raise
        return result

    def _info(self, node: Ast) -> Node:
        from ._special import INFO

        parts = node.value.split(".")
        target: Any = INFO
        try:
            for part in parts[1:]:
                target = getattr(target, part)
        except AttributeError:
            raise NotSupportedError(f"the DAX function {node.value}") from None
        if node.args:
            raise DaxError(f"{node.value} takes no arguments")
        return target()  # type: ignore[no-any-return]

    def _constant(self, node: Ast) -> Any:
        """A DATATABLE value: a literal, or BLANK(), TRUE() or FALSE()."""
        if node.kind == "call" and not node.args and node.value in ("BLANK", "TRUE", "FALSE"):
            return {"BLANK": None, "TRUE": True, "FALSE": False}[node.value]
        if node.kind == "ident" and node.value.upper() in ("TRUE", "FALSE"):
            return node.value.upper() == "TRUE"
        return self.arg(node)

    def _datatable(self, node: Ast) -> Node:
        from ._special import DATATABLE

        *pairs, rows_ast = node.args
        if len(pairs) % 2 or not pairs or rows_ast.kind != "ctor":
            raise self.error(node, "DATATABLE needs name, type pairs and then the rows")
        columns: dict[str, str] = {}
        for i in range(0, len(pairs), 2):
            name_ast, type_ast = pairs[i], pairs[i + 1]
            if name_ast.kind != "lit" or not isinstance(name_ast.value, str):
                raise self.error(name_ast, "DATATABLE column names must be text")
            if type_ast.kind != "ident" or type_ast.value.upper() not in _DATATABLE_TYPES:
                raise self.error(type_ast, "expected a DATATABLE type such as STRING")
            columns[name_ast.value] = type_ast.value.upper()
        rows: list[list[Any]] = []
        for row in rows_ast.args:
            items = row.args if row.kind == "ctor" else [row]
            rows.append([self._constant(v) for v in items])
        return DATATABLE(columns, rows)


_YEAR_END_THIRD = {"TOTALYTD", "OPENINGBALANCEYEAR", "CLOSINGBALANCEYEAR"}


def _function(name: str) -> Any:
    import pythonic_dax

    if name not in pythonic_dax.__all__:
        return None
    fn = getattr(pythonic_dax, name)
    return fn if callable(fn) and not isinstance(fn, Keyword) else None


def parse(
    text: str,
    resolver: Resolver,
    home: Table | None = None,
) -> Node:
    """Parse a DAX expression and build it as a library expression."""
    tree = parse_ast(text)
    return Converter(text, resolver, home).node(tree)


# --- DAX queries ----------------------------------------------------------------------


@dataclass
class Definition:
    kind: str  # MEASURE, COLUMN, VAR, TABLE
    table: str | None
    name: str
    text: str  # the expression's DAX


@dataclass
class Query:
    definitions: list[Definition]
    evaluate: str
    order_by: str | None = None


def split_query(text: str) -> Query:
    """Split a DAX query into its DEFINE entries, EVALUATE expression and ORDER BY.

    Only one EVALUATE is supported, as in Power BI's DAX query view result pane.
    """
    tokens = tokenize(text)
    marks: list[tuple[str, int, int]] = []  # (keyword, token index, char position)
    depth = 0
    open_vars = 0  # VAR ... RETURN blocks still open inside the current definition
    for i, tok in enumerate(tokens):
        if tok.kind == "OP" and tok.text in ("(", "{"):
            depth += 1
        elif tok.kind == "OP" and tok.text in (")", "}"):
            depth -= 1
        elif depth == 0 and tok.kind == "IDENT":
            word = tok.text.upper()
            if word in ("DEFINE", "EVALUATE", "MEASURE", "COLUMN", "TABLE", "ORDER", "START"):
                if word == "ORDER" and tokens[i + 1].text.upper() != "BY":
                    continue
                if word == "TABLE" and not _starts_definition(tokens, i):
                    continue
                marks.append((word, i, tok.pos))
                open_vars = 0
            elif word == "VAR":
                after_equals = i > 0 and tokens[i - 1].text == "="
                if after_equals:
                    open_vars += 1  # the expression itself starts with VAR
                elif open_vars == 0 and _inside_define(marks):
                    marks.append(("VAR", i, tok.pos))  # a query variable
            elif word == "RETURN" and open_vars:
                open_vars -= 1
    if [m[0] for m in marks].count("EVALUATE") != 1:
        raise NotSupportedError("a DAX query without exactly one EVALUATE")
    if any(m[0] == "START" for m in marks):
        raise NotSupportedError("START AT in a DAX query")

    query = Query([], "")
    ends = [m[2] for m in marks[1:]] + [len(text)]
    for (word, index, _pos), end in zip(marks, ends, strict=True):
        if word == "DEFINE":
            continue
        if word == "EVALUATE":
            query.evaluate = text[tokens[index + 1].pos : end]
            continue
        if word == "ORDER":
            query.order_by = text[tokens[index + 2].pos : end]
            continue
        query.definitions.append(_definition(text, tokens, word, index, end))
    return query


def _inside_define(marks: list[tuple[str, int, int]]) -> bool:
    kinds = [m[0] for m in marks]
    return "DEFINE" in kinds and "EVALUATE" not in kinds


def _starts_definition(tokens: list[Token], i: int) -> bool:
    nxt = tokens[i + 1]
    return nxt.kind in ("IDENT", "QUOTED") and tokens[i + 2].text == "="


def _definition(text: str, tokens: list[Token], word: str, index: int, end: int) -> Definition:
    t = index + 1
    table: str | None = None
    if word in ("MEASURE", "COLUMN"):
        if tokens[t].kind not in ("IDENT", "QUOTED") or tokens[t + 1].kind != "BRACKET":
            raise _syntax(text, tokens[t].pos, f"expected Table[Name] after {word}")
        table = tokens[t].value
        name = tokens[t + 1].value
        t += 2
    else:
        name = tokens[t].value
        t += 1
    if tokens[t].text != "=":
        raise _syntax(text, tokens[t].pos, "expected =")
    return Definition(word, table, name, text[tokens[t + 1].pos : end])


__all__ = ["ModelError", "parse", "parse_ast", "split_query", "tokenize"]


def evaluate_query(model: Any, text: str) -> Any:
    """Run a DAX query (``DEFINE ... EVALUATE ...``) on a model and return a DataFrame.

    As in Power BI, ``DEFINE MEASURE`` and ``DEFINE COLUMN`` last only for the query (a
    query measure replaces a model measure of the same name), and definitions may refer
    to each other in any order.
    """
    query = split_query(text)
    if query.order_by is not None:
        raise NotSupportedError("ORDER BY in a DAX query")
    if any(d.kind == "TABLE" for d in query.definitions):
        raise NotSupportedError("DEFINE TABLE in a DAX query")
    builder = _QueryBuilder(model, query)
    try:
        return model.evaluate(builder.build())
    finally:
        builder.undo()


class _QueryBuilder:
    """Builds a query's definitions on demand, so they may be used before they appear."""

    def __init__(self, model: Any, query: Query) -> None:
        self.model = model
        self.query = query
        self.base = ModelResolver(model)
        self.pending: dict[tuple[str, str], Definition] = {}
        self.vars: dict[str, Node] = {}
        self.building: list[str] = []
        self.undo_steps: list[Callable[[], Any]] = []
        for d in query.definitions:
            if d.kind == "MEASURE":
                self.pending[("measure", d.name.casefold())] = d
            elif d.kind == "COLUMN":
                assert d.table is not None
                self.pending[("column", f"{d.table.casefold()}[{d.name.casefold()}]")] = d

    # Resolver protocol
    def table(self, name: str) -> Table | None:
        return self.base.table(name)

    def column(self, table: Table, name: str) -> Column | None:
        key = ("column", f"{table.name.casefold()}[{name.casefold()}]")
        if key in self.pending:
            self._build(self.pending.pop(key))
        return self.base.column(table, name)

    def measure(self, name: str) -> Measure | None:
        key = ("measure", name.casefold())
        if key in self.pending:
            self._build(self.pending.pop(key))
        return self.base.measure(name)

    def _converter(self, text: str, home: Table | None = None, *, query: bool = False) -> Converter:
        conv = Converter(text, self, home)
        if query:
            conv.scopes.append(dict(self.vars))
        else:
            conv.scopes.append({k: _QUERY_VAR for k in self.vars})
        return conv

    def _build(self, d: Definition) -> None:
        label = f"{d.table}[{d.name}]" if d.table else d.name
        if label in self.building:
            raise DaxError(
                f"A circular dependency was detected: {', '.join([*self.building, label])}"
            )
        self.building.append(label)
        try:
            assert d.table is not None
            home = self.model.table(d.table)
            if d.kind == "MEASURE":
                existing = self.base.measure(d.name)
                if existing is not None:
                    # A query measure replaces the model's measure for this query.
                    if existing.table is not home:
                        raise DaxError(f"the measure [{d.name}] belongs to {existing.table.ref}")
                    old = existing.expression
                    self.undo_steps.append(lambda: setattr(existing, "expression", old))
                    expr = coerce_any(self._converter(d.text).node(parse_ast(d.text)))
                    existing.expression = expr
                    return
                expr = self._converter(d.text).node(parse_ast(d.text))
                self.model.add_measure(d.name, expr, table=home)
                self.undo_steps.append(lambda: self.model._measures.remove(d.name))
            else:
                if d.name in home:
                    raise NotSupportedError(
                        f"DEFINE COLUMN for {home.ref}[{d.name}], which the model already has"
                    )
                expr = self._converter(d.text, home).node(parse_ast(d.text))
                column = self.model.add_column(home, d.name, expr)
                self.undo_steps.append(lambda: _drop_column(self.model, column))
        finally:
            self.building.pop()

    def undo(self) -> None:
        """Remove what the query defined: query definitions last only for the query."""
        for step in reversed(self.undo_steps):
            step()
        self.undo_steps.clear()

    def build(self) -> Node:
        defined: list[tuple[str, Node]] = []
        for d in self.query.definitions:
            if d.kind == "VAR":
                conv = self._converter(d.text, query=True)
                value = coerce_any(conv.node(parse_ast(d.text)))
                if value.kind == "modifier":
                    raise DaxError("a variable can't hold a CALCULATE modifier")
                defined.append((d.name, value))
                self.vars[d.name.casefold()] = (
                    VarRef(d.name, value) if value.kind == "scalar" else TableVarRef(d.name, value)
                )
        for d in list(self.query.definitions):
            if d.kind == "MEASURE" and ("measure", d.name.casefold()) in self.pending:
                self.measure(d.name)
            elif d.kind == "COLUMN" and d.table is not None:
                table = self.model.table(d.table)
                self.column(table, d.name)
        conv = self._converter(self.query.evaluate, query=True)
        body = coerce_any(conv.node(parse_ast(self.query.evaluate)))
        if body.kind != "table":
            raise DaxError("EVALUATE needs a table expression")
        return TableVarBlock(defined, body) if defined else body


def _drop_column(model: Any, column: Any) -> None:
    table = column.table
    table._columns.remove(column.name)
    table._data = table._data.drop(column.name)
    model._calculated[:] = [c for c in model._calculated if c is not column]
    for rel in model._relationships:
        rel._invalidate()


_QUERY_VAR: Any = object()
"""Stands for a query variable inside a DEFINE MEASURE or COLUMN (not supported)."""
