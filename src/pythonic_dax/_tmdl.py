"""A reader for TMDL (Tabular Model Definition Language) folders.

TMDL is the text format Power BI Desktop saves a semantic model in (in a ``.pbip``
project's ``<name>.SemanticModel/definition`` folder). This module turns it into the
same JSON shape as a ``model.bim`` file, so one builder (:mod:`pythonic_dax.importing`)
handles both formats.

The format, from Microsoft's TMDL overview:

* Objects are declared as ``<type> <name>``, optionally followed by ``= <expression>``
  (a measure's formula, a calculated column's formula, a partition's kind).
* Child objects and properties are indented one tab deeper than their parent.
  Properties are ``name: value``, ``name = <expression>``, or a bare ``name`` for a
  true flag such as ``isHidden``.
* A multi-line expression continues on lines indented deeper than its properties,
  or is fenced between ``````` lines.
* ``///`` lines before an object are its description.
* Names with spaces or special characters are in single quotes, with ``''`` for a
  quote inside.
"""

from __future__ import annotations

import re
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import ModelError


@dataclass
class Node:
    """One TMDL object: its type, name, default expression, properties and children."""

    type: str
    name: str
    expression: str | None = None
    properties: dict[str, Any] = field(default_factory=dict)
    children: list[Node] = field(default_factory=list)
    description: str | None = None
    line: int = 0

    def of(self, kind: str) -> list[Node]:
        return [c for c in self.children if c.type == kind]


@dataclass
class _Line:
    number: int
    indent: int
    text: str  # without indentation or the trailing newline
    raw: str = ""  # the whole line, for expressions


_OBJECT_TYPES = frozenset(
    {
        "model", "database", "table", "column", "measure", "hierarchy", "level",
        "partition", "relationship", "annotation", "extendedProperty", "expression",
        "cultureInfo", "translations", "linguisticMetadata", "role", "tablePermission",
        "perspective", "perspectiveTable", "perspectiveColumn", "perspectiveMeasure",
        "perspectiveHierarchy", "calculationGroup", "calculationItem", "kpi",
        "dataSource", "queryGroup", "ref", "variation", "function", "calendar",
        "calendarColumnGroup", "columnPermission", "member", "changedProperty",
        "formatStringDefinition", "detailRowsDefinition", "alternateOf", "refreshPolicy",
        "dataAccessOptions", "sourceLineageTag", "queryGroups",
    }
)  # fmt: skip
"""Keywords that start an object (or an expression property) rather than a plain property."""

# Objects whose children are not TMDL objects but translated names of model objects.
_PROPERTY_ONLY = frozenset({"translations"})


def read_folder(path: str | Path) -> dict[str, Any]:
    """Read a TMDL folder into the JSON shape of a ``model.bim`` file.

    ``path`` is the ``definition`` folder, or a folder that contains one (such as a
    ``.SemanticModel`` folder).
    """
    root = Path(path)
    if not root.is_dir():
        raise ModelError(f"{root} is not a folder")
    if (root / "definition").is_dir() and not (root / "model.tmdl").exists():
        root = root / "definition"
    if not (root / "model.tmdl").exists():
        raise ModelError(f"{root} has no model.tmdl, so it isn't a TMDL folder")

    nodes: list[Node] = []
    files = [root / "database.tmdl", root / "model.tmdl", root / "relationships.tmdl"]
    files += [root / "expressions.tmdl", root / "functions.tmdl"]
    for sub in ("tables", "roles", "perspectives", "cultures"):
        if (root / sub).is_dir():
            files += sorted((root / sub).glob("*.tmdl"))
    for file in files:
        if file.exists():
            nodes += parse(file.read_text(encoding="utf-8-sig"), str(file))
    return to_bim(nodes)


# --- parsing --------------------------------------------------------------------


def parse(text: str, source: str = "<tmdl>") -> list[Node]:
    """Parse TMDL text into its top-level objects."""
    lines = _lines(text, source)
    parser = _Parser(lines, source)
    return parser.block(0, -1)


def _lines(text: str, source: str) -> list[_Line]:
    out: list[_Line] = []
    indent_unit: str | None = None
    for number, raw in enumerate(text.splitlines(), start=1):
        stripped = raw.lstrip(" \t")
        lead = raw[: len(raw) - len(stripped)]
        if not stripped:
            out.append(_Line(number, -1, "", ""))  # blank: belongs to whatever surrounds it
            continue
        if "\t" in lead:
            indent = lead.count("\t") + lead.replace("\t", "").__len__() // 4
        elif lead:
            if indent_unit is None:
                indent_unit = lead
            indent = len(lead) // len(indent_unit)
        else:
            indent = 0
        out.append(_Line(number, indent, stripped.rstrip(), raw.rstrip()))
    return out


_DECL = re.compile(r"^(?P<type>[A-Za-z]+)(?:\s+(?P<rest>.*))?$")


class _Parser:
    def __init__(self, lines: list[_Line], source: str) -> None:
        self.lines = lines
        self.pos = 0
        self.source = source

    def error(self, line: _Line, message: str) -> ModelError:
        return ModelError(f"{self.source}, line {line.number}: {message}")

    def peek(self) -> _Line | None:
        while self.pos < len(self.lines) and self.lines[self.pos].indent < 0:
            self.pos += 1
        return self.lines[self.pos] if self.pos < len(self.lines) else None

    def _next_content(self) -> _Line | None:
        i = self.pos
        while i < len(self.lines) and self.lines[i].indent < 0:
            i += 1
        return self.lines[i] if i < len(self.lines) else None

    def block(self, indent: int, parent_indent: int, parent: Node | None = None) -> list[Node]:
        """Read the children of an object whose own line is at ``parent_indent``."""
        nodes: list[Node] = []
        description: list[str] = []
        while (line := self.peek()) is not None:
            if line.indent <= parent_indent:
                break
            if line.indent > indent:
                raise self.error(line, "unexpected indentation")
            self.pos += 1
            text = line.text
            if text.startswith("///"):
                description.append(text[3:].removeprefix(" "))
                continue
            if text.startswith("//"):
                continue
            node = self.statement(line, parent)
            if node is not None:
                if description:
                    node.description = "\n".join(description)
                nodes.append(node)
            description = []
        return nodes

    def statement(self, line: _Line, parent: Node | None) -> Node | None:
        text = line.text
        if parent is not None and parent.type in _PROPERTY_ONLY:
            return self.object(line, parent_type=parent.type)
        m = _DECL.match(text)
        keyword = m.group("type") if m else ""
        if m and keyword in _OBJECT_TYPES and not _is_property(text, keyword):
            return self.object(line)
        # An object type this reader doesn't know, recognized by its indented children.
        nxt = self._next_content()
        if (
            m
            and ":" not in text
            and "=" not in text
            and nxt is not None
            and nxt.indent > line.indent
        ):
            return self.object(line)
        # A property of the parent.
        if parent is None:
            raise self.error(line, f"expected an object, found {text!r}")
        key, value = self.property(line)
        parent.properties[key] = value
        return None

    def object(self, line: _Line, parent_type: str | None = None) -> Node:
        m = _DECL.match(line.text)
        if m is None:
            raise self.error(line, f"can't read {line.text!r}")
        kind = m.group("type")
        rest = m.group("rest") or ""
        if kind == "ref":  # ref table Sales: the order of objects, nothing more
            kind, rest = [*rest.split(None, 1), ""][:2]
            name, _ = _read_name(rest)
            node = Node("ref", name, None, {"kind": kind}, line=line.number)
            node.children = self.block(line.indent + 1, line.indent, node)
            return node
        name, after = _read_name(rest)
        expression: str | None = None
        after = after.strip()
        if after.startswith("="):
            expression = self.expression(line, after[1:], object_line=True)
        elif after:
            raise self.error(line, f"unexpected text after the name: {after!r}")
        node = Node(kind, name, expression, line=line.number)
        node.children = self.block(line.indent + 1, line.indent, node)
        return node

    def property(self, line: _Line) -> tuple[str, Any]:
        text = line.text
        colon = text.find(":")
        equals = text.find("=")
        if colon > 0 and (equals < 0 or colon < equals):
            key = text[:colon].strip()
            return key, _value(text[colon + 1 :].strip())
        if equals > 0:
            key = text[:equals].strip()
            return key, self.expression(line, text[equals + 1 :], object_line=False)
        return text.strip(), True

    def expression(self, line: _Line, first: str, object_line: bool) -> str:
        """An expression after ``=``: the rest of the line, then any deeper lines."""
        first = first.strip()
        # Continuation lines sit deeper than the object's properties (object) or
        # deeper than the property itself (property).
        floor = line.indent + (1 if object_line else 0)
        if first.startswith("```"):
            return self.fenced(line, first[3:])
        body: list[_Line] = []
        while self.pos < len(self.lines):
            nxt = self.lines[self.pos]
            if nxt.indent >= 0 and nxt.indent <= floor:
                break
            body.append(nxt)
            self.pos += 1
        while body and body[-1].indent < 0:
            body.pop()
        if not body:
            return first
        rendered = _dedent(body)
        return f"{first}\n{rendered}" if first else rendered

    def fenced(self, line: _Line, first: str) -> str:
        parts = [first] if first.strip() else []
        min_indent: int | None = None
        body: list[_Line] = []
        while self.pos < len(self.lines):
            nxt = self.lines[self.pos]
            self.pos += 1
            if nxt.text.startswith("```"):
                break
            body.append(nxt)
            if nxt.indent >= 0:
                min_indent = nxt.indent if min_indent is None else min(min_indent, nxt.indent)
        else:
            raise self.error(line, "a ``` block is never closed")
        parts.append(_dedent(body))
        return "\n".join(p for p in parts if p)


def _is_property(text: str, keyword: str) -> bool:
    """Whether a line starting with an object keyword is really a property (``kpi:``)."""
    after = text[len(keyword) :]
    return after.startswith(":")


def _dedent(lines: list[_Line]) -> str:
    """Expression lines with their common indentation removed, otherwise as written."""
    return textwrap.dedent("\n".join(ln.raw for ln in lines))


def _read_name(text: str) -> tuple[str, str]:
    """A possibly quoted name at the start of ``text``, and what follows it."""
    text = text.lstrip()
    if text.startswith("'"):
        i = 1
        out: list[str] = []
        while i < len(text):
            ch = text[i]
            if ch == "'":
                if text[i + 1 : i + 2] == "'":
                    out.append("'")
                    i += 2
                    continue
                return "".join(out), text[i + 1 :]
            out.append(ch)
            i += 1
        raise ModelError(f"a quoted name is never closed: {text!r}")
    m = re.match(r"[^\s=]+", text)
    if not m:
        return "", text
    return m.group(0), text[m.end() :]


def _value(text: str) -> Any:
    """A property value: ``true``/``false``, a quoted string, or text."""
    if text in ("true", "false"):
        return text == "true"
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        return text[1:-1].replace('""', '"')
    return text  # numbers stay text: a format string such as 0 is text


def split_ref(text: str) -> tuple[str, str]:
    """Split ``Table.Column`` (either part possibly quoted) into its two names."""
    text = text.strip()
    if text.startswith("'"):
        table, rest = _read_name(text)
    else:
        table, dot, tail = text.partition(".")
        rest = dot + tail
    if not rest.startswith("."):
        raise ModelError(f"expected Table.Column, found {text!r}")
    column, tail = _read_name(rest[1:])
    if tail.strip():
        raise ModelError(f"expected Table.Column, found {text!r}")
    return table, column


# --- conversion to the model.bim JSON shape ---------------------------------------------


_TABLE_PROPS = ("isHidden", "dataCategory", "isPrivate", "showAsVariationsOnly")
_COLUMN_PROPS = (
    "dataType", "sourceColumn", "formatString", "displayFolder", "isHidden", "isKey",
    "dataCategory", "summarizeBy", "sortByColumn", "isNullable", "isUnique", "type",
    "isNameInferred", "isDataTypeInferred",
)  # fmt: skip
_MEASURE_PROPS = ("formatString", "displayFolder", "isHidden", "dataCategory")


def to_bim(nodes: list[Node]) -> dict[str, Any]:
    """Turn parsed TMDL objects into a ``model.bim`` style dictionary."""
    model: dict[str, Any] = {"tables": [], "relationships": [], "cultures": []}
    database: dict[str, Any] = {"model": model}
    for node in nodes:
        if node.type == "database":
            database["name"] = node.name
            database.update(node.properties)
        elif node.type == "model":
            model["name"] = node.name
            for key, value in node.properties.items():
                model[key] = value
            if node.description:
                model["description"] = node.description
            model["annotations"] = _annotations(node)
        elif node.type == "table":
            model["tables"].append(_table(node))
        elif node.type == "relationship":
            model["relationships"].append(_relationship(node))
        elif node.type == "cultureInfo":
            model["cultures"].append(_culture(node))
        elif node.type == "role":
            model.setdefault("roles", []).append({"name": node.name})
        elif node.type == "perspective":
            model.setdefault("perspectives", []).append({"name": node.name})
        elif node.type == "expression":
            model.setdefault("expressions", []).append(
                {"name": node.name, "expression": node.expression}
            )
        elif node.type == "function":
            model.setdefault("functions", []).append(
                {"name": node.name, "expression": node.expression}
            )
        elif node.type == "dataSource":
            model.setdefault("dataSources", []).append({"name": node.name})
    order = [
        n.name.casefold() for n in nodes if n.type == "ref" and n.properties.get("kind") == "table"
    ]
    if order:
        rank = {name: i for i, name in enumerate(order)}
        model["tables"].sort(key=lambda t: rank.get(t["name"].casefold(), len(rank)))
    return database


def _common(node: Node, keys: tuple[str, ...]) -> dict[str, Any]:
    out: dict[str, Any] = {"name": node.name}
    for key in keys:
        if key in node.properties:
            out[key] = node.properties[key]
    if node.description:
        out["description"] = node.description
    annotations = _annotations(node)
    if annotations:
        out["annotations"] = annotations
    return out


def _annotations(node: Node) -> list[dict[str, Any]]:
    return [{"name": a.name, "value": a.expression or ""} for a in node.of("annotation")]


def _table(node: Node) -> dict[str, Any]:
    table = _common(node, _TABLE_PROPS)
    table["columns"] = []
    for col in node.of("column"):
        column = _common(col, _COLUMN_PROPS)
        if col.expression is not None:
            column["type"] = "calculated"
            column["expression"] = col.expression
        if col.of("variation"):
            column["variations"] = [{"name": v.name} for v in col.of("variation")]
        table["columns"].append(column)
    table["measures"] = []
    for meas in node.of("measure"):
        measure = _common(meas, _MEASURE_PROPS)
        measure["expression"] = meas.expression or ""
        for extra in ("formatStringDefinition", "detailRowsDefinition", "kpi"):
            if extra in meas.properties or meas.of(extra):
                measure[extra] = meas.properties.get(extra, True)
        table["measures"].append(measure)
    table["hierarchies"] = []
    for hier in node.of("hierarchy"):
        hierarchy = _common(hier, ("displayFolder", "isHidden"))
        hierarchy["levels"] = [
            {
                **_common(level, ("column", "ordinal")),
                "ordinal": int(level.properties.get("ordinal", i)),
            }
            for i, level in enumerate(hier.of("level"))
        ]
        table["hierarchies"].append(hierarchy)
    table["partitions"] = []
    for part in node.of("partition"):
        kind = part.expression or ""
        table["partitions"].append(
            {
                "name": part.name,
                "mode": part.properties.get("mode", "import"),
                "source": {
                    "type": {"calculated": "calculated", "m": "m", "entity": "entity"}.get(
                        kind, kind
                    ),
                    "expression": part.properties.get("source"),
                },
            }
        )
    if node.of("calculationGroup"):
        table["calculationGroup"] = {"name": node.name}
    if "refreshPolicy" in node.properties or node.of("refreshPolicy"):
        table["refreshPolicy"] = True
    return table


def _relationship(node: Node) -> dict[str, Any]:
    rel = _common(
        node,
        (
            "isActive", "crossFilteringBehavior", "fromCardinality", "toCardinality",
            "relyOnReferentialIntegrity", "securityFilteringBehavior", "joinOnDateBehavior",
        ),
    )  # fmt: skip
    for side in ("from", "to"):
        ref = node.properties.get(f"{side}Column")
        if not isinstance(ref, str):
            raise ModelError(f"relationship {node.name!r} has no {side}Column")
        table, column = split_ref(ref)
        rel[f"{side}Table"] = table
        rel[f"{side}Column"] = column
    return rel


def _culture(node: Node) -> dict[str, Any]:
    translations: dict[str, Any] = {}
    for block in node.of("translations"):
        for model in block.children:
            if model.type != "model":
                continue
            translations = {"model": _translated(model)}
    return {"name": node.name, "translations": translations}


def _translated(node: Node) -> dict[str, Any]:
    out: dict[str, Any] = {"name": node.name}
    for key in ("translatedCaption", "translatedDescription", "translatedDisplayFolder"):
        if key in node.properties:
            out[key] = node.properties[key]
    plural = {
        "table": "tables",
        "column": "columns",
        "measure": "measures",
        "hierarchy": "hierarchies",
        "level": "levels",
    }
    for child in node.children:
        group = plural.get(child.type)
        if group is not None:
            out.setdefault(group, []).append(_translated(child))
    return out
