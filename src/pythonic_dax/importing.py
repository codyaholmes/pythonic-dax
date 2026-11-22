"""Import a Power BI semantic model from a TMDL folder or a ``model.bim`` file.

The model's structure comes from the file: tables, columns and their types,
relationships (active and inactive), measures, calculated columns and calculated
tables (their DAX is parsed into the library's own expressions), hierarchies, and
metadata such as descriptions, format strings, display folders, hidden flags, sort-by
columns and the date table mark.

The data does not come from the file. Power BI loads it with Power Query, which this
library doesn't run, so you say where each table's rows come from::

    model = pdax.from_tmdl(
        "Sales.SemanticModel",
        data={
            "Sales": Parquet("data/sales/*.parquet"),
            "Product": Database("SELECT * FROM dim_product", connection),
        },
    )

or pass a function that returns the source for a table name::

    model = pdax.from_tmdl(
        "Sales.SemanticModel", data=lambda table: Parquet(f"data/{table}.parquet")
    )

A source's columns are matched to the model's by each column's ``sourceColumn`` (the
name Power Query gave it), or else by the model column's own name. Columns the model
doesn't define are dropped, and each column is stored with the model's data type.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import polars as pl

from . import _tmdl
from ._names import NameMap
from ._parser import Converter, ModelResolver, parse_ast
from .errors import DaxError, ModelError, NotSupportedError
from .expr import UnsupportedDax, coerce_any
from .model import Model
from .objects import DATA_CATEGORIES, Column, Measure, Table
from .sources import Source, as_source
from .types import DaxType

__all__ = ["from_bim", "from_tmdl"]

DataArg = Mapping[str, Any] | Callable[[str], Any]

_DATA_TYPES = {
    "int64": DaxType.INTEGER,
    "double": DaxType.DECIMAL,
    "decimal": DaxType.CURRENCY,
    "datetime": DaxType.DATETIME,
    "string": DaxType.STRING,
    "boolean": DaxType.BOOLEAN,
}

_SUMMARIZE_BY = {
    "default": "default", "none": "none", "sum": "sum", "min": "min", "max": "max",
    "count": "count", "average": "average", "distinctcount": "distinct_count",
}  # fmt: skip

_AUTO_DATE = re.compile(r"^(LocalDateTable|DateTableTemplate)_[0-9a-fA-F-]+$")


def from_tmdl(
    path: str | Path,
    data: DataArg,
    *,
    culture: str | None = None,
    skip_tables: Sequence[str] = (),
    overrides: Mapping[str, Any] | None = None,
) -> Model:
    """Build a model from a TMDL folder (a ``.SemanticModel`` folder or its ``definition``).

    ``data`` gives each table's rows (see the module docs). ``culture`` names a
    translation, such as ``"es-ES"``, whose captions become display names.
    ``skip_tables`` leaves tables out, along with their relationships. ``overrides``
    replaces the DAX of a measure (by name), a calculated column (``"Table[Column]"``)
    or a calculated table (by name) with a Python expression or other DAX text.
    """
    return build(_tmdl.read_folder(path), data, culture, skip_tables, overrides)


def from_bim(
    path: str | Path,
    data: DataArg,
    *,
    culture: str | None = None,
    skip_tables: Sequence[str] = (),
    overrides: Mapping[str, Any] | None = None,
) -> Model:
    """Build a model from a ``model.bim`` file (the JSON model of Analysis Services and
    older Power BI projects). The options are the same as :func:`from_tmdl`."""
    raw = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    return build(raw, data, culture, skip_tables, overrides)


# --- building ----------------------------------------------------------------------


def build(
    raw: dict[str, Any],
    data: DataArg,
    culture: str | None = None,
    skip_tables: Sequence[str] = (),
    overrides: Mapping[str, Any] | None = None,
) -> Model:
    """Build a model from a ``model.bim`` style dictionary."""
    spec = raw.get("model", raw)
    if not isinstance(spec, dict) or "tables" not in spec:
        raise ModelError("this isn't a semantic model: it has no tables")
    return _Importer(spec, data, culture, skip_tables, overrides or {}).run()


class _Importer:
    def __init__(
        self,
        spec: dict[str, Any],
        data: DataArg,
        culture: str | None,
        skip_tables: Sequence[str],
        overrides: Mapping[str, Any],
    ) -> None:
        self.spec = spec
        self.data = data
        self.culture = culture
        self.overrides = {_key(k): v for k, v in overrides.items()}
        self.used_overrides: set[str] = set()
        self.model = Model(
            description=_text(spec.get("description")),
            culture=spec.get("culture") or "en-US",
        )
        self.model.annotations = _annotation_dict(spec)
        self.notes = self.model.import_notes
        self.skip = {s.casefold() for s in skip_tables}
        self.tables: NameMap[dict[str, Any]] = NameMap("table")
        self.base = ModelResolver(self.model)
        # Objects still to build, by key: ("measure", name), ("column", "t[c]"), ("table", t)
        self.pending: dict[tuple[str, str], tuple[dict[str, Any], dict[str, Any] | None]] = {}
        self.building: list[str] = []
        self.building_keys: list[tuple[str, str]] = []
        self.relationships: list[dict[str, Any]] = []

    # --- the import, step by step ---

    def run(self) -> Model:
        self._check_model()
        for table in self.spec["tables"]:
            name = table["name"]
            if name.casefold() in self.skip:
                self.notes.append(f"skipped table {name!r}, as asked")
                continue
            if _AUTO_DATE.match(name):
                self.notes.append(f"skipped the automatic date table {name!r}")
                continue
            self._check_table(table)
            self.tables.add(name, table)
        unknown = [s for s in self.skip if s not in self.tables and not self._in_spec(s)]
        if unknown:
            raise ModelError(f"skip_tables names tables the model doesn't have: {unknown}")

        self._load_data_tables()
        for table in self.tables:
            if _is_calculated_table(table):
                self.pending[("table", table["name"].casefold())] = (table, None)
            for column in table.get("columns", []):
                if column.get("type") == "calculated":
                    ref = f"{table['name']}[{column['name']}]".casefold()
                    self.pending[("column", ref)] = (column, table)
            for measure in table.get("measures", []):
                self.pending[("measure", measure["name"].casefold())] = (measure, table)

        self.relationships = [r for r in self.spec.get("relationships", []) if self._keep(r)]
        self._add_ready_relationships()
        for pending in list(self.pending):
            if pending in self.pending:
                self._build(pending)
        self._add_ready_relationships()
        if self.relationships:
            names = [_rel_label(r) for r in self.relationships]
            raise ModelError(f"relationships whose columns don't exist: {names}")

        for table in self.tables:
            self._finish_table(table)
        self._apply_translations()
        unused = set(self.overrides) - self.used_overrides
        if unused:
            raise ModelError(f"overrides name objects the model doesn't have: {sorted(unused)}")
        return self.model

    def _in_spec(self, name: str) -> bool:
        return any(t["name"].casefold() == name for t in self.spec["tables"])

    def _check_model(self) -> None:
        if self.spec.get("functions"):
            names = [f["name"] for f in self.spec["functions"]]
            raise NotSupportedError(f"DAX user-defined functions ({', '.join(names)})")
        for kind in ("roles", "perspectives"):
            for item in self.spec.get(kind, []):
                label = "role" if kind == "roles" else "perspective"
                self.notes.append(
                    f"the {label} {item['name']!r} was not applied: queries run as the model owner"
                )

    def _check_table(self, table: dict[str, Any]) -> None:
        name = table["name"]
        if "calculationGroup" in table:
            raise NotSupportedError(
                f"calculation groups (table {name!r}); skip it with skip_tables"
            )
        for part in table.get("partitions", []):
            mode = str(part.get("mode", "import")).casefold()
            if mode not in ("import", "default"):
                raise NotSupportedError(
                    f"{mode} storage mode (table {name!r}): only Import tables are supported"
                )
        for column in table.get("columns", []):
            if column.get("type") == "rowNumber":
                continue
            data_type = str(column.get("dataType", "string")).casefold()
            if data_type not in _DATA_TYPES and column.get("type") != "calculated":
                raise NotSupportedError(
                    f"the {column.get('dataType')} data type ({name}[{column['name']}])"
                )

    # --- data tables ---

    def _load_data_tables(self) -> None:
        data_tables = [t for t in self.tables if not _is_calculated_table(t)]
        if isinstance(self.data, Mapping):
            given = NameMap[Any]("table")
            for key, value in self.data.items():
                given.add(key, value)
            for t in data_tables:  # a table with no data columns, such as a measures table
                if t["name"] not in given and not any(map(_is_data_column, t.get("columns", []))):
                    given.add(t["name"], None)
            missing = [t["name"] for t in data_tables if t["name"] not in given]
            if missing:
                raise ModelError(f"no data given for these tables: {missing}")
            extra = [
                k
                for k in given.names()
                if k not in self.tables or _is_calculated_table(self.tables[k])
            ]
            if extra:
                raise ModelError(
                    f"data names tables that aren't import tables of the model: {extra}"
                )
            sources = {t["name"]: given[t["name"]] for t in data_tables}
        elif callable(self.data):
            sources = {t["name"]: self.data(t["name"]) for t in data_tables}
        else:
            raise TypeError("data must be a dict of table name to source, or a function")

        for table in data_tables:
            name = table["name"]
            columns = [c for c in table.get("columns", []) if _is_data_column(c)]
            pairs = [(c["name"], str(c.get("sourceColumn") or c["name"])) for c in columns]
            types = {c["name"]: {"data_type": _data_type(c, name)} for c in columns}
            given_data = sources[name]
            if given_data is None:  # no rows: an empty table with the model's columns
                given_data = pl.DataFrame(
                    schema={c: t["data_type"].polars_dtype for c, t in types.items()}
                )
                pairs = [(c, c) for c, _ in pairs]
            source = _ModelColumns(as_source(given_data), name, pairs)
            added = self.model.add_table(name, source, columns=types)
            for column in columns:
                self._column_metadata(added[column["name"]], column)

    # --- relationships ---

    def _keep(self, rel: dict[str, Any]) -> bool:
        ends = (rel["fromTable"], rel["toTable"])
        if any(_AUTO_DATE.match(t) for t in ends):
            return False
        if any(t.casefold() in self.skip for t in ends):
            self.notes.append(f"skipped the relationship {_rel_label(rel)}: its table was skipped")
            return False
        if str(rel.get("joinOnDateBehavior", "")).casefold() == "datepartonly":
            raise NotSupportedError(f"relationships on the date part only ({_rel_label(rel)})")
        return True

    def _add_ready_relationships(self) -> None:
        waiting: list[dict[str, Any]] = []
        for rel in self.relationships:
            ends = [
                self._existing_column(rel[f"{s}Table"], rel[f"{s}Column"]) for s in ("from", "to")
            ]
            if ends[0] is None or ends[1] is None:
                waiting.append(rel)
                continue
            self._add_relationship(rel, ends[0], ends[1])
        self.relationships = waiting

    def _existing_column(self, table: str, column: str) -> Column | None:
        t = self.model._tables.get(table)
        return None if t is None else t._columns.get(column)

    def _add_relationship(self, rel: dict[str, Any], from_col: Column, to_col: Column) -> None:
        many_from = str(rel.get("fromCardinality", "many")).casefold() == "many"
        many_to = str(rel.get("toCardinality", "one")).casefold() == "many"
        if many_from and many_to:
            raise NotSupportedError(f"many-to-many relationships ({_rel_label(rel)})")
        if not many_from and many_to:  # written one side first
            from_col, to_col = to_col, from_col
        cardinality = "many_to_one" if many_from or many_to else "one_to_one"
        both = str(rel.get("crossFilteringBehavior", "oneDirection")).casefold() == "bothdirections"
        added = self.model.add_relationship(
            from_col,
            to_col,
            cardinality=cardinality,
            cross_filter="both" if both else "single",
            active=bool(rel.get("isActive", True)),
            name=rel.get("name"),
            rely_on_referential_integrity=bool(rel.get("relyOnReferentialIntegrity", False)),
        )
        added.set(annotations=_annotation_dict(rel))

    # --- calculated objects and measures, built on demand ---

    def table(self, name: str) -> Table | None:
        key = ("table", name.casefold())
        if key in self.pending:
            self._build(key)
        elif key in self.building_keys:
            self._circular(key)
        return self.base.table(name)

    def column(self, table: Table, name: str) -> Column | None:
        key = ("column", f"{table.name}[{name}]".casefold())
        if key in self.pending:
            self._build(key)
        elif key in self.building_keys:
            self._circular(key)
        return self.base.column(table, name)

    def measure(self, name: str) -> Measure | None:
        key = ("measure", name.casefold())
        if key in self.pending:
            self._build(key)
        elif key in self.building_keys:
            self._circular(key)
        return self.base.measure(name)

    def _circular(self, key: tuple[str, str]) -> None:
        chain = ", ".join([*self.building, self.building[self.building_keys.index(key)]])
        raise DaxError(f"A circular dependency was detected: {chain}")

    def _build(self, key: tuple[str, str]) -> None:
        item, table = self.pending.pop(key)
        kind = key[0]
        label = item["name"] if kind != "column" else f"{table['name']}[{item['name']}]"  # type: ignore[index]
        self.building.append(label)
        self.building_keys.append(key)
        try:
            if kind == "measure":
                self._build_measure(item, table)  # type: ignore[arg-type]
            elif kind == "column":
                self._build_column(item, table)  # type: ignore[arg-type]
            else:
                self._build_table(item)
        finally:
            self.building.pop()
            self.building_keys.pop()
        self._add_ready_relationships()

    def _expression(self, label: str, text: str, home: Table | None = None) -> Any:
        """The expression for an object: its override, or its DAX parsed."""
        key = _key(label)
        if key in self.overrides:
            self.used_overrides.add(key)
            value = self.overrides[key]
            if not isinstance(value, str):
                return value
            text = value
        return Converter(text, self, home).node(parse_ast(text))

    def _build_measure(self, item: dict[str, Any], table: dict[str, Any]) -> None:
        name = item["name"]
        text = _text(item.get("expression")) or ""
        home = self.model.table(table["name"])
        try:
            expression = self._expression(name, text)
            measure = self.model.add_measure(name, expression, table=home)
        except (NotSupportedError, DaxError) as exc:
            # Kept, so the rest of the model imports. Using it raises NotSupportedError.
            if isinstance(exc, NotSupportedError):
                reason = f"{exc.feature} (in the measure [{name}])"
            else:
                reason = f"the measure [{name}], whose DAX gives an error here ({exc})"
            self.notes.append(f"the measure [{name}] can't be evaluated: {exc}")
            measure = self.model.add_measure(name, UnsupportedDax(text, reason), table=home)
        self._measure_metadata(measure, item)

    def _build_column(self, item: dict[str, Any], table: dict[str, Any]) -> None:
        home = self.model.table(table["name"])
        label = f"{table['name']}[{item['name']}]"
        data_type = _DATA_TYPES.get(str(item.get("dataType", "")).casefold())
        try:
            expression = self._expression(label, _text(item.get("expression")) or "", home)
            column = self.model.add_column(home, item["name"], expression, data_type=data_type)
        except NotSupportedError as exc:
            raise NotSupportedError(
                f"{exc.feature} (in the calculated column {label}; replace its DAX with "
                "overrides=, or leave its table out with skip_tables=)"
            ) from exc
        self._column_metadata(column, item)

    def _build_table(self, item: dict[str, Any]) -> None:
        name = item["name"]
        part = next(p for p in item["partitions"] if _source_type(p) == "calculated")
        text = _text(part["source"].get("expression")) or ""
        try:
            expression = coerce_any(self._expression(name, text))
        except NotSupportedError as exc:
            raise NotSupportedError(
                f"{exc.feature} (in the calculated table {name!r}; replace its DAX with "
                "overrides=, or leave it out with skip_tables=)"
            ) from exc
        columns: dict[str, dict[str, Any]] = {}
        for c in item.get("columns", []):
            if c.get("type") in ("calculated", "rowNumber"):
                continue
            props: dict[str, Any] = {}
            if c.get("sourceColumn"):
                props["source_column"] = c["sourceColumn"]
            data_type = _DATA_TYPES.get(str(c.get("dataType", "")).casefold())
            if data_type is not None:
                props["data_type"] = data_type
            columns[c["name"]] = props
        added = self.model.add_calculated_table(name, expression, columns=columns)
        defined = {c["name"].casefold() for c in item.get("columns", [])}
        extra = [c.name for c in added.columns if c.name.casefold() not in defined]
        if extra:
            raise ModelError(
                f"the calculated table {name!r} returns columns the model doesn't define: {extra}"
            )
        for c in item.get("columns", []):
            if c.get("type") not in ("calculated", "rowNumber"):
                self._column_metadata(added[c["name"]], c)

    # --- metadata ---

    def _column_metadata(self, column: Column, item: dict[str, Any]) -> None:
        category = item.get("dataCategory")
        annotations = _annotation_dict(item)
        if category is not None and category not in DATA_CATEGORIES:
            annotations["dataCategory"] = category  # e.g. Years: kept, but not a DAX matter
            category = None
        summarize = item.get("summarizeBy")
        column.set(
            description=_text(item.get("description")),
            format_string=item.get("formatString"),
            display_folder=item.get("displayFolder"),
            is_hidden=bool(item.get("isHidden", False)),
            data_category=category,
            is_key=bool(item.get("isKey", False)),
            is_nullable=bool(item.get("isNullable", True)),
            annotations=annotations,
        )
        if summarize is not None:
            column.summarize_by = _SUMMARIZE_BY.get(str(summarize).casefold(), "default")

    def _measure_metadata(self, measure: Measure, item: dict[str, Any]) -> None:
        annotations = _annotation_dict(item)
        if item.get("formatStringDefinition"):
            definition = item["formatStringDefinition"]
            text = definition if isinstance(definition, str) else "(dynamic)"
            annotations["formatStringDefinition"] = text
            self.notes.append(
                f"the measure [{measure.name}] has a dynamic format string, which is not applied"
            )
        category = item.get("dataCategory")
        measure.set(
            description=_text(item.get("description")),
            format_string=item.get("formatString"),
            display_folder=item.get("displayFolder"),
            is_hidden=bool(item.get("isHidden", False)),
            data_category=category if category in DATA_CATEGORIES else None,
            annotations=annotations,
        )

    def _finish_table(self, item: dict[str, Any]) -> None:
        table = self.model.table(item["name"])
        table.set(
            description=_text(item.get("description")),
            is_hidden=bool(item.get("isHidden", False)) or bool(item.get("isPrivate", False)),
            annotations=_annotation_dict(item),
        )
        for c in item.get("columns", []):
            sort_by = c.get("sortByColumn")
            if sort_by and c["name"] in table:
                table[c["name"]].sort_by_column = table[sort_by]
        for h in item.get("hierarchies", []):
            levels = sorted(h.get("levels", []), key=lambda lv: lv.get("ordinal", 0))
            table.add_hierarchy(
                h["name"],
                [(lv["name"], lv["column"]) for lv in levels],
                description=_text(h.get("description")),
                display_folder=h.get("displayFolder"),
                is_hidden=bool(h.get("isHidden", False)),
                annotations=_annotation_dict(h),
            )
        if str(item.get("dataCategory", "")).casefold() == "time":
            keys = [c for c in table.columns if c.is_key and c.data_type is DaxType.DATETIME]
            if keys:
                table.mark_as_date_table(keys[0])

    def _apply_translations(self) -> None:
        if self.culture is None:
            return
        cultures = {c["name"].casefold(): c for c in self.spec.get("cultures", [])}
        found = cultures.get(self.culture.casefold())
        if found is None:
            raise ModelError(f"the model has no translations for the culture {self.culture!r}")
        model = (found.get("translations") or {}).get("model") or {}
        for t in model.get("tables", []):
            table = self.model._tables.get(t["name"])
            if table is None:
                continue
            _caption(table, t)
            for c in t.get("columns", []):
                col = table._columns.get(c["name"])
                if col is not None:
                    _caption(col, c)
            for m in t.get("measures", []):
                measure = self.model._measures.get(m["name"])
                if measure is not None:
                    _caption(measure, m)
            for h in t.get("hierarchies", []):
                hier = table._hierarchies.get(h["name"])
                if hier is not None:
                    _caption(hier, h)


class _ModelColumns(Source):
    """A table's source, narrowed to the model's columns and renamed to their names."""

    def __init__(self, inner: Source, table: str, pairs: list[tuple[str, str]]) -> None:
        self.inner = inner
        self.table = table
        self.pairs = pairs  # (model column, source column)

    def load(self) -> pl.DataFrame:
        frame = self.inner.load()
        exprs = []
        missing = []
        for name, source in self.pairs:
            if source in frame.columns:
                exprs.append(pl.col(source).alias(name))
            elif name in frame.columns:
                exprs.append(pl.col(name))
            else:
                missing.append(name if name == source else f"{name} (source column {source})")
        if missing:
            raise ModelError(f"table {self.table!r}: the data has no column for {missing}")
        return frame.select(exprs)

    def describe(self) -> str:
        return self.inner.describe()


# --- helpers ----------------------------------------------------------------------


def _key(label: str) -> str:
    return label.strip().casefold()


def _text(value: Any) -> str | None:
    """A text property, which a .bim file may split into a list of lines."""
    if value is None:
        return None
    if isinstance(value, list):
        return "\n".join(str(v) for v in value)
    return str(value)


def _annotation_dict(item: dict[str, Any]) -> dict[str, str]:
    return {a["name"]: _text(a.get("value")) or "" for a in item.get("annotations", []) or []}


def _source_type(partition: dict[str, Any]) -> str:
    return str((partition.get("source") or {}).get("type", "m")).casefold()


def _is_calculated_table(table: dict[str, Any]) -> bool:
    return any(_source_type(p) == "calculated" for p in table.get("partitions", []))


def _is_data_column(column: dict[str, Any]) -> bool:
    return column.get("type", "data") in ("data", None)


def _data_type(column: dict[str, Any], table: str) -> DaxType:
    raw = str(column.get("dataType", "string")).casefold()
    return _DATA_TYPES[raw]


def _rel_label(rel: dict[str, Any]) -> str:
    return f"{rel['fromTable']}[{rel['fromColumn']}] -> {rel['toTable']}[{rel['toColumn']}]"


def _caption(obj: Any, item: dict[str, Any]) -> None:
    caption = item.get("translatedCaption")
    if caption:
        obj.display_name = caption
