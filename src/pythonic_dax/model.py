"""The semantic model: tables, relationships and measures, with Power BI's rules."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import polars as pl

from ._names import NameMap
from ._special import _orders
from .errors import DaxError, ModelError, NotSupportedError
from .expr import Keyword, _Omitted, coerce_any, columns_in, measures_in
from .objects import (
    Cardinality,
    Column,
    CrossFilter,
    Measure,
    Relationship,
    Table,
    duplicate_count,
)
from .sources import as_source
from .types import DaxType


class Model:
    """A Power BI style semantic model.

    >>> model = pdax.Model(description="Exam revenue model")
    >>> Exams = model.add_table("Exams", Parquet("exams.parquet"), description="One row per exam.")
    >>> Exams["VOID_FL"].set(description="A flag for voids.")
    """

    def __init__(self, *, description: str | None = None, culture: str = "en-US") -> None:
        self.description = description
        self.culture = culture
        self.annotations: dict[str, str] = {}
        self._tables: NameMap[Table] = NameMap("table")
        self._measures: NameMap[Measure] = NameMap("measure")
        self._relationships: list[Relationship] = []

    # --- tables ---------------------------------------------------------------

    def add_table(
        self,
        name: str,
        data: Any,
        *,
        columns: dict[str, dict[str, Any]] | None = None,
        description: str | None = None,
        display_name: str | None = None,
        is_hidden: bool = False,
        annotations: dict[str, str] | None = None,
    ) -> Table:
        """Add a table from a DataFrame, a LazyFrame or a source.

        ``data`` is a Polars or pandas DataFrame, or a source from
        :mod:`pythonic_dax.sources` (``Parquet``, ``CSV``, ``Database``, ``DatabaseURI``).
        ``columns`` sets per-column properties, including ``data_type`` to override
        the inferred DAX type::

            columns={"Price": {"data_type": "currency", "format_string": "$#,0.00"}}
        """
        _check_name(name, "table")
        if name in self._tables:
            raise ModelError(f"a table named {self._tables[name].name!r} already exists")

        column_props = {k: dict(v) for k, v in (columns or {}).items()}
        column_types = {
            col: DaxType(props.pop("data_type"))
            for col, props in column_props.items()
            if "data_type" in props
        }

        table = Table(self, name, as_source(data))
        table._load(column_types, first_load=True)
        for col_name in column_types:
            table[col_name]  # raises KeyError for a typo in `columns`
        for col_name, props in column_props.items():
            table[col_name].set(**props)
        table.set(
            description=description,
            display_name=display_name,
            is_hidden=is_hidden,
            annotations=dict(annotations or {}),
        )
        self._tables.add(name, table)
        return table

    @property
    def tables(self) -> list[Table]:
        return list(self._tables)

    def table(self, name: str) -> Table:
        """Look up a table by name, ignoring case."""
        return self._tables[name]

    def refresh(self) -> None:
        """Reload every table from its source, like a Power BI dataset refresh.

        Column types and metadata are kept. Relationships, sort-by columns and date
        tables are validated again, and nothing changes if validation fails.
        """
        saved = {t.name: t._data for t in self._tables}
        try:
            for table in self._tables:
                table._load({}, first_load=False)
            for rel in self._relationships:
                rel._invalidate()
                _check_keys(rel)
            for table in self._tables:
                for col in table.columns:
                    if col.sort_by_column is not None:
                        col.sort_by_column = col.sort_by_column
                if table.date_column is not None:
                    table.mark_as_date_table(table.date_column)
        except Exception:
            for table in self._tables:
                table._data = saved[table.name]
            for rel in self._relationships:
                rel._invalidate()
            raise

    # --- relationships ----------------------------------------------------------

    def add_relationship(
        self,
        from_column: Column,
        to_column: Column,
        *,
        cardinality: Cardinality | str = Cardinality.MANY_TO_ONE,
        cross_filter: CrossFilter | str = CrossFilter.SINGLE,
        active: bool = True,
        name: str | None = None,
        description: str | None = None,
        rely_on_referential_integrity: bool = False,
    ) -> Relationship:
        """Relate a column on the many side to a column on the one side.

        Follows Power BI's rules:

        * The one side can't have duplicate keys (text is compared without case).
          If the columns are given the wrong way round, they are swapped, as Power BI
          Desktop does.
        * Both columns must have the same data type.
        * One-to-one relationships always filter both ways.
        * Only one active filter path may exist between two tables. Add a second
          relationship with ``active=False`` and use it through ``USERELATIONSHIP``.
        """
        cardinality = Cardinality(cardinality)
        cross_filter = CrossFilter(cross_filter)

        for col in (from_column, to_column):
            if col.table.model is not self or self._tables.get(col.table.name) is not col.table:
                raise ModelError(f"{col.ref} is not a column of this model")
        if from_column.table is to_column.table:
            raise ModelError("a relationship must connect two different tables")
        if from_column.data_type is not to_column.data_type:
            raise ModelError(
                f"{from_column.ref} is {from_column.data_type.value} but {to_column.ref} is "
                f"{to_column.data_type.value}: related columns must have the same data type"
            )
        if cardinality is Cardinality.MANY_TO_MANY:
            raise NotSupportedError("many-to-many relationships")

        swap = (
            cardinality is Cardinality.MANY_TO_ONE
            and duplicate_count(to_column) > 0
            and duplicate_count(from_column) == 0
        )
        if swap:  # the columns were given one side first: flip them, as Power BI Desktop does
            from_column, to_column = to_column, from_column
        if cardinality is Cardinality.ONE_TO_ONE:
            cross_filter = CrossFilter.BOTH

        rel = Relationship(
            self,
            from_column,
            to_column,
            cardinality,
            cross_filter,
            active,
            name or f"{from_column.table.name}_{from_column.name}_{to_column.table.name}",
            rely_on_referential_integrity,
        )
        rel.set(description=description)
        _check_keys(rel)
        if active:
            self._check_ambiguity([*self._relationships, rel], rel)
        self._relationships.append(rel)
        return rel

    @property
    def relationships(self) -> list[Relationship]:
        return list(self._relationships)

    def set_active(self, relationship: Relationship, active: bool = True) -> None:
        """Make a relationship the active (default) one, or inactive."""
        if relationship not in self._relationships:
            raise ModelError("that relationship is not part of this model")
        if active and not relationship.is_active:
            relationship.is_active = True
            try:
                self._check_ambiguity(self._relationships, relationship)
            except ModelError:
                relationship.is_active = False
                raise
        else:
            relationship.is_active = active

    def _check_ambiguity(self, relationships: list[Relationship], new: Relationship) -> None:
        """Reject a second active filter path between any two tables, as Power BI does."""
        active = [r for r in relationships if r.is_active]

        same_pair = [
            r
            for r in active
            if r is not new
            and {r.from_table.name, r.to_table.name} == {new.from_table.name, new.to_table.name}
        ]
        if same_pair:
            raise ModelError(
                f"there is already an active relationship between {new.from_table.ref} and "
                f"{new.to_table.ref} ({same_pair[0].from_column.ref} to "
                f"{same_pair[0].to_column.ref}). Add this one with active=False and use it "
                "through USERELATIONSHIP"
            )

        # Filters flow from the one side to the many side, and back too when the
        # relationship filters both ways. Two different routes from one table to
        # another make the model ambiguous.
        edges: dict[str, list[str]] = defaultdict(list)
        for r in active:
            edges[r.to_table.name.casefold()].append(r.from_table.name.casefold())
            if r.cross_filter is CrossFilter.BOTH:
                edges[r.from_table.name.casefold()].append(r.to_table.name.casefold())

        for start in list(edges):
            counts: dict[str, int] = defaultdict(int)

            def walk(node: str, seen: frozenset[str], counts: dict[str, int] = counts) -> None:
                for nxt in edges.get(node, []):
                    if nxt in seen:
                        continue
                    counts[nxt] += 1
                    walk(nxt, seen | {nxt})

            walk(start, frozenset({start}))
            for target, n in counts.items():
                if n > 1:
                    src, dst = self._tables[start].name, self._tables[target].name
                    raise ModelError(
                        f"making this relationship active would create more than one filter "
                        f"path from {src!r} to {dst!r}, which Power BI doesn't allow. Add it "
                        "with active=False and use it through USERELATIONSHIP"
                    )

    # --- measures --------------------------------------------------------------

    def add_measure(
        self,
        name: str,
        expression: Any,
        *,
        table: Table | str | None = None,
        description: str | None = None,
        display_name: str | None = None,
        format_string: str | None = None,
        display_folder: str | None = None,
        is_hidden: bool = False,
        data_category: str | None = None,
        annotations: dict[str, str] | None = None,
    ) -> Measure:
        """Add a measure and return it, ready to use in other expressions.

        ``table`` is the measure's home table, as in Power BI. It defaults to the first
        table in the model. Measure names are unique across the model (ignoring case)
        and can't match a column name in the home table.
        """
        _check_name(name, "measure")
        if not self._tables:
            raise ModelError("add a table before adding measures: every measure has a home table")
        if isinstance(table, str):
            home = self._tables[table]
        else:
            home = table if table is not None else self.tables[0]
        if self._tables.get(home.name) is not home:
            raise ModelError(f"{home.ref} is not a table of this model")
        if name in self._measures:
            raise ModelError(f"a measure named {self._measures[name].name!r} already exists")
        if name in home:
            raise ModelError(f"{home.ref} already has a column named {home[name].name!r}")

        node = coerce_any(expression)
        if node.kind == "modifier":
            raise DaxError(f"a measure can't be a CALCULATE modifier: {node.to_dax()}")
        for col in columns_in(node):
            if col.table.model is not self:
                raise ModelError(f"{col.ref} belongs to a different model")
        measure = Measure(self, name, node, home)
        if any(m is measure for m in measures_in(node)):
            raise DaxError(f"A circular dependency was detected: {measure.ref}")
        measure.set(
            description=description,
            display_name=display_name,
            format_string=format_string,
            display_folder=display_folder,
            is_hidden=is_hidden,
            data_category=data_category,
            annotations=dict(annotations or {}),
        )
        self._measures.add(name, measure)
        return measure

    @property
    def measures(self) -> list[Measure]:
        return list(self._measures)

    def __getitem__(self, name: str) -> Measure:
        """Look up a measure by name, ignoring case: ``model["Total Sales"]``."""
        return self._measures[name]

    def to_dax_query(
        self,
        table_expression: Any,
        order_by: Sequence[Any] = (),
    ) -> str:
        """The DAX query that :meth:`evaluate` runs, to paste into Power BI Desktop.

        Includes a ``DEFINE MEASURE`` for every measure the query uses, so it runs as is
        in DAX query view. ``order_by`` lists expressions, each optionally followed by
        ``ASC`` or ``DESC``.
        """
        node = coerce_any(table_expression)
        if node.kind != "table":
            raise TypeError("EVALUATE needs a table expression, such as SUMMARIZECOLUMNS(...)")
        lines: list[str] = []
        order = {id(m): i for i, m in enumerate(self._measures)}
        measures = sorted(measures_in(node), key=lambda m: order.get(id(m), len(order)))
        if measures:
            lines.append("DEFINE")
            lines += [f"    {m.definition()}" for m in measures]
        lines += ["EVALUATE", f"    {node.to_dax()}"]
        if order_by:
            keys = _orders("ORDER BY", tuple(order_by))
            parts: list[str] = []
            for key in keys:
                if isinstance(key, Keyword):
                    parts[-1] += f" {key.name}"
                elif not isinstance(key, _Omitted):
                    parts.append(key.to_dax())
            lines += ["ORDER BY", "    " + ", ".join(parts)]
        return "\n".join(lines)

    # --- evaluation (Phase 3) ---------------------------------------------------

    def evaluate(self, table_expression: Any) -> Any:
        """Evaluate a table expression, like DAX ``EVALUATE``, and return a DataFrame."""
        raise NotSupportedError("Model.evaluate", phase=3)

    # --- metadata ---------------------------------------------------------------

    def describe(self) -> dict[str, pl.DataFrame]:
        """The model's metadata as tables: ``tables``, ``columns``, ``measures``, ``relationships``.

        Useful for documentation pages and for building report field pickers.
        """
        tables = pl.DataFrame(
            [
                {
                    "table": t.name,
                    "display_name": t.display_name,
                    "description": t.description,
                    "rows": len(t),
                    "is_hidden": t.is_hidden,
                    "is_date_table": t.is_date_table,
                    "source": t.source.describe(),
                }
                for t in self._tables
            ],
            schema={
                "table": pl.String, "display_name": pl.String, "description": pl.String,
                "rows": pl.Int64, "is_hidden": pl.Boolean, "is_date_table": pl.Boolean,
                "source": pl.String,
            },
        )  # fmt: skip
        columns = pl.DataFrame(
            [
                {
                    "table": c.table.name,
                    "column": c.name,
                    "display_name": c.display_name,
                    "description": c.description,
                    "data_type": c.data_type.value,
                    "format_string": c.format_string,
                    "display_folder": c.display_folder,
                    "is_hidden": c.is_hidden,
                    "data_category": c.data_category,
                    "summarize_by": c.summarize_by.value,
                    "sort_by_column": (
                        c.sort_by_column.name if c.sort_by_column is not None else None
                    ),
                }
                for t in self._tables
                for c in t.columns
            ],
            schema={
                "table": pl.String, "column": pl.String, "display_name": pl.String,
                "description": pl.String, "data_type": pl.String, "format_string": pl.String,
                "display_folder": pl.String, "is_hidden": pl.Boolean,
                "data_category": pl.String, "summarize_by": pl.String,
                "sort_by_column": pl.String,
            },
        )  # fmt: skip
        measures = pl.DataFrame(
            [
                {
                    "measure": m.name,
                    "table": m.table.name,
                    "display_name": m.display_name,
                    "description": m.description,
                    "format_string": m.format_string,
                    "display_folder": m.display_folder,
                    "is_hidden": m.is_hidden,
                }
                for m in self._measures
            ],
            schema={
                "measure": pl.String, "table": pl.String, "display_name": pl.String,
                "description": pl.String, "format_string": pl.String,
                "display_folder": pl.String, "is_hidden": pl.Boolean,
            },
        )  # fmt: skip
        relationships = pl.DataFrame(
            [
                {
                    "name": r.name,
                    "from": r.from_column.ref,
                    "to": r.to_column.ref,
                    "cardinality": r.cardinality.value,
                    "cross_filter": r.cross_filter.value,
                    "is_active": r.is_active,
                    "description": r.description,
                    "unmatched_rows": r.unmatched_rows,
                }
                for r in self._relationships
            ],
            schema={
                "name": pl.String, "from": pl.String, "to": pl.String,
                "cardinality": pl.String, "cross_filter": pl.String, "is_active": pl.Boolean,
                "description": pl.String, "unmatched_rows": pl.Int64,
            },
        )  # fmt: skip
        return {
            "tables": tables,
            "columns": columns,
            "measures": measures,
            "relationships": relationships,
        }

    def __repr__(self) -> str:
        return (
            f"Model({len(self._tables)} tables, {len(self._relationships)} relationships, "
            f"{len(self._measures)} measures)"
        )


def _check_name(name: str, kind: str) -> None:
    if not isinstance(name, str) or not name.strip():
        raise ModelError(f"a {kind} name can't be empty")
    if name != name.strip():
        raise ModelError(f"a {kind} name can't start or end with spaces: {name!r}")


def _check_keys(rel: Relationship) -> None:
    to_dupes = duplicate_count(rel.to_column)
    if to_dupes:
        raise ModelError(
            f"{rel.to_column.ref} has duplicate values, so it can't be the one side of a "
            "relationship. Both columns have duplicates: many-to-many relationships are "
            "not supported yet"
            if duplicate_count(rel.from_column)
            else f"{rel.to_column.ref} has duplicate values, so it can't be the one side of a "
            "relationship"
        )
    if rel.cardinality is Cardinality.ONE_TO_ONE and duplicate_count(rel.from_column):
        raise ModelError(
            f"{rel.from_column.ref} has duplicate values, so the relationship can't be one-to-one"
        )
