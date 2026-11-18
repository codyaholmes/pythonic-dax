"""Data sources for model tables.

Like Power BI's Import mode, a table's data is read from its source once, held in
memory, and read again only on :meth:`Model.refresh`. Every source returns a Polars
DataFrame.

>>> from pythonic_dax.sources import Parquet, Database
>>> model.add_table("Sales", Parquet("data/sales/*.parquet"))
>>> model.add_table("Product", Database("SELECT * FROM dim_product", connection))
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import polars as pl

from .errors import ModelError

PathLike = str | Path


class Source(ABC):
    """Where a table's data comes from."""

    @abstractmethod
    def load(self) -> pl.DataFrame:
        """Read the data. Called when the table is added and on every refresh."""

    def describe(self) -> str:
        """A short, human-readable description of the source."""
        return type(self).__name__


@dataclass(frozen=True)
class DataFrameSource(Source):
    """An in-memory Polars or pandas DataFrame, or a Polars LazyFrame.

    Refreshing re-reads the same object (and re-runs a LazyFrame's query).
    """

    data: Any

    def load(self) -> pl.DataFrame:
        data = self.data
        if isinstance(data, pl.DataFrame):
            return data
        if isinstance(data, pl.LazyFrame):
            return data.collect()
        if _is_pandas(data):
            return pl.from_pandas(data)
        raise ModelError(f"can't make a table from {type(data).__name__}")

    def describe(self) -> str:
        return f"in-memory {type(self.data).__name__}"


@dataclass(frozen=True)
class Parquet(Source):
    """One or more Parquet files: a path, a glob such as ``"sales/*.parquet"``, or a list.

    Cloud paths (``s3://``, ``az://``, ``gs://``) work too; pass credentials in
    ``storage_options``. Only the listed ``columns`` are read, if given.
    """

    path: PathLike | Sequence[PathLike]
    columns: Sequence[str] | None = None
    storage_options: dict[str, Any] | None = None
    hive_partitioning: bool | None = None

    def load(self) -> pl.DataFrame:
        source: Any = self.path if isinstance(self.path, str | Path) else list(self.path)
        frame = pl.scan_parquet(
            source,
            storage_options=self.storage_options,
            hive_partitioning=self.hive_partitioning,
        )
        if self.columns is not None:
            frame = frame.select(list(self.columns))
        return frame.collect()

    def describe(self) -> str:
        return f"Parquet {self.path}"


@dataclass(frozen=True)
class CSV(Source):
    """A CSV file. Extra keyword arguments go to :func:`polars.read_csv`."""

    path: PathLike
    options: dict[str, Any] = field(default_factory=dict)

    def load(self) -> pl.DataFrame:
        return pl.read_csv(self.path, try_parse_dates=True, **self.options)

    def describe(self) -> str:
        return f"CSV {self.path}"


@dataclass(frozen=True)
class Database(Source):
    """A SQL query run through an open connection.

    ``connection`` is any DBAPI2 connection (sqlite3, pyodbc, psycopg, duckdb,
    snowflake-connector) or a SQLAlchemy engine or connection. Uses
    :func:`polars.read_database`.
    """

    query: str
    connection: Any
    options: dict[str, Any] = field(default_factory=dict)

    def load(self) -> pl.DataFrame:
        frame: pl.DataFrame = pl.read_database(self.query, self.connection, **self.options)
        return frame

    def describe(self) -> str:
        return f"database query: {_shorten(self.query)}"


@dataclass(frozen=True)
class DatabaseURI(Source):
    """A SQL query run against a connection string such as ``postgresql://...``.

    Uses :func:`polars.read_database_uri`, whose ``connectorx`` and ``adbc`` engines
    are faster than :class:`Database` for large pulls. The engine's package must be
    installed separately.
    """

    query: str
    uri: str
    engine: str = "connectorx"
    options: dict[str, Any] = field(default_factory=dict)

    def load(self) -> pl.DataFrame:
        frame: pl.DataFrame = pl.read_database_uri(  # type: ignore[call-overload]
            self.query, self.uri, engine=self.engine, **self.options
        )
        return frame

    def describe(self) -> str:
        return f"database query: {_shorten(self.query)}"


def as_source(data: Any) -> Source:
    """Wrap a DataFrame-like object in a source. Sources pass through unchanged."""
    if isinstance(data, Source):
        return data
    if isinstance(data, pl.DataFrame | pl.LazyFrame) or _is_pandas(data):
        return DataFrameSource(data)
    raise ModelError(
        f"can't make a table from {type(data).__name__}. Pass a DataFrame or a source "
        "such as pythonic_dax.sources.Parquet or Database."
    )


def _is_pandas(data: Any) -> bool:
    return type(data).__module__.startswith("pandas") and type(data).__name__ == "DataFrame"


def _shorten(text: str, limit: int = 60) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"
