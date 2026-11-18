"""Case-insensitive name lookup, matching how DAX resolves table, column and measure names."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Generic, TypeVar

V = TypeVar("V")


class NameMap(Generic[V]):
    """An ordered mapping whose keys are compared without regard to case."""

    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._items: dict[str, tuple[str, V]] = {}

    @staticmethod
    def _key(name: str) -> str:
        return name.casefold()

    def add(self, name: str, value: V) -> None:
        key = self._key(name)
        if key in self._items:
            existing = self._items[key][0]
            raise KeyError(f"a {self._kind} named {existing!r} already exists")
        self._items[key] = (name, value)

    def remove(self, name: str) -> V:
        return self._items.pop(self._key(name))[1]

    def get(self, name: str) -> V | None:
        found = self._items.get(self._key(name))
        return None if found is None else found[1]

    def __getitem__(self, name: str) -> V:
        found = self._items.get(self._key(name))
        if found is None:
            raise KeyError(f"no {self._kind} named {name!r}")
        return found[1]

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and self._key(name) in self._items

    def __iter__(self) -> Iterator[V]:
        return (value for _, value in self._items.values())

    def __len__(self) -> int:
        return len(self._items)

    def names(self) -> list[str]:
        return [name for name, _ in self._items.values()]
