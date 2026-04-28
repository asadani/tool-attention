"""LazySchemaLoader: on-demand full-schema fetching with LRU caching.

Full JSON schemas never sit in the model's context unless the
IntentRouter has placed their tool in the active set for the
current turn. Schemas can be loaded from disk, a remote HTTP
endpoint, or any callable registered by the host.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from collections.abc import Callable
from typing import cast


class LazySchemaLoader:
    """LRU-cached loader for full JSON tool schemas.

    By default loads from `registry_path / "<tool_id>.json"`. A custom
    `fetcher` callable can be supplied to load from an MCP server,
    a database, or any other source.
    """

    def __init__(
        self,
        registry_path: Path,
        capacity: int = 256,
        fetcher: Callable[[str], dict[str, object]] | None = None,
    ) -> None:
        self.registry_path: Path = Path(registry_path)
        self.capacity: int = int(capacity)
        self._fetcher: Callable[[str], dict[str, object]] | None = fetcher
        self._cache: OrderedDict[str, dict[str, object]] = OrderedDict()

    def get(self, tool_id: str) -> dict[str, object]:
        if tool_id in self._cache:
            self._cache.move_to_end(tool_id)
            return self._cache[tool_id]
        schema = (
            self._fetcher(tool_id)
            if self._fetcher is not None
            else self._load_from_disk(tool_id)
        )
        self._cache[tool_id] = schema
        if len(self._cache) > self.capacity:
            _ = self._cache.popitem(last=False)
        return schema

    def _load_from_disk(self, tool_id: str) -> dict[str, object]:
        path = self.registry_path / f"{tool_id}.json"
        if not path.exists():
            raise KeyError(f"no schema registered for tool {tool_id!r}")
        return cast(dict[str, object], json.loads(path.read_text()))

    def clear(self) -> None:
        self._cache.clear()

    def __contains__(self, tool_id: str) -> bool:
        return tool_id in self._cache
