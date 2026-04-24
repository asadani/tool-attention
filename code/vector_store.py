"""ToolVectorStore: FAISS-backed store of compact tool summaries.

Companion code for "Tool Attention Is All You Need"
(Anuj Sadani, 2026). Each summary must be a short natural-language
sentence (<= 60 tokens under cl100k_base) that reads as a user
intent (e.g., "Search GitHub issues by label and assignee").
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import faiss  # type: ignore[import-untyped]
import numpy as np
from sentence_transformers import SentenceTransformer


class ToolVectorStore:
    """In-process FAISS (IndexFlatIP) store of tool summary embeddings."""

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.tool_ids: list[str] = []
        self.summaries: dict[str, str] = {}

    def add_tools(
        self,
        tools: Sequence[dict],
        encoder: SentenceTransformer,
    ) -> None:
        """Add tools to the index.

        `tools` must be a sequence of dicts with at least keys
        'id' (str) and 'summary' (str).
        """
        if not tools:
            return
        summaries = [t["summary"] for t in tools]
        vectors = encoder.encode(
            summaries, normalize_embeddings=True, show_progress_bar=False
        ).astype("float32")
        self.index.add(vectors)
        for t in tools:
            self.tool_ids.append(t["id"])
            self.summaries[t["id"]] = t["summary"]

    def search(self, query_vec: np.ndarray, k: int) -> list[tuple[str, float]]:
        """Return up to `k` (tool_id, cosine_score) pairs sorted by score desc."""
        if self.index.ntotal == 0:
            return []
        k = min(k, self.index.ntotal)
        D, I = self.index.search(query_vec.reshape(1, -1).astype("float32"), k)
        return [
            (self.tool_ids[int(i)], float(d))
            for d, i in zip(D[0], I[0])
            if int(i) >= 0
        ]

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path / "index.faiss"))
        (path / "meta.json").write_text(
            json.dumps(
                {"tool_ids": self.tool_ids, "summaries": self.summaries},
                indent=2,
            )
        )

    @classmethod
    def load(cls, path: Path, dim: int = 384) -> "ToolVectorStore":
        store = cls(dim=dim)
        store.index = faiss.read_index(str(path / "index.faiss"))
        meta = json.loads((path / "meta.json").read_text())
        store.tool_ids = list(meta["tool_ids"])
        store.summaries = dict(meta["summaries"])
        return store

    def __len__(self) -> int:
        return self.index.ntotal
