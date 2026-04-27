"""ToolVectorStore: Chroma-backed store of compact tool summaries.

Companion code for "Tool Attention Is All You Need"
(Anuj Sadani, 2026). Each summary must be a short natural-language
sentence (<= 60 tokens under cl100k_base) that reads as a user
intent (e.g., "Search GitHub issues by label and assignee").

Uses ChromaDB for dense semantic embeddings and ChromaBm25EmbeddingFunction
for sparse keyword embeddings.  Hybrid search is performed via Reciprocal
Rank Fusion (RRF) of the dense and sparse rankings.

Note: Local (single-node) ChromaDB does not yet expose the ``Search`` /
``Knn`` / ``Rrf`` query API, so RRF is implemented manually in Python
for broad compatibility.  When running against Chroma Cloud or a
future local release that supports ``collection.search(Search(...))``,
the dense and sparse ``Knn`` expressions can be replaced with the native
Chroma RRF plan shown in the doc comments below.
"""

from __future__ import annotations

import json
from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import Protocol, cast, final

import chromadb
import numpy as np
from chromadb.api.types import Metadatas, PyEmbeddings
from chromadb.api import ClientAPI
from chromadb.utils.embedding_functions import ChromaBm25EmbeddingFunction
from sentence_transformers import SentenceTransformer


class _SparseVector(Protocol):
    """Structural stand-in for chromadb.base_types.SparseVector."""

    indices: list[int]
    values: list[float]


@final
class ToolVectorStore:
    """Chroma-backed store with dense semantic + BM25 sparse hybrid search."""

    def __init__(
        self,
        dim: int = 384,
        collection_name: str = "tool_summaries",
        persist_dir: str | None = ".chroma_db",
    ) -> None:
        self.dim: int = dim
        self.collection_name: str = collection_name
        self.client: ClientAPI = (
            chromadb.PersistentClient(path=persist_dir)
            if persist_dir is not None
            else chromadb.Client()
        )
        self.collection: chromadb.Collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.tool_ids: list[str] = []
        self.summaries: dict[str, str] = {}
        self._sparse: dict[str, _SparseVector] = {}
        self._bm25: ChromaBm25EmbeddingFunction = ChromaBm25EmbeddingFunction()

    def add_tools(
        self,
        tools: Sequence[Mapping[str, object]],
        encoder: SentenceTransformer,
    ) -> None:
        """Add tools to the index.

        `tools` must be a sequence of dicts with at least keys
        'id' (str) and 'summary' (str).
        """
        if not tools:
            return
        summaries: list[str] = [cast(str, t["summary"]) for t in tools]
        ids: list[str] = [cast(str, t["id"]) for t in tools]
        vectors: np.ndarray = np.asarray(
            encoder.encode(  # pyright: ignore[reportUnknownMemberType]  # sentence-transformers overload stubs are incomplete
                summaries, normalize_embeddings=True, show_progress_bar=False
            )
        ).astype("float32")
        sparse_embeddings: list[_SparseVector] = cast(
            list[_SparseVector], self._bm25(summaries)
        )

        # Persist sparse vectors as ChromaDB metadata so they live in the
        # collection (not just an in-memory sidecar).  Native sparse-vector
        # indexing is unavailable in local/embedded ChromaDB, so we still
        # compute keyword scores manually in search().
        metadatas: list[dict[str, object]] = [
            {"sparse_indices": sp.indices, "sparse_values": sp.values}
            for sp in sparse_embeddings
        ]

        self.collection.add(
            ids=ids,
            embeddings=cast(PyEmbeddings, vectors.tolist()),
            documents=summaries,
            metadatas=cast(Metadatas, metadatas),
        )
        for t, sp in zip(tools, sparse_embeddings):
            self.tool_ids.append(cast(str, t["id"]))
            self.summaries[cast(str, t["id"])] = cast(str, t["summary"])
            self._sparse[cast(str, t["id"])] = sp

    def search(
        self,
        query_vec: np.ndarray,
        k: int,
        *,
        query_text: str | None = None,
    ) -> list[tuple[str, float]]:
        """Return up to `k` (tool_id, score) pairs sorted by score desc.

        When *query_text* is supplied the method fuses dense semantic
        rankings with BM25 sparse keyword rankings via RRF (k=60,
        equal weights).  Otherwise it falls back to dense-only search.
        """
        if not self.tool_ids:
            return []
        n_candidates = min(max(k * 4, 20), len(self.tool_ids))

        # Dense semantic search via ChromaDB
        dense_results = self.collection.query(
            query_embeddings=[query_vec.tolist()],
            n_results=n_candidates,
            include=["distances"],
        )
        dense_ids = dense_results["ids"][0]
        distances = dense_results["distances"]
        if distances is None:
            return []
        dense_distances = distances[0]
        dense_rank = {tid: rank for rank, tid in enumerate(dense_ids)}

        # Hybrid: RRF with BM25 sparse keyword search
        if query_text and self._sparse:
            query_sparse = self._bm25([query_text])[0]
            sparse_scores = {
                tid: self._sparse_dot(query_sparse, sp)
                for tid, sp in self._sparse.items()
            }
            sorted_sparse = sorted(sparse_scores.items(), key=lambda x: -x[1])
            sparse_rank = {tid: rank for rank, (tid, _) in enumerate(sorted_sparse)}

            all_ids = set(dense_rank.keys()) | set(sparse_rank.keys())
            rrf_scores: dict[str, float] = {}
            for tid in all_ids:
                score = 0.0
                if tid in dense_rank:
                    score += 1.0 / (60 + dense_rank[tid])
                if tid in sparse_rank:
                    score += 1.0 / (60 + sparse_rank[tid])
                rrf_scores[tid] = score

            # Normalise to roughly the same scale as cosine similarity [0, 1]
            max_rrf = 2.0 / 60  # two rankings, both at rank 0
            ranked = sorted(rrf_scores.items(), key=lambda x: -x[1])
            return [(tid, float(score) / max_rrf) for tid, score in ranked[:k]]

        # Fallback to dense-only
        return [
            (tid, 1.0 / (1.0 + dist)) for tid, dist in zip(dense_ids, dense_distances)
        ][:k]

    @staticmethod
    def _sparse_dot(a: _SparseVector, b: _SparseVector) -> float:
        """Dot product of two ChromaDB SparseVector objects."""
        b_dict: dict[int, float] = {idx: val for idx, val in zip(b.indices, b.values)}
        return sum(val * b_dict.get(idx, 0.0) for idx, val in zip(a.indices, a.values))

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        sparse_data: dict[str, dict[str, list[int] | list[float]]] = {
            tid: {"indices": sp.indices, "values": sp.values}
            for tid, sp in self._sparse.items()
        }
        _ = (path / "meta.json").write_text(
            json.dumps(
                {
                    "tool_ids": self.tool_ids,
                    "summaries": self.summaries,
                    "sparse_embeddings": sparse_data,
                },
                indent=2,
            )
        )
        if self.collection.count() > 0:
            result = self.collection.get(
                include=["embeddings", "documents", "metadatas"]
            )
            embeddings = result.get("embeddings")
            chroma_data: dict[str, object] = {
                "ids": result["ids"],
                "embeddings": [
                    e.tolist() if isinstance(e, np.ndarray) else e
                    for e in (embeddings if embeddings is not None else [])
                ],
                "documents": result.get("documents") or [],
                "metadatas": result.get("metadatas") or [],
            }
            _ = (path / "chroma_data.json").write_text(
                json.dumps(chroma_data, indent=2)
            )

    @classmethod
    def load(cls, path: Path, dim: int = 384) -> "ToolVectorStore":
        from chromadb.base_types import SparseVector

        store = cls(dim=dim)
        if (path / "meta.json").exists():
            meta = cast(
                dict[str, object],
                json.loads((path / "meta.json").read_text()),
            )
            store.tool_ids = list(cast(list[str], meta["tool_ids"]))
            store.summaries = dict(cast(dict[str, str], meta["summaries"]))
            sparse_embeddings = cast(
                dict[str, dict[str, list[int] | list[float]]],
                meta.get("sparse_embeddings", {}),
            )
            for tid, sp_data in sparse_embeddings.items():
                store._sparse[tid] = cast(
                    _SparseVector,
                    SparseVector(
                        indices=cast(list[int], sp_data["indices"]),
                        values=cast(list[float], sp_data["values"]),
                    ),
                )

        if (path / "chroma_data.json").exists():
            with open(path / "chroma_data.json") as f:
                chroma_data = cast(dict[str, object], json.load(f))
            ids = cast(list[str], chroma_data["ids"])
            embeddings = cast(list[list[float]], chroma_data["embeddings"])
            documents = cast(list[str], chroma_data["documents"])
            metadatas = cast(list[dict[str, object]], chroma_data["metadatas"])
            store.collection.add(
                ids=ids,
                embeddings=cast(list[Sequence[float]], embeddings),
                documents=documents,
                metadatas=cast(Metadatas, metadatas),
            )
            # Rebuild in-memory sparse cache from ChromaDB metadata
            # (covers cases where meta.json is missing / stale).
            for tid, meta in zip(ids, metadatas):
                if tid not in store._sparse and meta:
                    sp_indices = meta.get("sparse_indices")
                    sp_values = meta.get("sparse_values")
                    if sp_indices is not None and sp_values is not None:
                        store._sparse[tid] = SparseVector(
                            indices=cast(list[int], sp_indices),
                            values=cast(list[float], sp_values),
                        )
                        if tid not in store.tool_ids:
                            store.tool_ids.append(tid)

        return store

    @property
    def sparse_embeddings(self) -> dict[str, _SparseVector]:
        """Read-only access to the in-memory sparse embedding cache."""
        return self._sparse

    def __len__(self) -> int:
        return self.collection.count()
