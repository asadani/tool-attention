"""IntentRouter: query-to-tool semantic router with state-aware gating.

Implements the Intent-Schema Overlap (ISO) score from
"Tool Attention Is All You Need" (Sadani, 2026):

    ISO(q, t_i) = cos(phi(q), phi(s_i))

with a thresholded top-k gate and an optional precondition predicate
that queries agent state to enforce auth scopes, milestones, etc.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from sentence_transformers import SentenceTransformer

from vector_store import ToolVectorStore


@dataclass(frozen=True)
class RoutingResult:
    """A single routed tool with its ISO score."""

    tool_id: str
    score: float


class IntentRouter:
    """Embeds queries and returns thresholded, state-gated top-k tools."""

    def __init__(
        self,
        store: ToolVectorStore,
        encoder: SentenceTransformer | None = None,
        encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        threshold: float = 0.28,
        top_k: int = 10,
    ) -> None:
        self.store = store
        self.encoder = encoder or SentenceTransformer(encoder_name)
        self.threshold = threshold
        self.top_k = top_k

    def embed_query(self, query: str) -> np.ndarray:
        vec = self.encoder.encode(
            [query], normalize_embeddings=True, show_progress_bar=False
        )
        return np.asarray(vec[0], dtype="float32")

    def route(
        self,
        query: str,
        precondition_check: Callable[[str], bool] | None = None,
    ) -> list[RoutingResult]:
        """Route a query to up to `top_k` tools whose ISO score >= threshold
        and whose declared preconditions are satisfied by the agent state.
        """
        eq = self.embed_query(query)
        # Retrieve a wider slate so the precondition filter can't starve the gate.
        slate = self.store.search(eq, k=max(self.top_k * 4, 20))
        gated: list[RoutingResult] = []
        for tool_id, score in slate:
            if score < self.threshold:
                continue
            if precondition_check is not None and not precondition_check(tool_id):
                continue
            gated.append(RoutingResult(tool_id=tool_id, score=float(score)))
            if len(gated) >= self.top_k:
                break
        return gated
