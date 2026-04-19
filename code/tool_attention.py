"""ToolAttention: the top-level middleware orchestrator.

Wire into LangGraph / LangChain 1.0 as:

    ta = ToolAttention(store, loader, router, token_counter=count)
    ...
    agent.set_middleware(before_model=ta.before_model, after_model=ta.after_model)

The `before_model` hook rewrites the prompt to carry only the Phase-1
summary pool plus Phase-2 full schemas for the routed active set.
The `after_model` hook rejects tool calls for any tool that was not
promoted this turn (hallucination gate).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

from intent_router import IntentRouter, RoutingResult
from lazy_loader import LazySchemaLoader
from vector_store import ToolVectorStore


@dataclass
class AttentionResult:
    """Outcome of a single Tool Attention pass."""

    active: list[RoutingResult] = field(default_factory=list)
    summaries_pool: dict[str, str] = field(default_factory=dict)
    full_schemas: dict[str, dict] = field(default_factory=dict)
    phase1_tokens: int = 0
    phase2_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.phase1_tokens + self.phase2_tokens

    @property
    def active_ids(self) -> list[str]:
        return [r.tool_id for r in self.active]


class ToolAttention:
    """Drop-in middleware orchestrator: route -> gate -> lazy-load."""

    def __init__(
        self,
        store: ToolVectorStore,
        loader: LazySchemaLoader,
        router: IntentRouter,
        token_counter: Callable[[str], int],
    ) -> None:
        self.store = store
        self.loader = loader
        self.router = router
        self.count = token_counter

    def before_model(
        self,
        query: str,
        precondition_check: Callable[[str], bool] | None = None,
    ) -> AttentionResult:
        """Compute the per-turn active set and assemble the two-phase payload."""
        active = self.router.route(query, precondition_check=precondition_check)
        full_schemas: dict[str, dict] = {}
        phase2 = 0
        for r in active:
            schema = self.loader.get(r.tool_id)
            full_schemas[r.tool_id] = schema
            phase2 += self.count(_stringify(schema))
        phase1 = sum(self.count(s) for s in self.store.summaries.values())
        return AttentionResult(
            active=active,
            summaries_pool=dict(self.store.summaries),
            full_schemas=full_schemas,
            phase1_tokens=phase1,
            phase2_tokens=phase2,
        )

    def after_model(
        self,
        active_ids: Sequence[str],
        requested_tool: str | None,
    ) -> str | None:
        """Hallucination rejection gate.

        Returns an error-string payload if the model tried to call a tool
        whose schema was not promoted this turn; returns None otherwise.
        """
        if requested_tool is None:
            return None
        if requested_tool in active_ids:
            return None
        return (
            f"tool_not_available: {requested_tool!r}. "
            f"Available this turn: {list(active_ids)}"
        )


def _stringify(schema: dict) -> str:
    """Stable, deterministic string form for token accounting."""
    import json
    return json.dumps(schema, sort_keys=True, separators=(",", ":"))
