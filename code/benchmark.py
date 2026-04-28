"""benchmark.py: reproduce the tool-token column of Table 3.

Runs the synthetic 120-tool catalog through:
  * Naive full-schema injection (baseline B1)
  * Simple retrieval (baseline B3) -- full schemas for top-k
  * Tool Attention (ours)          -- summary pool + lazy top-k schemas

Usage:
    python benchmark.py

Outputs a Markdown-style comparison table to stdout. Requires the
synthetic catalog at catalog/tools.json (generate via build_catalog.py
if missing).
"""

from __future__ import annotations

import json
import random
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import cast

try:
    import tiktoken
except ImportError:  # pragma: no cover
    print("missing dependency: pip install -r requirements.txt", file=sys.stderr)
    raise

from sentence_transformers import SentenceTransformer

from intent_router import IntentRouter
from lazy_loader import LazySchemaLoader
from tool_attention import ToolAttention
from vector_store import ToolVectorStore

SEED = 42
random.seed(SEED)
HERE = Path(__file__).parent
CATALOG = HERE / "catalog" / "tools.json"
SCHEMAS_DIR = HERE / "catalog" / "schemas"
QUERIES = HERE / "catalog" / "queries.jsonl"
ENC = tiktoken.get_encoding("cl100k_base")


def count(s: str) -> int:
    return len(ENC.encode(s))


def naive_schema_tokens(tools: Iterable[dict[str, object]]) -> int:
    return sum(count(json.dumps(t["full_schema"])) for t in tools)


def main() -> int:
    if not CATALOG.exists():
        print(
            f"missing catalog at {CATALOG}.\nRun `python build_catalog.py` to generate the synthetic 120-tool testbed.",
            file=sys.stderr,
        )
        return 2

    tools: list[dict[str, object]] = cast(
        list[dict[str, object]], json.loads(CATALOG.read_text())
    )
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    store = ToolVectorStore(dim=384)
    store.add_tools(tools, encoder)
    loader = LazySchemaLoader(registry_path=SCHEMAS_DIR)
    router = IntentRouter(store=store, encoder=encoder, threshold=0.28, top_k=10)
    ta = ToolAttention(store, loader, router, token_counter=count)

    naive_total = naive_schema_tokens(tools)

    queries: list[dict[str, str]] = (
        [cast(dict[str, str], json.loads(line)) for line in QUERIES.open()]
        if QUERIES.exists()
        else []
    )
    if not queries:
        queries = [
            {"text": "list open pull requests labeled bug in the auth repo"},
            {"text": "search Slack for CSAT complaints in the last 7 days"},
            {"text": "query the customers table for signups after 2026-01-01"},
        ]

    phase1_vals: list[int] = []
    phase2_vals: list[int] = []
    retr_totals: list[int] = []
    for q in queries:
        r = ta.before_model(q["text"])
        phase1_vals.append(r.phase1_tokens)
        phase2_vals.append(r.phase2_tokens)
        # Simple retrieval baseline: full schemas for top-k (no phase split).
        retr_totals.append(
            sum(count(json.dumps(loader.get(rr.tool_id))) for rr in r.active)
        )

    def mean(xs: list[int]) -> float:
        return sum(xs) / max(len(xs), 1)

    phase1 = mean(phase1_vals)
    phase2 = mean(phase2_vals)
    ta_total = phase1 + phase2
    retr_mean = mean(retr_totals)

    print(f"Catalog: {len(tools)} tools, {len(queries)} queries, seed={SEED}\n")
    print("| Method                              | tokens/turn |   reduction |")
    print("|-------------------------------------|------------:|------------:|")
    print(
        f"| B1 Naive Full-Schema                | {naive_total:>11,} | {0.0:>10.1f}% |"
    )
    print(
        f"| B3 Simple Retrieval (top-k schemas) | {retr_mean:>11,.0f} | {100 * (1 - retr_mean / naive_total):>10.1f}% |"
    )
    print(
        f"| Tool Attention: Phase-1 only        | {phase1:>11,.0f} | {100 * (1 - phase1 / naive_total):>10.1f}% |"
    )
    print(
        f"| Tool Attention: Phase-2 only        | {phase2:>11,.0f} | {100 * (1 - phase2 / naive_total):>10.1f}% |"
    )
    print(
        f"| Tool Attention: first turn (P1+P2)  | {ta_total:>11,.0f} | {100 * (1 - ta_total / naive_total):>10.1f}% |"
    )
    print()
    print("Notes:")
    print("  * Phase-1 is the always-resident summary pool. It is stable across")
    print("    turns and therefore prompt-cacheable (paper §5.4, §6.6), so its")
    print("    per-turn marginal cost approaches zero under prompt caching.")
    print("  * Phase-2 is the per-turn marginal cost (full schemas for top-k tools).")
    print("  * Under prompt caching, steady-state per-turn cost of Tool Attention")
    print("    is dominated by Phase-2 alone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
