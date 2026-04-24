# Tool Attention — Reference Implementation

Companion code for the paper
**"Tool Attention Is All You Need: Dynamic Tool Gating and Lazy Schema Loading for Eliminating the MCP/Tools Tax in Scalable Agentic Workflows"**
(Sadani, 2026). Published on arXiv: [arxiv.org/abs/2604.21816](https://arxiv.org/abs/2604.21816).
See [`paper.md`](paper.md) for the Markdown version or [`latex/paper.pdf`](latex/paper.pdf) for the rendered PDF.

Tool Attention is a drop-in middleware layer for LLM agents that eliminates
the *MCP Tax* — the 10k–60k tokens of tool-schema overhead that stateless
MCP injection imposes on every conversational turn. It combines:

1. an **Intent–Schema Overlap (ISO)** score over sentence embeddings,
2. a **state-aware gating function** enforcing preconditions and scopes, and
3. a **two-phase lazy schema loader** (summary pool + on-demand full schemas).

On a 120-tool synthetic MCP catalog, the reference implementation measures a
**98.6% reduction in the always-resident summary pool** and a **91.9%
reduction in the per-turn marginal schema cost** versus naive full-schema
injection, with steady-state cost dominated by the cache-amortized Phase-2
payload.

---

## Repository layout

```
.
├── paper.md                     # full preprint (Markdown, LaTeX math)
├── latex/
│   ├── paper.tex                # arXiv-style LaTeX source
│   ├── references.bib           # 35-entry bibliography
│   └── paper.pdf                # 19-page rendered preprint
└── code/
    ├── requirements.txt
    ├── vector_store.py          # FAISS-backed tool-summary index
    ├── intent_router.py         # ISO router + state-aware gate
    ├── lazy_loader.py           # LRU-cached full-schema loader
    ├── tool_attention.py        # before_model / after_model middleware
    ├── build_catalog.py         # generates the 120-tool synthetic testbed
    └── benchmark.py             # token-counting harness
```

---

## Installation

Requires Python 3.10+.

```bash
cd code
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The first run will download the
[`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
encoder (~90 MB).

---

## Quick start: reproduce the token-reduction numbers

```bash
cd code
python build_catalog.py   # creates catalog/tools.json and catalog/schemas/*.json
python benchmark.py       # prints the reduction table
```

Expected output (seed 42, 120 tools, 7 queries):

```
| Method                              | tokens/turn |   reduction |
|-------------------------------------|------------:|------------:|
| B1 Naive Full-Schema                |      57,452 |        0.0% |
| B3 Simple Retrieval (top-k schemas) |       5,390 |       90.6% |
| Tool Attention: Phase-1 only        |         787 |       98.6% |
| Tool Attention: Phase-2 only        |       4,672 |       91.9% |
| Tool Attention: first turn (P1+P2)  |       5,459 |       90.5% |
```

Under prompt caching the steady-state per-turn cost is dominated by Phase-2
(the full schemas for the top-\(k\) active tools); Phase-1 is cached after
the first turn.

---

## Minimal library usage

```python
from pathlib import Path
import tiktoken
from sentence_transformers import SentenceTransformer

from vector_store import ToolVectorStore
from intent_router import IntentRouter
from lazy_loader import LazySchemaLoader
from tool_attention import ToolAttention

enc = tiktoken.get_encoding("cl100k_base")
count = lambda s: len(enc.encode(s))

encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

tools = [
    {"id": "github__list_open_prs",
     "summary": "List open GitHub pull requests filtered by label and assignee."},
    {"id": "slack__search",
     "summary": "Search Slack messages by channel, author, and time range."},
    # ... hundreds more
]

store = ToolVectorStore(dim=384)
store.add_tools(tools, encoder)

loader = LazySchemaLoader(registry_path=Path("catalog/schemas"))
router = IntentRouter(store=store, encoder=encoder, threshold=0.28, top_k=10)
ta     = ToolAttention(store, loader, router, token_counter=count)

# Per-turn middleware call
result = ta.before_model("list open PRs labeled `bug` in the auth repo")
print(result.active_ids)            # routed tools for this turn
print(result.phase1_tokens, result.phase2_tokens)
```

### State-aware gating

`IntentRouter.route(query, precondition_check=...)` takes an optional
predicate that receives a tool_id and returns `True` if the agent's current
state satisfies the tool's preconditions (auth scopes, workflow milestones,
etc.):

```python
def allowed(tool_id: str) -> bool:
    if tool_id.startswith("github__") and not state.authenticated("github:write"):
        return False
    return True

result = ta.before_model(query, precondition_check=allowed)
```

### Hallucination rejection gate

After the model returns a tool call, wrap it through `after_model` to
reject any call targeting a tool whose schema was not promoted this turn:

```python
err = ta.after_model(result.active_ids, requested_tool=model_output.tool_name)
if err:
    # Return `err` to the model as a structured error; it will retry.
    pass
```

---

## LangGraph / LangChain 1.0 middleware

The `before_model` and `after_model` methods match the LangGraph middleware
contract. A minimal wiring:

```python
from langgraph.graph import StateGraph
# ... build graph ...
graph.add_middleware(before_model=ta.before_model, after_model=ta.after_model)
```

The Phase-1 summary pool should be placed in the **stable prefix** of the
prompt (so it stays inside the prompt-cache boundary), and Phase-2 full
schemas immediately before the user's current message.

---

## Extending

* **Swap the encoder**: instantiate a different `SentenceTransformer` (e.g.
  `BAAI/bge-large-en-v1.5`) and pass it into both `ToolVectorStore.add_tools`
  and `IntentRouter`. Expect ~2–4 pp retrieval improvement on well-named
  catalogs; ~3× latency cost.
* **Swap the backend**: `ToolVectorStore` can be replaced with a Chroma-
  backed store implementing the same `add_tools`/`search` surface.
* **Threshold calibration**: collect ≥100 (query, ground-truth-tool) pairs
  and sweep `threshold` in `IntentRouter` to maximise F1. Typical optimum:
  0.22–0.32 for MiniLM-L6.
* **Remote schemas**: pass a `fetcher` callable into `LazySchemaLoader` that
  calls an MCP server's `tools/list` on demand instead of loading from disk.

---

## Reproducibility

All experiments in the paper use `seed=42`. The synthetic catalog, the
threshold, and the top-\(k\) are all deterministic. `build_catalog.py`
regenerates the same 120-tool testbed byte-for-byte across runs.

---

## Citation

If you build on this work, please cite:

```bibtex
@misc{sadani2026toolattention,
  title        = {Tool Attention Is All You Need: Dynamic Tool Gating and
                  Lazy Schema Loading for Eliminating the MCP/Tools Tax in
                  Scalable Agentic Workflows},
  author       = {Sadani, Anuj},
  year         = {2026},
  eprint       = {2604.21816},
  archivePrefix = {arXiv},
  primaryClass = {cs.AI},
  url          = {https://arxiv.org/abs/2604.21816}
}
```

---

## License

MIT — see [`LICENSE`](LICENSE).

---

## Contact

Anuj Sadani — `anuj.k.sadani@gmail.com`
