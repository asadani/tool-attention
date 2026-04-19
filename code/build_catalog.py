"""build_catalog.py: generate the 120-tool synthetic MCP testbed (Table 2).

Produces:
  catalog/tools.json         (list of {id, summary, full_schema})
  catalog/schemas/<id>.json  (per-tool full JSON schemas for the lazy loader)
  catalog/queries.jsonl      (sample user intents used by benchmark.py)

Numbers are chosen to match the per-server footprints reported in
Table 2 of the paper. Deterministic under seed 42.
"""
from __future__ import annotations

import json
import random
import string
from pathlib import Path

SEED = 42
HERE = Path(__file__).parent
OUT = HERE / "catalog"
SCHEMAS_DIR = OUT / "schemas"

# (server, tool_count, avg_schema_tokens_target, verbs, objects, domain_noun)
SERVERS = [
    ("github", 30, 520, ["list", "get", "create", "close", "search", "comment"],
        ["pull_request", "issue", "repo", "branch", "commit", "workflow_run"],
        "GitHub"),
    ("filesystem", 10, 180, ["read", "write", "list", "search", "delete"],
        ["file", "directory"], "the local filesystem"),
    ("database", 20, 410, ["query", "describe", "insert", "update", "delete", "explain"],
        ["customers", "orders", "invoices", "sessions", "events", "schema"],
        "the primary OLTP database"),
    ("slack", 15, 290, ["post", "read", "search", "list", "dm"],
        ["channel", "thread", "user", "message"], "Slack"),
    ("web", 10, 220, ["search", "fetch", "extract", "screenshot"],
        ["page", "image", "snippet"], "the public web"),
    ("jira", 35, 470, ["create", "list", "transition", "comment", "search", "link"],
        ["issue", "epic", "sprint", "project", "worklog", "component"],
        "Jira"),
]

FILLER = (
    "Returns a structured JSON object representing the operation result. "
    "Supports pagination, filtering by common attributes, and optional "
    "expansion of related entities. Authentication is handled automatically "
    "via the host's credential manager. Rate limits apply per server."
)


def _schema_blob(target_tokens: int, rnd: random.Random) -> dict:
    """Build a JSON-Schema-shaped blob whose serialized size approximates target."""
    props: dict[str, dict] = {}
    required: list[str] = []
    approx_chars = target_tokens * 4  # cl100k ~4 chars/token
    i = 0
    while sum(len(json.dumps(v)) for v in props.values()) < approx_chars:
        name = "".join(rnd.choices(string.ascii_lowercase, k=rnd.randint(3, 10)))
        props[name] = {
            "type": rnd.choice(["string", "integer", "boolean"]),
            "description": (FILLER * (rnd.randint(1, 3)))[: rnd.randint(50, 220)],
        }
        if rnd.random() < 0.4:
            required.append(name)
        i += 1
        if i > 40:
            break
    return {
        "type": "object",
        "properties": props,
        "required": required,
        "additionalProperties": False,
    }


def build() -> None:
    rnd = random.Random(SEED)
    OUT.mkdir(exist_ok=True)
    SCHEMAS_DIR.mkdir(exist_ok=True)
    tools: list[dict] = []
    queries: list[dict] = []
    for server, count, avg_tok, verbs, objects, domain in SERVERS:
        for j in range(count):
            verb = rnd.choice(verbs)
            obj = rnd.choice(objects)
            tid = f"{server}__{verb}_{obj}_{j:02d}"
            summary = f"{verb.capitalize()} {obj.replace('_', ' ')} in {domain}."
            schema = _schema_blob(avg_tok, rnd)
            full = {
                "name": tid,
                "description": summary + " " + FILLER,
                "inputSchema": schema,
            }
            tools.append({"id": tid, "summary": summary, "full_schema": full})
            (SCHEMAS_DIR / f"{tid}.json").write_text(json.dumps(full, indent=2))
            if rnd.random() < 0.08:
                queries.append({"text": summary})

    (OUT / "tools.json").write_text(json.dumps(tools, indent=2))
    with (OUT / "queries.jsonl").open("w") as f:
        for q in queries:
            f.write(json.dumps(q) + "\n")

    print(f"wrote {len(tools)} tools -> {OUT/'tools.json'}")
    print(f"wrote {len(tools)} schemas -> {SCHEMAS_DIR}")
    print(f"wrote {len(queries)} sample queries -> {OUT/'queries.jsonl'}")


if __name__ == "__main__":
    build()
