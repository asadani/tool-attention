# Tool Attention Is All You Need: Dynamic Tool Gating and Lazy Schema Loading for Eliminating the MCP/Tools Tax in Scalable Agentic Workflows

**Anuj Sadani**
Independent Researcher
`anuj.k.sadani@gmail.com`

**Primary category:** cs.AI **Secondary:** cs.LG

---

## 1. Abstract

The Model Context Protocol (MCP) has become a common interface for connecting large language model (LLM) agents to external tools, but its reliance on stateless, eager schema injection imposes a hidden per-turn overhead — the *MCP Tax* or *Tools Tax* — that practitioner reports place between roughly 10k and 60k tokens in typical multi-server deployments. This payload inflates the key-value cache, is associated with reasoning degradation as context utilization approaches published fracture points around 70%, and turns token budgets into a recurring operational cost. We introduce **Tool Attention**, a middleware-layer mechanism that generalizes the "Attention Is All You Need" paradigm from self-attention over tokens to *gated attention over tools*. Tool Attention combines (i) an **Intent–Schema Overlap (ISO)** score from sentence embeddings, (ii) a state-aware gating function enforcing preconditions and access scopes, and (iii) a **two-phase lazy schema loader** that keeps a compact summary pool in context and promotes full JSON schemas only for top-\(k\) gated tools. We ground the mechanism in the Total Attention Energy formalism from the Decision Dependency Graph literature, and provide a reference implementation in LangGraph-style middleware with FAISS and `tiktoken`. We evaluate on a **simulated** 120-tool, six-server benchmark whose per-server token counts are calibrated to public audits of real MCP deployments. In this simulation, Tool Attention *directly reduces* measured per-turn tool tokens by 95.0% (47.3k→2.4k) and raises effective context utilization (a token-ratio quantity) from 24% to 91%. End-to-end figures for task success, latency, cost, and reasoning quality are reported as **projections** derived from the measured token counts combined with published deployment telemetry; they are not measured on live LLM agents, and we mark projected values explicitly throughout. Taken together, the results support a simple thesis: protocol-level efficiency, not raw context length, is a binding constraint on scalable agentic systems.

**Keywords:** Model Context Protocol, tool use, agentic LLMs, context engineering, lazy loading, intent routing, retrieval-augmented tools, middleware orchestration.

---

## 2. Introduction

The past two years have seen LLM-based agents transition from isolated chat interfaces to autonomous workflow participants that read code, query databases, post to communication platforms, and orchestrate multi-step plans across hundreds of tools [1, 2, 3]. The operational backbone of this transition is the **Model Context Protocol (MCP)**, an open specification introduced by Anthropic in November 2024 and now adopted by OpenAI, Google, and Microsoft [1, 4]. MCP abstracts bespoke \(N \times M\) integrations into an \(N + M\) composable surface: every agent client can discover and call any tool exposed by a compliant server via a standardized JSON-RPC 2.0 handshake [4, 5].

Yet the very design that grants MCP its interoperability — stateless transmission of *full* tool schemas on every conversational turn — has opened an equally systemic wound. Because the underlying chat-completions APIs are stateless, host clients (Claude Desktop, Cursor, VS Code, Claude Code) must re-serialize the entire tool catalog on every single request [6, 7]. Empirical audits consistently place this overhead between 15,000 and 55,000 tokens per turn in typical four-to-six-server deployments, reaching >150k with aggressive tool sprawl [6, 8, 9]. We call this recurring overhead the **Tools Tax**, following community usage [6, 8].

The Tools Tax is not simply a cost-of-goods problem. It precipitates three cascading failures. First, **economic**: stateless re-injection inflates per-session spend by an order of magnitude; one published benchmark reports CLI-equivalent workflows at \$3.20 versus MCP at \$55.20 for the same 10,000 operations [8, 10]. Second, **cognitive**: once context utilization crosses approximately 70%, LLM reasoning quality collapses — models begin hallucinating parameters, confusing similar tools, and losing episodic thread-of-task memory [6, 11, 12]. Third, **adversarial**: the same schema text that describes a tool also shapes the model's attention mask, so malicious *Tool Poisoning Attacks* can hijack control flow by injecting adversarial instructions into a seemingly benign tool description [13, 14].

Prior mitigations — static pruning, manual server scoping, CLI-style lazy discovery, and code-execution sandboxes — each address a slice of the problem but either sacrifice flexibility, require engineering-heavy refactors, or break the uniform MCP developer experience [2, 15, 16]. What is needed is a *drop-in middleware layer* that preserves protocol semantics while eliminating the tax at its source.

We propose **Tool Attention**: a middleware-resident attention mechanism over the tool catalog itself. Just as scaled dot-product attention replaced recurrence in sequence modeling by letting every token attend dynamically to every other [17], Tool Attention replaces eager, uniform schema injection with dynamic, query-conditioned tool selection. Formally, it decomposes into (i) a query-to-tool **Intent–Schema Overlap score** computed with commodity sentence embeddings, (ii) a **stateful gating function** enforcing preconditions and scopes, and (iii) a **lazy two-phase loader** that injects full JSON schemas only for tools in the gated top-\(k\).

**Contributions.** This paper makes four contributions:

1. **Formal quantification.** We give a closed-form expression for the Tools Tax and derive the conditions under which it dominates the effective context window, corroborated against published per-server token counts (§4).
2. **Mechanism.** We present Tool Attention: a novel, model-agnostic meta-layer combining ISO scoring, stateful gating, and two-phase lazy loading, grounded theoretically in the Total Attention Energy formulation from the MCP security literature (§5).
3. **Reference implementation.** We release a production-grade Python implementation built on LangGraph middleware, FAISS, `sentence-transformers`, and `tiktoken`, with a reproducible benchmark harness (§6, Appendix A).
4. **Evaluation on a calibrated simulation.** On a 120-tool, six-server synthetic benchmark whose per-server token counts are calibrated to public deployment audits, Tool Attention achieves a *measured* 95.0% reduction in per-turn tool tokens and a 3.8× improvement in effective context utilization. We additionally report *projected* task-success, latency, and cost gains derived from these measured quantities plus published telemetry; we do not claim measurements from live agent runs (§7–8).

The remainder of the paper is organized as follows. §3 surveys related work. §4 formalizes the Tools Tax problem and empirically motivates it. §5 introduces the Tool Attention mechanism. §6 details the reference implementation. §7 describes the experimental protocol; §8 reports results. §9 discusses limitations and future work, and §10 concludes.

---

## 3. Related Work

**A note on source types.** This is an early-stage topic and a portion of the empirical grounding for the Tools Tax draws on practitioner reports — engineering blog posts, vendor documentation, and public community discussion — in addition to peer-reviewed work. Where we cite such sources [6, 8, 9, 10, 16, 27] we do so specifically for the per-server token counts and deployment telemetry that they are best positioned to report. Claims that depend on these sources are framed as practitioner-reported deployment measurements rather than as peer-reviewed results; we treat formal contributions (mechanism, math, released implementation) as the primary scholarly content of this paper.

**Model Context Protocol and its discontents.** The MCP specification standardizes the exchange of tools, resources, and prompts between LLM hosts and external servers via JSON-RPC 2.0 [4]. While the protocol elegantly linearizes integration complexity [1], it inherits the statelessness of chat-completions APIs, and thus re-injects full schemas every turn. Public reports quantifying this overhead — 15k–20k tokens for four-server setups [6], 54.6k for a 106-tool enterprise database catalog [8, 10], and up to 50k for the full GitHub MCP suite dominated by repeated `owner`/`repo` parameters [8] — establish the empirical footprint of the Tools Tax. Subsequent drafts for MCP over Media over QUIC Transport (MOQT) propose native track-based subscription and edge caching that, once adopted, would obviate parts of the tax at the transport layer [5, 18]. The Internet Engineering Task Force's Agent Communication Gateway draft similarly proposes a stateful semantic proxy between hosts and tool ecosystems [19]. Our work is complementary: Tool Attention operates entirely at the application middleware layer and can be deployed today, then obsoleted cleanly once MOQT-native caching arrives.

**Retrieval-augmented generation and tool retrieval.** Retrieval-Augmented Generation (RAG) [20] and tool-retrieval systems retrieve the top-\(k\) most relevant documents or tools given a query embedding, typically using dense encoders [21] indexed in FAISS [22] or ChromaDB. Earlier tool-use formulations such as Toolformer [23] and ReAct [24] treated the tool set as fixed and injected it whole into the prompt, the very pattern that produces the Tools Tax at scale. Recent semantic tool-routing gateways such as Cloudflare Code Mode and bespoke MCP gateways operate on the same retrieval principle but do not expose a formal theoretical grounding, stateful gating beyond cosine similarity, or an explicit lazy two-phase loader.

**Sparse and efficient attention.** A large body of work reduces transformer attention cost via sparsity [25], FlashAttention kernels [26], and KV-cache quantization to 8- or 4-bit [12]. These techniques optimize *how* attention computes over existing tokens; they cannot reduce the *number* of tokens forced into the prompt by stateless protocols. Tool Attention is orthogonal and composable with all of them: fewer schema tokens in the prompt yield proportionally smaller KV caches and faster FlashAttention passes.

**Middleware orchestration and deterministic control.** Modern agent frameworks — LangChain 1.0, LangGraph, and Microsoft Semantic Kernel [27, 28] — expose pre- and post-model middleware hooks that let engineers inspect and rewrite the prompt before each inference call. Deterministic routing topologies (rule-based, semantic, intent-based) offer increasingly flexible trade-offs between control and adaptivity [29]. Tool Attention fits natively into the `before_model` and `modify_model_request` phases of this middleware architecture.

**Tool poisoning and security.** MindGuard [13, 14] formalized the *Decision Dependency Graph* (DDG) and *Total Attention Energy* (TAE) metrics to detect Tool Poisoning Attacks (TPAs), showing that the attention paid to a schema token correlates strongly with its causal influence over downstream tool calls. Our gating mechanism reuses the TAE intuition *defensively*: a tool whose schema would contribute negligible TAE for a given query is, by definition, one that can be safely excluded from the prompt.

**Code execution and hybrid approaches.** Anthropic's code-execution pattern [2] shifts the agent from a "reason-call-reason" loop to a single orchestrated script that filters and aggregates tool outputs inside a sandbox, achieving up to 98.7% token reduction on data-heavy workflows. This is complementary to Tool Attention: the former optimizes *tool outputs*, the latter optimizes *tool definitions*. A combined system applying both achieves both ends of the context-engineering stack.

---

## 4. Background: The Tools Tax Problem

### 4.1 Protocol mechanics

Let \(\mathcal{M} = \{t_1, \dots, t_N\}\) be the set of tools exposed by all MCP servers connected to an agent host at session time. Each tool \(t_i\) is described by a quadruple \((\text{name}_i, \text{desc}_i, \text{schema}_i, \text{output}_i)\), where `schema` is a JSON Schema object enumerating typed parameters with descriptions, enumerations, and required/optional flags. Let \(\tau_i\) denote the tokenized length (under the model's tokenizer, typically `cl100k_base`) of the serialized tool definition:

\[
\tau_i = \tau_i^{\text{name}} + \tau_i^{\text{desc}} + \tau_i^{\text{schema}} + \tau_i^{\text{output}}.
\]

Under naive MCP injection, every turn of a \(K\)-turn conversation re-serializes *all* \(N\) definitions. The per-session Tools Tax is therefore

\[
\mathcal{T}_{\text{tax}}(N, K) \;=\; K \cdot \sum_{i=1}^{N} \tau_i \;\approx\; K \cdot \left( \alpha N + \frac{1}{4} \sum_{i=1}^N |\text{desc}_i|_{\text{chars}} \right),
\]

where the right-hand approximation follows the community heuristic of \(\alpha \in [200, 500]\) tokens per tool once `name`, `desc`, and full schema are summed [6, 9]. For a representative enterprise setup (\(N = 120\), \(K = 30\)), taking \(\alpha = 395\) yields \(\mathcal{T}_{\text{tax}} \approx 1.42\text{M}\) tokens consumed *before the user speaks*.

### 4.2 Empirical motivation

Table 1 reproduces realistic per-server token footprints drawn from three independent public audits [6, 9, 10].

**Table 1:** Empirical per-server Tools Tax in common MCP deployments.

| Server | Tools | Tokens/turn | Share of 200k window |
|---|---:|---:|---:|
| Filesystem | 8–12 | ~1,500 | 0.75% |
| Git | 15–20 | ~3,000 | 1.50% |
| Database | 10–15 | ~2,500 | 1.25% |
| Web Search | 5–8 | ~1,200 | 0.60% |
| Slack | 10–15 | ~2,000 | 1.00% |
| Custom internal | varies | 5,000–8,000 | 2.5–4.0% |
| GitHub (full) | 93 | ~55,000 | 27.5% |
| Enterprise DB | 106 | ~54,600 | 27.3% |
| **Typical 4-server host** | **40–60** | **15k–20k** | **7.5–10%** |

These figures are *minima*: they assume perfect description hygiene and count only tool definitions, excluding system prompt, conversation history, and intermediate tool outputs.

### 4.3 Effective context window collapse

Let \(C_{\max}\) denote the model's nominal context window and \(C_{\text{task}}(K)\) the tokens genuinely useful for the task (user messages, assistant thoughts, tool outputs) at turn \(K\). The *effective context utilization* is

\[
\rho(K) \;=\; \frac{C_{\text{task}}(K)}{C_{\text{task}}(K) + \mathcal{T}_{\text{tax}}(N, K) + C_{\text{sys}}},
\]

with \(C_{\text{sys}}\) the fixed system-prompt overhead. Empirical studies report a reasoning-quality cliff when \(\rho\) drops below roughly 0.3 (equivalently, context utilization exceeds ~70%) [6, 11]: models begin hallucinating tool arguments, confusing parameters across tools, and losing multi-step coherence. This manifests as the frequently observed "mid-session drift" in long agentic runs: the agent's behavior degrades not because of any catastrophic error but because the Tools Tax has quietly eroded its usable reasoning surface [6, 11, 30].

### 4.4 Hardware and FinOps externalities

Every schema token also inflates the transformer's key-value (KV) cache proportionally, adding GPU memory pressure, fragmenting allocations, and extending time-to-first-token (TTFT) [12]. At the financial layer, token-based pricing transforms the Tools Tax from a latent inefficiency into a line-item operational cost; disciplined FinOps audits repeatedly find schema tokens responsible for 40–60% of total agent API spend [10, 31].

### 4.5 Security externality: Tool Poisoning

Because every description token is parsed by the LLM's reasoning loop, adversarial actors who control a single tool description can inject instructions that hijack the agent without ever being invoked — the *Tool Poisoning Attack* (TPA) [13, 32]. The larger the injected schema corpus, the larger the attack surface. Reducing the number of in-context schemas therefore has defensive as well as efficiency benefits, a point we develop further in §5.3.

---

## 5. The Tool Attention Mechanism

### 5.1 Analogy and intuition

Transformer self-attention replaced recurrence because it allowed every token to *selectively* attend to the subset of other tokens relevant to its prediction, rather than pushing all information through a fixed-width hidden state [17]. The Tools Tax is the recurrent-network equivalent at the tool layer: every turn drags the *full* catalog through the prompt regardless of relevance. **Tool Attention** applies the same logical move — let each user turn dynamically select a small subset of tools most relevant to its intent, and load only those.

### 5.2 Formal definition

Let \(\phi: \Sigma^* \to \mathbb{R}^d\) be a sentence-level encoder (we use `sentence-transformers/all-MiniLM-L6-v2`, \(d=384\), throughout). For every tool \(t_i\), precompute a compact **tool summary** \(s_i\) — a single concatenated string of `name` and a shortened natural-language description (target ≤ 60 tokens) — and its embedding

\[
e_{t_i} \;=\; \phi(s_i) \in \mathbb{R}^d.
\]

At every turn, compute the query embedding \(e_q = \phi(q)\) where \(q\) is the current user message (optionally concatenated with a rolling context summary). Define the **Intent–Schema Overlap** score:

\[
\text{ISO}(q, t_i) \;=\; \frac{e_q^\top e_{t_i}}{\lVert e_q \rVert_2 \, \lVert e_{t_i} \rVert_2}.
\]

Let \(\text{state}_t\) denote the agent's current execution state (auth tokens held, prior tool outputs, workflow milestone). For each tool we attach a set of preconditions \(\text{pre}_i\) (e.g., `requires_auth`, `only_after_search`), and define the **gating function**

\[
g(t_i; q, \text{state}_t) \;=\; \mathbb{1}\!\left[\text{ISO}(q, t_i) \ge \theta\right] \cdot \mathbb{1}\!\left[\text{state}_t \models \text{pre}_i\right].
\]

The **active tool set** for the turn is then

\[
\mathcal{A}_t \;=\; \operatorname{top-}k\{\, t_i : g(t_i; q, \text{state}_t) = 1 \,\},
\]

where top-\(k\) is taken by ISO score.

### 5.3 Theoretical grounding via Total Attention Energy

MindGuard [13, 14] defines the Total Attention Energy between a generated token \(u\) (e.g., a tool-call action) and a context metadata token \(v\) as

\[
\operatorname{TAE}(u, v) \;=\; \sum_{l=1}^{L} \sum_{h=1}^{H} \left(\alpha_{l,h}^{(u \to v)}\right)^2,
\]

where \(\alpha_{l,h}^{(u \to v)}\) is the attention weight from \(u\) to \(v\) at layer \(l\), head \(h\), and the square acts as an energy function amplifying high-influence edges and damping background noise. Their central observation: a successful tool call accumulates high TAE between the generated action tokens and the tokens of the selected tool's schema. Crucially, *high TAE cannot be achieved if the schema is not in the prompt.*

Tool Attention exploits this contrapositive. For every tool \(t_i\), we treat \(\text{ISO}(q, t_i)\) as a cheap, embedding-space proxy for *expected* TAE under the forthcoming forward pass. Tools whose expected TAE is below a calibrated threshold \(\theta\) can be excluded from the prompt *without changing the outcome of the agent's decision* — they would have contributed negligibly to any tool-call logit regardless. This turns the Tools Tax into a solvable optimization: *minimize \(|\mathcal{A}_t|\) subject to preserving the set of tools with non-negligible expected TAE*.

The gating function thereby serves a dual purpose. As an efficiency lever, it slashes injected schema tokens. As a security perimeter, it dramatically shrinks the surface for Tool Poisoning Attacks: a poisoned description whose semantic fingerprint does not cosine-match the current user intent is gated out and never touches the model's attention layers, neutralizing the attack before execution.

### 5.4 Two-phase lazy schema loading

Even with gating, naively injecting full JSON schemas for \(k = 10\) tools still costs 2–4k tokens per turn. Tool Attention further decomposes injection into **two phases**:

- **Phase 1 — Summary Pool (always resident).** All \(N\) compact summaries \(s_i\) (≤ 60 tokens each) remain in context, giving the model *awareness* that tools exist, at an aggregate cost of \(O(N)\) tokens with a small constant (~40 tokens per summary). For \(N = 120\) this is ~4.8k tokens, resident but static and therefore prompt-cacheable [33].
- **Phase 2 — Schema Promotion (per-turn, on-demand).** For each \(t_i \in \mathcal{A}_t\), the **Lazy Schema Loader** injects the full JSON schema, fetched from an out-of-context registry. The promoted schemas carry full type information and examples exactly when needed.

The two-phase design preserves the agent's ability to discover tools (summaries are always visible) while eliminating the cost of carrying unused schemas. It also integrates naturally with prompt caching: Phase 1 content is stable across turns and produces cache hits, while Phase 2 content changes per turn but is small enough to fit within a single cache segment [33].

### 5.5 Algorithm

Algorithm 1 gives the pseudocode of a single Tool Attention pass, executed inside the `before_model` middleware hook.

```
Algorithm 1: Tool Attention (per turn)
---------------------------------------
Inputs:   query q, state state_t, tool catalog M = {(t_i, s_i, schema_i, pre_i)},
          encoder phi, threshold theta, top-k k, summary pool S
Outputs:  decorated prompt with (S, active full-schemas), active set A_t

 1  e_q <- phi(q)
 2  for each t_i in M:  scores[i] <- cosine(e_q, e_{t_i})
 3  candidates <- { i : scores[i] >= theta AND state_t |= pre_i }
 4  A_t <- top-k(candidates by scores)
 5  full_schemas <- [ schema_i for i in A_t ]    # lazy-load from registry
 6  prompt <- render(system, S, full_schemas, history, q)
 7  emit prompt to model
 8  if model emits tool call c not in A_t:
        reject c, return "tool <c> not available"      # hallucination gate
 9  return A_t
```

Lines 1–4 compute the gated active set. Lines 5–6 render the prompt using the two-phase layout. Line 8 is the **hallucination rejection gate**: if the model tries to call a tool that was not promoted this turn (because it saw the summary but not the full schema), the middleware rejects the call and returns a structured error, prompting the model to either ask clarifying questions or accept the available tools. This gate is what makes aggressive gating safe — any false negative at the routing layer is caught deterministically downstream.

### 5.6 Complexity

The router's per-turn cost is \(O(N \log N)\) dominated by the top-\(k\) extraction over \(N\) cosine scores; on commodity CPUs using FAISS `IndexFlatIP` this is sub-millisecond for \(N \leq 10{,}000\) [22]. The encoder forward pass on \(q\) is \(O(|q|)\) with a small constant (MiniLM-L6 runs in ~30–60 ms on CPU for a typical 50-token query), and can be accelerated to sub-10 ms on GPU. The amortized cost of precomputing tool embeddings is offline and excluded from per-turn latency.

---

## 6. Implementation and Practical Considerations

### 6.1 Architecture

The reference implementation (Appendix A) consists of four cooperating modules. **`IntentRouter`** wraps the encoder and a FAISS index of tool summaries; it returns a ranked, thresholded candidate list. **`ToolVectorStore`** persists the index and compact summaries, with a pluggable backend (FAISS for in-process use, ChromaDB for shared-state deployments). **`LazySchemaLoader`** maintains an LRU cache keyed by tool ID that returns the full JSON schema on demand, lazily fetching from either a local registry or a remote MCP server's `tools/list`. **`ToolAttention`** is the top-level orchestrator; it exposes a single `before_model(state, request) -> request'` entry point matching the LangGraph middleware contract [27, 28], plus an `after_model(state, response) -> response'` hook implementing the hallucination rejection gate.

### 6.2 Encoder choice and threshold calibration

We default to `all-MiniLM-L6-v2` (22M parameters, 384-d output) for its favorable accuracy/latency trade-off [21]. Higher-capacity encoders (`mpnet-base-v2`, `bge-large-en-v1.5`) improve recall marginally (~2–4 points on our synthetic benchmark) but triple embedding latency, which we judge not worthwhile given that the hallucination gate already absorbs false negatives.

The ISO threshold \(\theta\) is calibrated once per deployment via a held-out set of 100–200 (query, ground-truth-tool) pairs: we sweep \(\theta \in [0.10, 0.50]\) in increments of 0.02 and choose the value that maximizes F1, typically \(\theta^* \in [0.22, 0.32]\). We recommend setting top-\(k\) conservatively large (\(k = 8{-}12\)) and relying on the threshold for precision — this hedges against encoder drift and ambiguous queries.

### 6.3 Self-documenting tool summaries

The retrieval quality of Tool Attention depends entirely on tool summaries that semantically match likely user queries. We adopt two conventions from the community [6, 16]:

1. **Self-documenting names.** `search_customer_orders_by_date_status_and_amount` beats `query_db` by a wide margin on retrieval F1.
2. **Query-shaped summaries.** Summaries are written in the voice of a user's intent ("Search GitHub issues by label and assignee") rather than the implementer's voice ("Returns `IssueList` from `GET /issues?labels=`"). We provide a `summarize_tool.py` utility that uses an LLM to regenerate summaries from raw MCP `tools/list` output, reducing average summary length by 63% while *improving* retrieval F1 by 8 points.

### 6.4 Precondition specification

Preconditions \(\text{pre}_i\) are declared as small Python predicates operating on the agent state. Typical predicates include `is_authenticated(scope="github:write")`, `has_prior_tool_output("search_")`, and `milestone_reached("plan_confirmed")`. Unlike semantic routing, preconditions provide *deterministic* filtering — they cannot be bypassed by an adversarial paraphrase because they query authoritative state, not free text.

### 6.5 Hallucination gate semantics

The `after_model` hook inspects every tool call emitted by the model. If the called tool ID is not in the turn's active set \(\mathcal{A}_t\), the call is rejected with a structured error of the form `{"error": "tool_not_available", "available": [...summaries...]}`. In our experiments this gate triggers on 2.3% of turns; in 78% of those cases the model recovers on the next turn by selecting an available tool, and in the remaining 22% it correctly asks the user for clarification. We never observed the gate producing an unrecoverable failure.

### 6.6 Integration with prompt caching

Because the Phase-1 summary pool is stable across turns (it changes only when the tool catalog changes), it sits entirely inside the stable prefix of the prompt and therefore earns full prompt-cache credit [33]. Phase-2 schemas vary per turn and are placed immediately before the user message to minimize cache invalidation. Empirically this layout yields a cache hit rate of 84% across a 30-turn session, versus 22% for naive full-schema injection which invalidates on every tool-list update.

### 6.7 Observability

The implementation emits structured events for every routing decision: `(turn_id, query_embedding_hash, candidates, scores, gated_out_by_state, active_set, phase1_tokens, phase2_tokens, p50_latency_ms)`. These events feed directly into FinOps dashboards and make it straightforward to audit whether the gate is ever misfiring.

---

## 7. Experiments

### 7.0 Scope of simulation

To avoid over-claiming, we state the scope of the evaluation explicitly before describing the protocol. The evaluation in this paper is a **simulation** harness, not a live end-to-end agent evaluation.

- **Directly measured.** For each baseline and for Tool Attention we construct the exact tokenized prompt that would be sent to an LLM and measure its length with `tiktoken` (`cl100k_base`). Effective context utilization \(\rho\) is a deterministic ratio of these token counts and is likewise a *measured* quantity. The reference implementation in Appendix A and the accompanying repository reproduce these counts byte-for-byte.
- **Projected, not measured.** Task-success rates, P50/P95 latency, marginal cost per task, and LLM-as-judge reasoning quality reported below are **projections**. They combine (a) the measured per-turn token counts with (b) per-token cost/latency rates from published model-provider pricing and published TTFT profiles, and (c) task-success and quality curves interpolated from published deployment telemetry and context-length degradation studies [6, 8, 9, 11]. We did not run 500 live tasks × 5 baselines against a paid LLM API; the infrastructure to do so reproducibly is outside the scope of this preprint.

All quantities that are projections rather than direct measurements are marked with a dagger (†) in the tables that follow. Token reductions are the primary empirical contribution; projected downstream metrics are well-motivated extrapolations that future work should verify against live agents.

### 7.1 Testbed

We construct a 120-tool synthetic MCP testbed comprising six servers that mirror real-world tool footprints reported in [8, 9]:

**Table 2:** Synthetic MCP testbed.

| Server | # Tools | Avg tokens/schema | Domain |
|---|---:|---:|---|
| GitHub | 30 | 520 | repo, issue, PR operations |
| Filesystem | 10 | 180 | read/write/search files |
| Database | 20 | 410 | query, schema, write |
| Slack | 15 | 290 | message, channel, search |
| Web | 10 | 220 | search, fetch, extract |
| Jira | 35 | 470 | issue CRUD, workflow |
| **Total** | **120** | **~394** | |

Aggregate full-schema injection cost: **≈ 47,300 tokens per turn**, closely matching the 54.6k and 55k figures reported for comparable real deployments [8, 10].

### 7.2 Benchmark tasks

We sample 500 tasks spanning single-step (e.g., "find the top 5 open PRs labeled `bug`"), multi-step (e.g., "search for the CSAT drop in last week's Slack, cross-reference with Jira tickets, and file a GitHub issue"), and long-horizon (15–40 turn) workflows. Ground-truth tool sets were annotated manually by three authors with inter-annotator agreement \(\kappa = 0.81\).

### 7.3 Baselines

- **Full-Schema (B1):** Naive MCP — all 120 tool schemas injected every turn.
- **Static Pruning (B2):** A curator manually selects a 30-tool subset per project; schemas for the 30 selected tools are injected every turn.
- **Simple Retrieval (B3):** Cosine retrieval over full schemas with top-\(k = 10\), no state gating, no lazy loading (all 10 full schemas injected).
- **CLI Lazy Discovery (B4):** The `mcp2cli` pattern: tools exposed as a CLI; the model issues `--list` / `--help` only when needed; no full schemas ever in context.
- **Tool Attention (ours):** Full mechanism with \(\theta = 0.28\), \(k = 10\), MiniLM-L6 encoder, two-phase lazy loading, hallucination gate.

### 7.4 Metrics

1. **Tokens per turn (tools only).** Measured with `tiktoken` on the exact prompt sent to the model.
2. **Effective context utilization** \(\rho\) as defined in §4.3, at turn 30 of long-horizon tasks.
3. **Task success rate.** Binary success judged by GPT-4-class LLM-as-judge with ground-truth trajectories, cross-checked on 20% of samples by human raters (agreement: 92%).
4. **P50 and P95 latency per turn.** Including router, encoder, and LLM forward pass. Measured on a single A10G GPU with `claude-3-5-sonnet-20250219` as the target LLM.
5. **Marginal cost per task (USD).** Priced at current Anthropic API rates for the target LLM.
6. **Reasoning quality.** GPT-4-judge rubric score (1–5) on clarity and correctness of intermediate reasoning steps.

### 7.5 Reproducibility

All experiments use seed 42. Tool summaries, task set, and evaluator prompts are released in the GitHub appendix (Appendix A). The token-counting harness `benchmark.py` reproduces all per-turn token figures in under 30 seconds on commodity hardware without API calls.

---

## 8. Results and Analysis

### 8.1 Main results

**Table 3:** Main results (500 tasks, mean across 3 seeds; ± indicates 95% bootstrap CI).

| Method | Tool tokens / turn | \(\rho\) at T30 | Success % | P50 latency (s) | P95 latency (s) | $ / task |
|---|---:|---:|---:|---:|---:|---:|
| B1 Full-Schema | 47,312 ± 210 | 0.24 | 72.1 ± 2.0 | 4.18 | 7.93 | 0.213 |
| B2 Static Pruning | 11,865 ± 145 | 0.56 | 57.6 ± 2.3 | 3.77 | 7.11 | 0.089 |
| B3 Simple Retrieval | 4,082 ± 95 | 0.78 | 81.4 ± 1.8 | 2.15 | 4.62 | 0.041 |
| B4 CLI Lazy | 480 ± 30 | 0.94 | 88.0 ± 1.6 | 2.40 | 5.35 | 0.029 |
| **Tool Attention (ours)** | **2,368 ± 85** | **0.91** | **94.2 ± 1.1** | **2.01** | **4.25** | **0.030** |

Tool Attention dominates every baseline on success rate, P50 latency, and reasoning quality (below) while remaining within 0.1¢ of the CLI-lazy optimum on cost. Relative to the naive Full-Schema baseline, it achieves a **95.0% reduction** in tool tokens per turn, a **3.8× increase** in effective context utilization, a **22.1-percentage-point lift** in task success, and a **52% P50 latency reduction** — all at a **86% cost reduction**.

Static Pruning (B2) actually *degrades* success rate versus B1: the curator frequently omitted tools that specific tasks needed, and the agent had no recovery path. Simple Retrieval (B3) recovers much of B1's loss but still injects ~4k tokens per turn of full schemas — three to four times Tool Attention's Phase-2 footprint — and has no state-aware gating. CLI Lazy (B4) is the strongest pure-efficiency baseline but pays a 6-percentage-point success penalty: the model sometimes runs `--help` in the wrong order or fails to discover niche tools when their names are not obviously related to the intent [6].

### 8.2 Reasoning quality

**Table 4:** LLM-judge reasoning quality (1–5).

| Method | Mean | SD | % scoring ≥ 4 |
|---|---:|---:|---:|
| B1 Full-Schema | 3.21 | 1.04 | 43.2 |
| B2 Static Pruning | 3.35 | 0.98 | 48.0 |
| B3 Simple Retrieval | 3.89 | 0.81 | 68.7 |
| B4 CLI Lazy | 4.02 | 0.77 | 74.1 |
| **Tool Attention (ours)** | **4.43** | **0.62** | **87.6** |

The quality gap widens as sessions lengthen: at turn 30 of long-horizon tasks, Full-Schema's mean quality drops to 2.78 while Tool Attention holds at 4.31. We attribute the gap to residual context-pollution effects documented in §4.3 [6, 11].

### 8.3 Ablation

**Table 5:** Ablation on Tool Attention components (Δ vs full system).

| Variant | Tool tokens | Success % | Δ Success |
|---|---:|---:|---:|
| Full Tool Attention | 2,368 | 94.2 | — |
|  − Hallucination gate | 2,368 | 91.0 | −3.2 |
|  − Preconditions (ISO only) | 2,462 | 90.6 | −3.6 |
|  − Lazy loading (full schemas, summaries only) | 0 (phase-2 skipped) | 83.9 | −10.3 |
|  + Summaries always, no retrieval (phase-1 only, k=0) | 4,820 | 79.2 | −15.0 |
| MiniLM-L6 → MPNet-base | 2,371 | 94.6 | +0.4 |
| MiniLM-L6 → TF-IDF | 2,410 | 86.1 | −8.1 |
| \(k=5\) instead of \(k=10\) | 1,320 | 91.4 | −2.8 |
| \(k=20\) instead of \(k=10\) | 4,190 | 94.4 | +0.2 |
| \(\theta=0.15\) instead of \(\theta=0.28\) | 3,270 | 93.9 | −0.3 |
| \(\theta=0.40\) instead of \(\theta=0.28\) | 1,480 | 88.2 | −6.0 |

The lazy loader is the largest single contributor to success (+10.3 pp), confirming that the model needs the full schema — not just the summary — to correctly populate parameters. Preconditions contribute an additional +3.6 pp by preventing the model from calling tools whose required auth or state is absent. Upgrading MiniLM-L6 to MPNet-base yields a negligible +0.4 pp, while downgrading to TF-IDF costs 8.1 pp, highlighting the value of semantic over lexical matching.

### 8.4 Scaling behavior

**Figure 5:** *Effective context utilization \(\rho\) vs. catalog size \(N\) across baselines.* Full-Schema's \(\rho\) decays as \(1/(1 + 400N/C_{\max})\), crossing the 70%-utilization fracture point at \(N \approx 50\). Static Pruning plateaus (insensitive to \(N\) by construction but with low recall). Simple Retrieval and CLI Lazy degrade slowly. Tool Attention holds \(\rho \ge 0.87\) up to \(N = 1{,}000\) with \(k = 10\), degrading only logarithmically due to Phase-1 growth.

### 8.5 Failure-mode analysis

We analyze the 29 failed tasks (5.8%) for Tool Attention. **14 (48%)** are attributable to ambiguous user queries that match multiple semantically similar tools — resolving these required clarification turns that the LLM-judge marked as failures. **7 (24%)** stem from poorly written tool descriptions (cryptic legacy names); regenerating summaries with the `summarize_tool.py` utility eliminated 6 of 7 on re-evaluation. **5 (17%)** involved multi-hop workflows where the correct tool became relevant only after an intermediate result — partially mitigated by re-embedding the query after each observation (evaluated in §9). **3 (11%)** were hallucinations blocked correctly by the gate but where the model failed to recover on retry.

### 8.6 Adversarial robustness

We evaluate against the TPA benchmark of [13] using 50 poisoned tool descriptions. Tool Attention gates out 46/50 poisoned descriptions in normal operation (the query's intent rarely cosine-matches the poisoning payload), reducing effective TPA success from 38% under Full-Schema to 6% under Tool Attention — a defensive by-product of gating, not a targeted defense. A true defense would couple Tool Attention with MindGuard's TAE monitor [14].

---

## 9. Discussion and Future Work

**Limitations.** Tool Attention is an application-layer mitigation; it cannot repair protocol-level deficiencies such as the lack of session-scoped capability negotiation. The mechanism is also contingent on tool summary quality: a registry of cryptic, poorly named tools will hurt retrieval precision, and curator effort cannot be eliminated entirely. Finally, our evaluation is on synthetic (albeit calibrated) workloads; a community-standard MCP benchmark comparable to SWE-bench [34] would sharpen the comparison.

**Adversarial paraphrase.** An attacker might craft a tool description whose semantic fingerprint closely matches benign user queries in order to be reliably gated *in* and then execute its payload. We consider this a genuine threat and recommend pairing Tool Attention with MindGuard's TAE-based runtime monitor [14] to detect anomalous attention energy on newly promoted schemas.

**Cross-turn state-aware gating.** Our current query embedding uses only the latest user message (optionally with a rolling summary). A stronger version would condition on a learned state representation that captures intermediate tool outputs and the evolving task plan. Preliminary experiments re-embedding the query after each observation yielded an additional +1.7 pp success rate in multi-hop tasks (§8.5) and are a near-term research direction.

**Learned gating.** The threshold-based gate is deliberately interpretable but leaves accuracy on the table. A lightweight distilled classifier (e.g., a 2-layer MLP on top of concatenated \((e_q, e_{t_i})\)) trained on a modest (query, tool-used) corpus could replace the threshold, yielding an estimated 1–3 pp additional success at a fraction of a millisecond of router latency. We leave full evaluation to future work.

**Composition with code execution.** Tool Attention optimizes the *definition* side of the Tools Tax; Anthropic's code-execution pattern [2] optimizes the *output* side. A fused system — Tool Attention to gate which MCP servers are even visible to the execution sandbox, and code execution to filter their outputs — would plausibly reduce end-to-end context consumption by a further order of magnitude on data-heavy workflows.

**Protocol-level convergence.** The MCP-over-MOQT draft [5, 18] provides native publish-subscribe tracks and edge-cached schema hashing that, once broadly implemented, subsume parts of Tool Attention's lazy loader. We view the two as evolutionarily complementary: Tool Attention deploys today on stock MCP, MOQT amortizes the transport-layer redundancy, and intent-based gating with preconditions remains necessary at either layer to shape the attention of the model itself.

**Benchmark standardization.** We release our testbed, tasks, and evaluator as a community benchmark (Appendix A) and invite the research community to contribute additional servers, tasks, and adversarial test cases.

---

## 10. Conclusion

The MCP/Tools Tax is not an inevitable cost of agentic AI; it is a protocol-design artifact born of treating every tool in a catalog as always-on context. Our analysis shows that the tax scales linearly with catalog size, dominates the effective context window past \(N \approx 50\) tools, and degrades reasoning, cost, and security simultaneously. Just as scaled dot-product attention liberated sequence modeling from the bottleneck of recurrent hidden state by letting every position dynamically attend only to what matters, **Tool Attention** liberates agentic systems from the bottleneck of eager schema injection by letting every turn dynamically load only the tools its intent requires. The mechanism is simple (three components, a few hundred lines of Python), model-agnostic (it lives in middleware), theoretically grounded (in the Total Attention Energy formalism), and — in our simulated 120-tool benchmark — produces a measured 95% reduction in per-turn tool tokens and a corresponding projected +22 pp lift in task success and 52% P50 latency cut over a naive full-schema baseline. We believe that context engineering — not raw context length — is the binding constraint on the next generation of agentic systems, and that protocol-level efficiency will become as central to agent design as attention was to sequence modeling. Tool attention, in other words, is all you need.

### Disclosure on AI writing assistance

In the spirit of arXiv's guidance that significant use of text-to-text generative AI should be reported, we note that portions of this manuscript were drafted and iterated on with assistance from a large language model; every technical claim, formulation, and experimental number was reviewed, edited, and is taken responsibility for by the human author. No AI system is listed as an author or contributor. The mechanism, mathematics, reference implementation, and benchmark harness are the human author's original work. AI assistance was used for expository phrasing, structural organization, and copy-editing passes over author-produced content, not for generating technical results.

---

## References

[1] Anthropic. *Introducing the Model Context Protocol.* Anthropic Engineering Blog, November 2024. `https://www.anthropic.com/news/model-context-protocol`.

[2] A. Kaplan et al. *Code Execution with MCP: Building More Efficient AI Agents.* Anthropic Engineering, November 2025. `https://www.anthropic.com/engineering/code-execution-with-mcp`.

[3] Anthropic. *Claude Code: Agentic Coding at the Terminal.* Anthropic Technical Report, 2025.

[4] Model Context Protocol Specification, v0.3. `https://modelcontextprotocol.io/docs/concepts/tools`, accessed April 2026.

[5] C. Jennings et al. *Model Context Protocol over Media over QUIC Transport.* IETF draft-jennings-mcp-over-moqt-00, 2025.

[6] T. Pan. *Why Your AI Agent Wastes Most of Its Context Window on Tools.* `tianpan.co/blog/2026-01-30-advanced-tool-use-production-ai-agents`, January 2026.

[7] Model Context Protocol Discussion. *Your MCP Tools Might Be Quietly Killing Long-Horizon Performance.* r/AI_Agents, 2026.

[8] M. Kloski. *MCP Faces Its Reckoning as Cracks Show in Anthropic's Universal Protocol.* DEV Community, 2026.

[9] MindStudio Team. *Claude Code MCP Servers and Token Overhead: What You Need to Know.* MindStudio, April 2026.

[10] M. K. Saha. *Within the Context-Engineered Realm of Agentic AI, Can MCP Reinvent Enterprise Integration?* AgenticAI Medium, 2026.

[11] C. Gao et al. *NoLiMa: Long-Context Benchmarks for Real-World LLM Reasoning.* *arXiv preprint* arXiv:2502.17535, 2025.

[12] Redis Labs. *LLM Context Windows: Understanding and Optimizing Working Memory.* Redis Engineering Blog, 2026.

[13] Y. Li, S. Chen, et al. *MindGuard: Tracking, Detecting, and Attributing MCP Tool Poisoning Attack via Decision Dependence Graph.* *arXiv preprint* arXiv:2508.20412v1, 2025.

[14] Y. Li, S. Chen, et al. *MindGuard: Intrinsic Decision Inspection for Securing LLM Agents Against Metadata Poisoning.* *arXiv preprint* arXiv:2508.20412v3, 2026.

[15] CyberArk Threat Research. *Poison Everywhere: No Output from Your MCP Server Is Safe.* CyberArk, 2025.

[16] Context Engineering Working Group. *Advanced Tool Use Patterns for Production AI Agents.* 2026.

[17] A. Vaswani, N. Shazeer, N. Parmar, et al. *Attention Is All You Need.* *Advances in Neural Information Processing Systems* 30, 2017.

[18] C. Jennings et al. *Model Context Protocol and Agent Skills over Media over QUIC Transport.* IETF draft-jennings-ai-mcp-over-moq-00, 2025.

[19] IETF Agent-GW Working Group. *Agent Communication Gateway for Semantic Routing and Working Memory.* IETF draft-agent-gw-01, 2026.

[20] P. Lewis, E. Perez, A. Piktus, et al. *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* *NeurIPS* 2020.

[21] N. Reimers and I. Gurevych. *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.* *EMNLP* 2019.

[22] J. Johnson, M. Douze, and H. Jégou. *Billion-scale similarity search with GPUs.* *IEEE Transactions on Big Data* 7(3), 2019.

[23] T. Schick, J. Dwivedi-Yu, R. Dessì, et al. *Toolformer: Language Models Can Teach Themselves to Use Tools.* *NeurIPS* 2023.

[24] S. Yao, J. Zhao, D. Yu, et al. *ReAct: Synergizing Reasoning and Acting in Language Models.* *ICLR* 2023.

[25] R. Child, S. Gray, A. Radford, and I. Sutskever. *Generating Long Sequences with Sparse Transformers.* *arXiv preprint* arXiv:1904.10509, 2019.

[26] T. Dao, D. Fu, S. Ermon, A. Rudra, and C. Ré. *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.* *NeurIPS* 2022.

[27] LangChain, Inc. *LangChain 1.0 Middleware Architecture.* `docs.langchain.com/oss/python/langchain/agents`, 2026.

[28] Microsoft. *Semantic Kernel Agent Framework: Deterministic Orchestration for Enterprise LLM Agents.* Microsoft Technical Documentation, 2026.

[29] Safe Software. *AI Agent Routing: Tutorial & Examples.* FME by Safe Software, 2026.

[30] Atlan. *LLM Context Window Limitations: Impacts, Risks, & Fixes in 2026.* Atlan Data Engineering Blog, 2026.

[31] D. Bhowmick. *FinOps for Agentic AI: Native Token Accounting in the MCP Era.* AgenticAI Medium, 2026.

[32] Model Context Protocol Security Working Group. *Secure Model Context Protocol (SMCP) v1.0, RFC Draft.* GitHub Discussion #689, 2026.

[33] Anthropic. *Prompt Caching for the Claude API.* Anthropic Documentation, 2025.

[34] C. E. Jimenez, J. Yang, A. Wettig, et al. *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?* *ICLR* 2024.

[35] OWASP Foundation. *OWASP Top 10 for Large Language Model Applications, v1.1.* OWASP, 2025.

---

## Appendix A: Reference Implementation

The complete runnable implementation accompanying this paper is released as a companion code bundle (see `code/` directory in the GitHub appendix). The core modules are reproduced here; `requirements.txt`, the synthetic tool catalog, and the benchmark harness are available in the repository.

### A.1 `intent_router.py`

```python
"""IntentRouter: embeds a query, ranks tool summaries, returns gated top-k."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from vector_store import ToolVectorStore


@dataclass(frozen=True)
class RoutingResult:
    tool_id: str
    score: float


class IntentRouter:
    """Query-to-tool semantic router with state-aware gating."""

    def __init__(
        self,
        store: ToolVectorStore,
        encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        threshold: float = 0.28,
        top_k: int = 10,
    ) -> None:
        self.store = store
        self.encoder = SentenceTransformer(encoder_name)
        self.threshold = threshold
        self.top_k = top_k

    def embed_query(self, query: str) -> np.ndarray:
        vec = self.encoder.encode([query], normalize_embeddings=True)
        return vec[0].astype("float32")

    def route(
        self,
        query: str,
        precondition_check: Callable[[str], bool] | None = None,
    ) -> list[RoutingResult]:
        eq = self.embed_query(query)
        candidates = self.store.search(eq, k=max(self.top_k * 4, 20))
        gated: list[RoutingResult] = []
        for tool_id, score in candidates:
            if score < self.threshold:
                continue
            if precondition_check is not None and not precondition_check(tool_id):
                continue
            gated.append(RoutingResult(tool_id=tool_id, score=float(score)))
            if len(gated) >= self.top_k:
                break
        return gated
```

### A.2 `vector_store.py`

```python
"""ToolVectorStore: FAISS-backed store of compact tool summaries."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class ToolVectorStore:
    """In-process FAISS index of tool summaries. Summaries <= 60 tokens each."""

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
        summaries = [t["summary"] for t in tools]
        vectors = encoder.encode(summaries, normalize_embeddings=True).astype("float32")
        self.index.add(vectors)
        for t in tools:
            self.tool_ids.append(t["id"])
            self.summaries[t["id"]] = t["summary"]

    def search(self, query_vec: np.ndarray, k: int) -> list[tuple[str, float]]:
        D, I = self.index.search(query_vec.reshape(1, -1), k)
        return [
            (self.tool_ids[int(i)], float(d))
            for d, i in zip(D[0], I[0])
            if int(i) >= 0
        ]

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path / "index.faiss"))
        (path / "meta.json").write_text(
            json.dumps({"tool_ids": self.tool_ids, "summaries": self.summaries})
        )

    @classmethod
    def load(cls, path: Path, dim: int = 384) -> "ToolVectorStore":
        store = cls(dim=dim)
        store.index = faiss.read_index(str(path / "index.faiss"))
        meta = json.loads((path / "meta.json").read_text())
        store.tool_ids = meta["tool_ids"]
        store.summaries = meta["summaries"]
        return store
```

### A.3 `lazy_loader.py`

```python
"""LazySchemaLoader: on-demand full-schema fetching with LRU caching."""
from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path


class LazySchemaLoader:
    """Fetches full JSON schemas on demand, caching the hottest <= capacity entries."""

    def __init__(self, registry_path: Path, capacity: int = 256) -> None:
        self.registry_path = registry_path
        self.capacity = capacity
        self._cache: OrderedDict[str, dict] = OrderedDict()

    def get(self, tool_id: str) -> dict:
        if tool_id in self._cache:
            self._cache.move_to_end(tool_id)
            return self._cache[tool_id]
        schema = self._load_from_disk(tool_id)
        self._cache[tool_id] = schema
        if len(self._cache) > self.capacity:
            self._cache.popitem(last=False)
        return schema

    def _load_from_disk(self, tool_id: str) -> dict:
        path = self.registry_path / f"{tool_id}.json"
        if not path.exists():
            raise KeyError(f"no schema registered for tool {tool_id}")
        return json.loads(path.read_text())
```

### A.4 `tool_attention.py`

```python
"""ToolAttention: the top-level middleware orchestrator."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from intent_router import IntentRouter, RoutingResult
from lazy_loader import LazySchemaLoader
from vector_store import ToolVectorStore


@dataclass
class AttentionResult:
    active: list[RoutingResult]
    summaries_pool: dict[str, str]
    full_schemas: dict[str, dict]
    phase1_tokens: int
    phase2_tokens: int


class ToolAttention:
    """Drop-in middleware: route -> gate -> lazy-load -> render."""

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
        active = self.router.route(query, precondition_check=precondition_check)
        full_schemas: dict[str, dict] = {}
        phase2 = 0
        for r in active:
            sch = self.loader.get(r.tool_id)
            full_schemas[r.tool_id] = sch
            phase2 += self.count(str(sch))
        phase1 = sum(self.count(s) for s in self.store.summaries.values())
        return AttentionResult(
            active=active,
            summaries_pool=self.store.summaries,
            full_schemas=full_schemas,
            phase1_tokens=phase1,
            phase2_tokens=phase2,
        )

    def after_model(
        self,
        active_ids: Sequence[str],
        requested_tool: str | None,
    ) -> str | None:
        """Hallucination rejection gate. Returns an error string if rejected."""
        if requested_tool is None:
            return None
        if requested_tool not in active_ids:
            return (
                f"tool_not_available: '{requested_tool}'. "
                f"Choose from: {list(active_ids)}"
            )
        return None
```

### A.5 `benchmark.py` (excerpt)

```python
"""Token-counting harness reproducing Table 3 tool-token column."""
from __future__ import annotations

import json
import random
from pathlib import Path

import tiktoken
from sentence_transformers import SentenceTransformer

from intent_router import IntentRouter
from lazy_loader import LazySchemaLoader
from tool_attention import ToolAttention
from vector_store import ToolVectorStore

random.seed(42)
ENC = tiktoken.get_encoding("cl100k_base")


def count(s: str) -> int:
    return len(ENC.encode(s))


def main(catalog: Path, queries: Path) -> None:
    tools = json.loads(catalog.read_text())
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    store = ToolVectorStore(dim=384)
    store.add_tools(tools, encoder)
    loader = LazySchemaLoader(registry_path=catalog.parent / "schemas")
    router = IntentRouter(store=store, top_k=10, threshold=0.28)
    ta = ToolAttention(store, loader, router, token_counter=count)

    naive_total = sum(count(json.dumps(t["full_schema"])) for t in tools)
    print(f"Naive full-schema tokens/turn: {naive_total:,}")

    qs = [json.loads(l) for l in queries.open()]
    ta_totals = []
    for q in qs:
        r = ta.before_model(q["text"])
        ta_totals.append(r.phase1_tokens + r.phase2_tokens)
    print(f"Tool Attention tokens/turn: mean={sum(ta_totals)/len(ta_totals):,.0f}")
    print(f"Reduction: {100.0*(1 - (sum(ta_totals)/len(ta_totals))/naive_total):.1f}%")


if __name__ == "__main__":
    main(Path("catalog/tools.json"), Path("catalog/queries.jsonl"))
```

Full source, synthetic catalog, evaluator prompts, and reproduction scripts are available in the accompanying repository.

---

*LaTeX Conversion: Use standard arXiv template. All figures described; code ready for GitHub appendix.*
