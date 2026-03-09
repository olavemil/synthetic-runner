# Implementation Plan: Symbiosis Toolkit

This document describes the technical implementation path from the current working system toward the three-layer architecture described in ARCHITECTURE.md. The goal is a toolkit that makes both reference configurations (persistent memory agent, multi-instance hivemind) straightforward to set up without touching core machinery.

Work items are organized by architectural layer. Where the current code already has something relevant, the gap is described rather than starting from scratch.

---

## 1. Config System

**Current state:** Flat `.env` file with hardcoded key names in `lib.py`. One global instance. No structure for provider config, instance config, or multi-instance setups.

**Target:** Structured YAML config, separate from secrets, with a clear schema per layer.

### File layout

```
config/
  harness.yaml          # providers, adapters, scheduler settings
  instances/
    draum-1.yaml        # per-instance: species, model, messaging identity
    mayfly-1.yaml
.env                    # secrets only: API keys, tokens, credentials
```

### harness.yaml

```yaml
providers:
  - id: lmstudio
    type: openai_compat
    base_url: ${LMSTUDIO_BASE_URL}
    api_key: lm-studio

  - id: anthropic
    type: anthropic
    api_key: ${ANTHROPIC_API_KEY}

adapters:
  - id: matrix-main
    type: matrix
    homeserver: ${MATRIX_HOMESERVER}
    access_token: ${MATRIX_ACCESS_TOKEN}
```

### instances/draum-1.yaml

```yaml
species: draum
provider: anthropic         # default LLM provider for this instance
model: claude-opus-4-6

messaging:
  adapter: matrix-main
  entity_id: "@draum-bot:matrix.org"
  spaces:
    - name: main
      handle: "!LSbuHgXzmFaKNkpyfJ:matrix.org"
    - name: agents
      handle: "!KhVcjpFxUhh:matrix.org"

schedule:
  heartbeat: "0 * * * *"   # cron: every hour
```

### Notes

- `${VAR}` references are resolved from `.env` / environment at load time. Config files are safe to commit; secrets are not.
- Instance config is read-only at runtime. The Harness loads it; the instance can query it via `ctx.config(key)` but cannot write it.
- In the hivemind configuration, instance config omits the `messaging` block entirely.

---

## 2. LLM Provider Abstraction

**Current state:** `lib.py` has `make_client()` returning an OpenAI-compat client and `api_create()` wrapping it. Anthropic and other providers require their own SDKs and have minor API differences (tool_choice format, message structure).

**Target:** A provider interface that normalizes these differences so Species code never imports a vendor SDK directly.

### Interface

```python
class LLMProvider:
    def create(
        self,
        model: str,
        messages: list[dict],
        system: str | None = None,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
        max_tokens: int = 1024,
        caller: str = "?",
    ) -> LLMResponse:
        ...
```

`LLMResponse` is a normalized wrapper — `.message`, `.tool_calls`, `.finish_reason` — so Species code doesn't branch on provider.

### Implementations needed

| Provider | Notes |
|----------|-------|
| `OpenAICompatProvider` | Covers LMStudio, OpenAI, any OpenAI-compat endpoint. Current `api_create()` becomes this. |
| `AnthropicProvider` | Uses `anthropic` SDK. Needs translation: system prompt is a top-level param (already), tool_choice format differs, `stop_reason` vs `finish_reason`. |

### Retry and error handling

Retry logic (currently in `api_create()`) moves into the provider implementation. Species code sees a clean call or a raised exception.

### Think-block stripping

Models that emit `<think>...</think>` in their output — this is currently stripped only at send time in `matrix_send_room`. The provider layer is the better place, since think blocks can appear in intermediate tool call responses too.

---

## 3. Messaging Adapter Abstraction

**Current state:** Matrix-specific HTTP calls are directly in `lib.py` (`fetch_recent_messages`, `matrix_send_room`, etc.). No abstraction.

**Target:** A messaging adapter interface that Matrix, Slack, and a local-file adapter can all implement.

### Interface

```python
class MessagingAdapter:
    def send(self, space_handle: str, message: str, reply_to: str | None = None) -> str:
        """Send a message. Returns event/message ID."""

    def poll(self, space_handle: str, since_token: str | None) -> tuple[list[Event], str]:
        """Return (new_events, next_token). Events are normalized dicts."""

    def get_space_context(self, space_handle: str) -> dict:
        """Return name, topic, members. Used for composing context blocks."""
```

`Event` is a normalized dict: `{event_id, sender, body, timestamp}` — no platform-specific fields in Species code.

### Adapter implementations

| Adapter | Notes |
|---------|-------|
| `MatrixAdapter` | Current Matrix HTTP calls, reorganized. Poll uses `/sync` with `since` token. |
| `LocalFileAdapter` | Reads/writes message files in a local directory. Suitable for Config B or testing. No external dependency. |
| `SlackAdapter` | Bolt or direct API. Lower priority; same interface. |

The Harness holds adapter instances, keyed by adapter ID from `harness.yaml`. The instance context resolves logical space names to handles via the instance config, then calls the adapter.

---

## 4. Instance Context

**Current state:** No explicit context object. All operations use globals (`MEMORY_DIR`, `TOKEN`, etc.).

**Target:** An `InstanceContext` object constructed by the Harness at dispatch time, scoping all operations to one instance.

### Interface

```python
class InstanceContext:
    instance_id: str

    # Storage — all paths are relative to the instance namespace
    def read(self, path: str) -> str: ...
    def write(self, path: str, content: str) -> None: ...
    def list(self, prefix: str) -> list[str]: ...

    # Config — read-only
    def config(self, key: str) -> Any: ...

    # LLM — uses the instance's configured provider and model by default
    def llm(self, messages, *, model=None, provider=None, **kwargs) -> LLMResponse: ...

    # Messaging — resolves logical space name → handle via config
    def send(self, space: str, message: str, reply_to: str | None = None) -> str: ...
    def poll(self, space: str, since_token: str | None = None) -> tuple[list[Event], str]: ...

    # Inter-instance
    def send_to(self, target_instance_id: str, message: str) -> None: ...
    def read_inbox(self) -> list[dict]: ...

    # Structured store — typed key/value, backed by SQLite
    def store(self, namespace: str) -> NamespacedStore: ...          # instance-private
    def shared_store(self, namespace: str) -> NamespacedStore: ...   # shared across all instances

    # Spawn
    def spawn(self, new_instance_id: str) -> None: ...
```

Species handlers receive `ctx` as their only argument. They never construct absolute paths, never import provider SDKs, never call `httpx` directly.

---

## 5. Scheduling and Dispatch

**Current state:** `launchd` (macOS) runs `poll.py` and `daily.py` as separate scripts on separate schedules. No unified scheduler.

**Target:** A scheduler that manages all instances and entry points from a single process, using the Harness registry.

### Approach

Two dispatch modes:

- **Scheduled**: Cron or interval expressions in instance config. The scheduler fires the named entry point at the right time, constructing an `InstanceContext` and calling the Species handler.
- **Reactive**: On receiving a message event, the Harness fires the Species's reactive entry point with the event as input. Reactive can be:
  - **Poll-driven**: The scheduler polls registered adapters at a configurable interval (e.g., every 30 seconds) and fires on new events. Simple; no web server needed.
  - **Push-driven**: The adapter delivers events via callback (webhook, WebSocket). More efficient; requires an HTTP listener or similar.

Both modes work correctly from the system's perspective. The choice is a deployment concern, not a Species concern.

### Registry

The Harness maintains:

```python
[
  { instance_id, entry_point_name, schedule_or_trigger, handler },
  ...
]
```

Built at startup by reading instance configs and loading their Species manifests.

---

## 6. Species Toolkit: Pattern Library

**Current state:** Patterns exist but are standalone scripts (`compose.py`, `distill.py`, `relationships.py`, etc.) with implicit shared state via globals.

**Target:** A library of reusable patterns, each taking an `InstanceContext` and returning a result. Species entry points compose patterns.

### Patterns to formalize

| Pattern | Current location | Input → Output |
|---------|-----------------|----------------|
| `gut_response` | `poll.py` inline | `(ctx, events)` → guidance string |
| `plan_response` | `plan.py` | `(ctx, events, gut)` → plan string |
| `compose_response` | `compose.py` | `(ctx, guidance, room_context, relationships)` → message string |
| `run_subconscious` | `subconscious.py` | `(ctx, session_type)` → writes `subconscious.md` |
| `run_react` | `react.py` | `(ctx, session_type)` → writes `intentions.md`, optionally compacts `thinking.md` |
| `distill_memory` | `distill.py` | `(ctx, exclude)` → digest string |
| `distill_messages` | `distill.py` | `(ctx, messages)` → compressed string |
| `update_relationships` | `relationships.py` | `(ctx, session_type, events)` → writes relationship files |
| `run_session` | `lib.py` | `(ctx, system, initial_message, tools)` → bool (sent message) |

All patterns accept `ctx` as first argument. They read and write memory via `ctx.read()`/`ctx.write()`, call the LLM via `ctx.llm()`.

### Composition example

```python
# draum_species.py
from toolkit.patterns import gut_response, plan_response, compose_response, run_subconscious, run_react, update_relationships

def on_message(ctx: InstanceContext, events: list[Event]):
    gut = gut_response(ctx, events)
    plan = plan_response(ctx, events, gut)
    message = compose_response(ctx, plan, ...)
    if message:
        ctx.send("main", message)
    run_subconscious(ctx, "reactive")
    run_react(ctx, "reactive")
    update_relationships(ctx, "reactive", events)
```

---

## 7. Species Toolkit: Declarative Pipeline (YAML/TOML)

Allows simple Species definitions without writing Python. Suitable for Config B (hivemind) or lightweight agents.

Each stage in a pipeline declares its **inputs**, **outputs**, whether it **consumes** its inputs on success, and optional **preprocessors** applied to inputs before the stage runs. This makes the dataflow explicit and keeps stages composable without implicit shared state.

### Stage declaration model

```yaml
- stage: <pattern_name>
  inputs:
    <slot>: <source>     # where to read from
  outputs:
    <slot>: <destination> # where to write results
  consume_inputs:
    <slot>: true         # clear source on success (e.g. inbox, vote slots)
  preprocessors:
    <slot>:
      type: <preprocessor>
      <options>
```

**Sources** can be:

| Source | Example |
| ------ | ------- |
| Memory file | `memory.thinking` |
| Previous stage output | `pipeline.gut_guidance` |
| Inbox | `inbox.messages` |
| Instance config | `config.entity_id` |
| Structured store | `store.<namespace>.<key>` |

**Preprocessors** transform an input before passing it to the stage. The pipeline runner handles them; the stage receives the result.

| Preprocessor | Description |
| ------------ | ----------- |
| `truncate` | Hard-limit characters/tokens. No LLM. |
| `distill` | LLM summarization (maps to `distill_messages` pattern). |
| `map` | Apply a sub-pipeline to each item in a list, collect results. |
| `reduce` | LLM-merge a list of items into one (maps to `distill_memory`-style recursive merge). |

`consume_inputs` is the explicit mechanism for clearing processed state. An inbox slot with `consume: true` is cleared after the stage succeeds, so reprocessing doesn't happen on the next run. Memory files are typically not consumed — they persist across sessions.

### Example: reactive species with preprocessing

```yaml
species_id: simple-worker
provider: lmstudio
model: local-model

memory:
  files:
    - name: log.md
      initial: "# Log\n"

pipeline:
  on_inbox:
    steps:
      - stage: gut_response
        inputs:
          events: inbox.messages
          context: memory.thinking
        preprocessors:
          events:
            type: distill
            keep_recent: 4       # keep last 4 verbatim, summarize rest
          context:
            type: truncate
            max_chars: 2000
        outputs:
          guidance: pipeline.guidance
        consume_inputs:
          events: true           # clear inbox after processing

      - stage: compose_response
        inputs:
          guidance: pipeline.guidance
          memory: memory.thinking
        outputs:
          message: pipeline.message

      - stage: send_message
        inputs:
          message: pipeline.message
        consume_inputs:
          message: true

      - stage: run_subconscious
        inputs:
          thinking: memory.thinking
          sessions: memory.sessions

  on_schedule:
    cron: "0 * * * *"
    steps:
      - stage: distill_memory
        inputs:
          memory_dir: memory.*
        outputs:
          digest: pipeline.digest

      - stage: heartbeat_session
        inputs:
          digest: pipeline.digest
          subconscious: memory.subconscious
          intentions: memory.intentions
```

The Harness interprets the YAML and constructs entry point handlers from the named pipeline steps. Each step is either a built-in Harness pattern or a registered Species-level function.

**Tradeoffs:** YAML pipelines are linear only — no branching, no conditional stages. Conditional logic, dynamic fanout, or cross-stage state mutation requires dropping to Python. The intent is not to replace Python Species but to lower the floor for simple agents and make the Draum pipeline explicitly inspectable.

---

## 8. Inter-Instance Communication (Mailboxes)

**Current state:** Inter-instance communication happens via Matrix rooms (external broker). The Harness is uninvolved.

**Target:** Harness-mediated mailboxes as the canonical primitive for Config B. Matrix remains for human-facing spaces.

### Mechanics

- `ctx.send_to(target_id, message)` — Harness writes `{sender, body, timestamp}` to `/instances/{target}/inbox/{uuid}.json`.
- `ctx.read_inbox()` — Returns all inbox messages, then clears the inbox.
- The Harness does not interpret inbox contents; the receiving Species decides what to do with them.
- Inbox files are written atomically (write to temp, rename) to avoid partial reads.

---

## 9. Structured Data Store

**Current state:** Storage is entirely text files. Structured data is either serialized into markdown (losing queryability) or stored as ad-hoc JSON (e.g., `_index.json`). Neither supports concurrent writes, atomic operations, or aggregation.

**Target:** A lightweight typed store available to all Species, backed by SQLite. In-memory mode is available for ephemeral runtime state; file-backed mode persists across restarts.

### Motivation: hivemind voting

A purely file-based system cannot cleanly model the following:

1. A coordinator spawns 7 worker instances and writes a task description to shared state.
2. Each worker independently produces a candidate output based on its own memory and config.
3. Each worker casts a vote (its own candidate plus a ranking of others).
4. A tally step reads all votes, aggregates, selects a winner.
5. The coordinator acts on the result.

Votes are structured variables — `{voter_id, candidate_id, score, rationale}` — not prose. Forcing them into markdown would require parsing to aggregate. An atomic claim operation (`claim this task if unclaimed`) cannot be safely implemented with file writes.

### Store API

```python
class NamespacedStore:
    # Basic typed key/value — values are any JSON-serializable type
    def put(self, key: str, value: Any) -> None: ...       # upsert
    def get(self, key: str) -> Any | None: ...
    def delete(self, key: str) -> None: ...
    def scan(self, prefix: str = "") -> list[tuple[str, Any]]: ...
    def count(self, prefix: str = "") -> int: ...

    # Atomic claim — returns False if already claimed by another owner
    def claim(self, key: str, owner_id: str) -> bool: ...
    def release(self, key: str, owner_id: str) -> bool: ...
```

`ctx.store(namespace)` — instance-private; path is automatically prefixed with `instances/{id}/store/{namespace}`.

`ctx.shared_store(namespace)` — readable and writable by all instances; path is `species/{species_id}/store/{namespace}`. Species code uses this for coordination.

### Voting example (Python Species)

```python
def on_schedule(ctx):
    # Worker: produce a candidate and cast a vote
    candidate = generate_candidate(ctx)   # LLM call, instance-specific

    shared = ctx.shared_store("election")
    shared.put(f"candidate:{ctx.instance_id}", {
        "content": candidate,
        "author": ctx.instance_id,
    })

    # Read other candidates and vote
    candidates = shared.scan("candidate:")
    ranking = rank_candidates(ctx, candidates)   # LLM call
    shared.put(f"vote:{ctx.instance_id}", {
        "ranking": ranking,   # list of candidate keys, best first
        "timestamp": utcnow(),
    })


def tally(ctx):
    # Coordinator: aggregate votes once all are in
    shared = ctx.shared_store("election")
    votes = shared.scan("vote:")

    scores: dict[str, int] = {}
    for _, vote in votes:
        for rank, key in enumerate(vote["ranking"]):
            scores[key] = scores.get(key, 0) + (len(vote["ranking"]) - rank)

    winner_key = max(scores, key=scores.__getitem__)
    winner = shared.get(winner_key)
    ctx.write("memory/output.md", winner["content"])
```

### Voting example (YAML pipeline)

```yaml
pipeline:
  on_schedule:
    steps:
      - stage: generate_candidate
        outputs:
          candidate: store.shared.election/candidate:{instance_id}

      - stage: rank_candidates
        inputs:
          candidates: store.shared.election/candidate:*
        outputs:
          vote: store.shared.election/vote:{instance_id}

      - stage: tally_votes           # only runs when vote_count == expected
        condition: all_votes_in
        inputs:
          votes: store.shared.election/vote:*
        outputs:
          winner: memory.output
        consume_inputs:
          votes: true
          candidates: true
```

### Backend

SQLite is the backing store for both modes:

| Mode | Connection string | Notes |
| ---- | ----------------- | ----- |
| In-memory | `sqlite:///:memory:` | Shared within a process; lost on restart. Good for single-run coordination. |
| File-backed | `sqlite:///harness.db` | Persists across restarts. Needed if instances run on separate schedules. |

SQLite provides the row-level locking needed for `claim()`. All store operations go through the Harness; Species code never touches the database directly.

---

## Open Questions

- **Config secrets**: `.env` is fine for local dev. For deployments with multiple instances or shared infrastructure, a separate secrets store (OS keychain, Vault, etc.) is cleaner. Defer until multi-host is needed.
- **Anthropic tool_choice**: Anthropic's `tool_choice` API and message format differ enough from OpenAI's that the provider adapter needs careful testing, especially for `required` tool calls and parallel tool use.
- **Reactive polling interval**: Poll-driven reactive mode needs a sensible default interval (30s? configurable?). Too fast wastes resources; too slow makes the bot feel unresponsive.
- **YAML pipeline expressiveness**: How much branching / conditional logic is worth supporting before it's just Python in disguise? Suggest keeping YAML pipelines linear only; conditional logic drops to Python.
- **Package structure**: Single package `symbiosis/` with `harness/`, `toolkit/`, `species/` subpackages, or separate installable packages? Separate packages are cleaner for Config B (deploy only the worker, not the full Draum stack) but overkill until there are multiple real deployments.
- **Concurrency**: If two entry points for the same instance fire simultaneously (e.g., a heartbeat and an incoming message), the Harness should queue them, not run them in parallel. Simple lock-per-instance is sufficient.

---

## Build Order

Dependencies constrain the order:

1. **Config system** — Everything depends on structured config. Start here.
2. **Instance context** — Core abstraction; needed by patterns and adapters.
3. **LLM provider abstraction** — Unblocks Anthropic support; wraps existing `api_create`.
4. **Messaging adapter abstraction** — Refactors existing Matrix calls; enables Config B local adapter.
5. **Pattern library** — Formalize existing scripts as `ctx`-taking functions.
6. **Scheduler** — Replace launchd scripts with unified dispatch.
7. **YAML pipeline** — Highest-level; lowest priority; build last.
