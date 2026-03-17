---
name: tools-patterns
description: Tool modules reference for Symbiosis — make_tools() options, phase scopes, patterns.py high-level patterns, prompts.py helpers, pipeline YAML format, and all tool module exports. Use when building species handlers, working with LLM tool loops, or using any library/tools/ module.
user_invocable: true
---

# Tools & Patterns Reference

All tool code lives in `library/tools/`. Tools are registered as OpenAI function schemas and dispatched through `handle_tool()`.

---

## make_tools() Options

`make_tools(ctx, options: dict | None) -> list[dict]`  (`library/tools/tools.py`)

| Option key | Default | Adds tools |
|------------|---------|-----------|
| `messaging` | `True` | `send_message` |
| `rooms` | `True` | `list_rooms` |
| `introspect` | `True` | `introspect` |
| `inter_instance` | `False` | `send_to_instance` |
| `publish` | `False` | `publish` |
| `graph` | `False` | 7 graph tools (see below) |
| `activation_map` | `False` | 8 map tools (see below) |

Always included: `read_file`, `write_file`, `list_files`, `done`.

**`write_file` auto-compacts** content exceeding `compact.threshold_chars` (calls `ctx.compact_file()`). Tool result includes "auto-compacted (N → M chars)" if triggered.

---

## Phase-Scoped Tool Filtering

`library/tools/phases.py` — use `get_tools_for_phase(phase, *, graph, activation_map)` to get the right tool set for each phase.

```python
from library.tools.phases import THINKING, COMPOSING, REVIEWING
from library.tools.phases import get_tools_for_phase

tools = get_tools_for_phase(THINKING, graph=True, activation_map=True)
```

| Phase constant | Tool scope |
|---------------|-----------|
| `THINKING` | Read/list tools + `append_thinking` + `archive_thoughts` |
| `COMPOSING` | Read/list tools only |
| `REVIEWING` | Organize tools (read/write topics, archive) + read/list |

Graph and activation_map tools are available in all phases when enabled.
Organize tools (`organize_*`) are always included in REVIEWING; optionally in THINKING.

---

## patterns.py — High-Level Patterns

`library/tools/patterns.py` — import directly in species handlers.

### Response pipeline

```python
gut = gut_response(ctx, events: list[Event], *, max_tokens=1024) -> dict
# Returns: {should_respond, urgency, brief, suggested_approach, rooms_to_respond}

plan = plan_response(ctx, events, gut_result, *, max_tokens=2048) -> dict
# Returns: {approach, key_points, tone, length, considerations}

response_text = compose_response(ctx, events, plan_result, *, max_tokens=4096) -> str
```

### Thinking phases

```python
thinking_session(
    ctx, initial_message: str, system: str,
    *, max_tokens=4096, max_turns=10,
    extra_tools: list | None = None,
) -> None
# Runs tool loop with append_thinking / replace_thinking / done tools.
# Writes to thinking.md. Auto-compacts thinking.md on append_thinking if over threshold.
```

```python
run_organize_phase(
    ctx, system: str,
    *, extra_context="", graph=True, activation_map=True,
    max_tokens=8192, label="species",
) -> None
# Runs knowledge organization tool loop. Uses REVIEWING phase tools.
# Builds knowledge structure summary and appends it to the initial message.
```

```python
run_create_phase(
    ctx, system: str,
    *, extra_context="", graph=True, activation_map=True,
    max_tokens=8192, label="species",
) -> None
# Creative output phase. Uses creative tools + read/list.
```

### Memory helpers

```python
digest = distill_memory(ctx, *, exclude=None, include=None) -> str
# Recursive memory compression via LLM. Returns a digest string.

summary = distill_messages(ctx, messages: list[Event]) -> str
# Compress a list of events into a summary string.
```

### Background patterns

```python
run_subconscious(ctx, *, context="") -> str
# Generates and writes subconscious.md. Returns content.

run_react(ctx, *, context="") -> None
# Generates and writes intentions.md from current context.

update_relationships(ctx, sender: str, events: list[Event]) -> None
# Writes/updates relationships/<sender>.md.
```

### LLM utilities

```python
text = llm_generate(ctx, system: str, message: str, *, max_tokens=2048, caller="?") -> str
# Simple one-shot LLM call, returns message text.
```

---

## prompts.py — Prompt Assembly Helpers

`library/tools/prompts.py`

```python
memory = read_memory(ctx, paths: list[str] | None = None) -> str
# Reads and concatenates memory files into a formatted block.
# Default paths: thinking.md, intentions.md, subconscious.md, project.md

entity_id = get_entity_id(ctx) -> str
# Returns instance's messaging entity_id (or empty string).

formatted = format_events(events: list[Event], *, max_events=20) -> str
# Formats event list as readable message history.

block = format_relationships_block(ctx) -> str
# Reads relationships/ files and formats as context block.

block = format_memory_context(ctx, *, extra_files: list[str] | None = None) -> str
# Combines memory + relationships into a full context block.

block = format_intentions_block(ctx) -> str
block = format_subconscious_block(ctx) -> str
```

---

## Pipeline YAML Format

`library/tools/pipeline.py` — `run_pipeline(ctx, pipeline_config, payload)`

Pipelines are YAML files loaded by species at startup. Each step maps to a patterns.py function or built-in stage.

```yaml
pipeline:
  on_message:
    steps:
      - stage: distill_messages
        inputs:
          messages: inbox.messages      # from payload
        outputs:
          summary: memory.inbox_summary # written to ctx store

      - stage: thinking_session
        inputs:
          system: prompts/think.md      # loaded from species prompts/
          max_tokens: 4096

      - stage: run_organize_phase
        inputs:
          graph: true
          activation_map: true

      - stage: llm_generate
        inputs:
          system: "Describe your dream."
        outputs:
          result: memory.dreams         # written to dreams.md

  on_schedule:
    cron: "*/15 * * * *"
    steps:
      - stage: distill_memory
        outputs:
          digest: memory.digest
```

**Available stages:** `distill_messages`, `distill_memory`, `run_subconscious`, `run_react`, `thinking_session`, `run_organize_phase`, `run_create_phase`, `llm_generate`, `compose_response`, `gut_response`, `plan_response`

---

## Graph Tools (`graph=True`)

`library/tools/graph.py` — `SemanticGraph` class, persistent JSON storage.

| Tool name | Description |
|-----------|-------------|
| `graph_add_node` | Add node with label + optional attributes |
| `graph_add_edge` | Add directed weighted edge between nodes |
| `graph_remove_node` | Remove node and its edges |
| `graph_remove_edge` | Remove specific edge |
| `graph_query` | Query nodes by label or attribute |
| `graph_describe` | Human-readable graph summary |
| `graph_snapshot` | Return full graph as JSON |

Stored as `graph.json` in instance memory. Can feed into slow net encoding.

---

## Activation Map Tools (`activation_map=True`)

`library/tools/activation_map.py` — `ActivationMap` class, 2D float grid.

| Tool name | Description |
|-----------|-------------|
| `map_define` | Create/resize map with dimension labels |
| `map_set` | Set value at (x, y) coordinate |
| `map_set_region` | Set values in a rectangular region |
| `map_get` | Read value at coordinate |
| `map_describe` | Human-readable heatmap summary |
| `map_clear` | Reset all values to zero |
| `map_snapshot` | Return full grid as JSON |
| `representation_summary` | Combined graph + map summary |

Stored as `activation_map.json`. Grid size typically 16×16 to 32×32. Feeds fast net encoding.

---

## Organize Tools

`library/tools/organize.py` — available in REVIEWING phase or via `handle_tool` dispatch.

| Tool name | Description |
|-----------|-------------|
| `organize_list_topics` | List all topics in a category |
| `organize_read_topic` | Read topic content |
| `organize_write_topic` | Write/update a topic entry |
| `organize_delete_topic` | Delete a topic |
| `organize_archive_thoughts` | Archive section of thinking.md with a label |

Topics stored as `knowledge/<category>/<topic>.md`.

---

## Creative Tools

`library/tools/creative.py` — enabled via `creative=True` in make_tools options (Neural Dreamer).

| Tool name | Description |
|-----------|-------------|
| `creative_new` | Create new artifact (md, svg, html, abc) |
| `creative_edit` | Edit existing artifact |
| `creative_read` | Read artifact content |
| `creative_list` | List artifacts |
| `creative_delete` | Delete artifact |

Stored in `creations/<type>/<name>.<ext>`.

---

## deliberate.py

`library/tools/deliberate.py` — multi-identity reasoning.

```python
response = generate_with_identity(
    ctx, identity: Identity, messages: list[dict],
    system: str, *, max_tokens=2048, caller="?"
) -> LLMResponse
# Calls ctx.llm with identity-formatted system prompt.

candidates = deliberate(
    ctx, identities: list[Identity], prompt: str,
    *, max_tokens=2048,
) -> list[str]
# Each identity generates a candidate response. Returns list of strings.

merged = recompose(
    ctx, candidates: list[str], composer_identity: Identity,
    context: str, *, max_tokens=4096,
) -> str
# Merge multiple candidates into one response using composer identity.
```

`Identity` (`library/tools/identity.py`):
```python
@dataclass
class Identity:
    name: str
    personality: str
    model: str | None = None     # overrides instance default
    provider: str | None = None  # currently ignored, uses instance provider
```

---

## voting.py

`library/tools/voting.py`

```python
winner, scores = borda_tally(votes: list[list[str]]) -> tuple[str, dict[str, int]]
# votes = list of ranked lists (first = top pick). Returns winner id + score map.

winners = approval_tally(votes: list[set[str]]) -> list[str]
# votes = list of approved sets. Returns candidates with majority approval.
```

---

## handle_tool() Dispatch Map

`library/tools/tools.py` — `handle_tool(ctx, name, arguments) -> tuple[str, bool]`

| Tool name pattern | Module | Notes |
|------------------|--------|-------|
| `read_file` | tools.py | returns file content |
| `write_file` | tools.py | auto-compacts via ctx.compact_file() |
| `list_files` | tools.py | |
| `send_message` | tools.py | calls ctx.send() |
| `list_rooms` | tools.py | calls ctx.get_all_space_contexts() |
| `send_to_instance` | tools.py | calls ctx.send_to() |
| `introspect` | tools.py | loads about.md + config_summary |
| `publish` | publish.py | publishes to data repo _published/ |
| `done` | tools.py | returns (summary, True) — terminates tool loop |
| `graph_*` | graph.py | `handle_graph_tool(ctx, name, args)` |
| `map_*` | activation_map.py | `handle_map_tool(ctx, name, args)` |
| `organize_*` | organize.py | `handle_organize_tool(ctx, name, args)` |
| `creative_*` | creative.py | `handle_creative_tool(ctx, name, args)` |
