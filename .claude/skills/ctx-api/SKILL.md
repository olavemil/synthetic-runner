---
name: ctx-api
description: Complete InstanceContext API reference, harness.yaml config format, instance YAML format, and provider/adapter setup for the Symbiosis toolkit. Use when working with ctx methods, configuring providers, or wiring harness components.
user_invocable: true
---

# InstanceContext API Reference

`InstanceContext` (`library/harness/context.py`) is the only interface between species code and infrastructure.

## Storage

```python
ctx.read(path: str) -> str               # returns "" if not found
ctx.write(path: str, content: str)       # creates parent dirs automatically
ctx.list(prefix: str = "") -> list[str]  # paths under prefix
ctx.exists(path: str) -> bool

ctx.read_binary(path: str) -> bytes | None
ctx.write_binary(path: str, data: bytes)
```

Paths are relative to `instances/<instance_id>/memory/`. No absolute paths needed.

## Auto-compaction

```python
ctx.compact_file(content: str, path: str | None = None) -> str | None
```

Returns compacted string if content exceeds `compact.threshold_chars` in harness.yaml (default 6000), else `None`. Uses the configured compact provider (devstral). Called automatically in `handle_tool("write_file")` and `append_thinking` in patterns.py. Call explicitly before writing large content:

```python
compacted = ctx.compact_file(new_content, path="thinking.md")
ctx.write("thinking.md", compacted or new_content)
```

## LLM

```python
response = ctx.llm(
    messages: list[dict],        # OpenAI-format message list
    *,
    model: str | None = None,    # overrides instance default
    system: str | None = None,   # system prompt
    tools: list[dict] | None,    # OpenAI function schemas
    tool_choice: str | dict | None,
    max_tokens: int = 4096,
    caller: str = "?",           # logged in analytics
) -> LLMResponse
```

`LLMResponse` fields: `.message: str`, `.tool_calls: list[ToolCall]`, `.finish_reason: str`, `.usage: dict`

`ToolCall` fields: `.id: str`, `.name: str`, `.arguments: dict`

Retries up to 3× on pathological responses. Provider is always the instance's configured provider (no runtime switching).

## Messaging

```python
event_id = ctx.send(space: str, message: str, reply_to: str | None = None) -> str
events, next_token = ctx.poll(space: str, since_token: str | None = None)
ctx_data = ctx.get_space_context(space: str) -> dict  # name, topic, members
all_contexts = ctx.get_all_space_contexts() -> dict[str, dict]
spaces = ctx.list_spaces() -> list[str]
```

`space` is a logical name (e.g. `"main"`) mapped to a platform handle in instance YAML.
`events` is `list[Event]`: `.event_id`, `.sender`, `.body`, `.timestamp`.

Send policy (set by Worker per phase — do not change in species code):
```python
ctx.configure_send_policy(allow_send: bool, max_sends: int | None, reason: str)
ctx.sent_message_count  # read-only
ctx.can_send_reply() -> bool  # checks min_thinks_per_reply throttle
```

## Inter-instance

```python
ctx.send_to(target_instance_id: str, message: str)
messages = ctx.read_inbox() -> list[dict]  # {"sender": ..., "message": ..., "timestamp": ...}
```

## Structured Store (SQLite)

```python
store = ctx.store(namespace: str)         # scoped to this instance
store = ctx.shared_store(namespace: str)  # shared across all instances of same species
```

`NamespacedStore` methods: `.get(key)`, `.put(key, value)`, `.delete(key)`, `.list() -> list[str]`, `.claim(key, value, ttl)` / `.release(key)` (atomic ops).

## Config Access

```python
ctx.config(key: str)  # reads from instance YAML: "instance_id", "species", "provider", "model", "entity_id", or any extra key
ctx.config_summary() -> dict  # all non-secret config as dict
ctx.instance_id  # property
ctx.species_id   # property
```

## Analytics

```python
ctx.track(event_name: str, properties: dict | None)
```

---

## harness.yaml Full Structure

```yaml
providers:
  - id: lmstudio
    type: openai_compat           # openai_compat | anthropic
    base_url: ${LMSTUDIO_BASE_URL}
    api_key: lm-studio
    max_concurrency: 1            # optional provider slot limit

  - id: anthropic
    type: anthropic
    api_key: ${ANTHROPIC_API_KEY}

adapters:
  - id: matrix-main
    type: matrix                  # matrix | local_file
    homeserver: ${MATRIX_HOMESERVER}

storage_dir: instances            # base dir for instance file storage
store_path: harness.db            # SQLite path

poll_interval: 30                 # legacy scheduler poll interval (seconds)

scheduler:
  check_interval: 300             # symbiosis check frequency (seconds)
  work_interval: 60               # symbiosis work frequency (seconds)
  log_file: logs/symbiosis.log

sync:
  repo: ../synthetic-space        # path to data repo (relative ok)
  prefix: symbiosis               # subdir within data repo
  branch: main
  remote: https://...             # optional, for initial clone

analytics:
  base_url: http://localhost:8000
  api_key: ${ANALYTICS_API_KEY}

compact:
  provider: lmstudio              # provider id from providers list
  model: mistralai/devstral-small-2-2512
  threshold_chars: 2048           # trigger when content > this many chars
```

`${VAR}` references are resolved from `.env` (project root) + `os.environ`. OS env takes precedence.

## Instance YAML Full Structure

```yaml
# config/instances/<instance_id>.yaml
species: draum
provider: anthropic
model: claude-opus-4-6

messaging:
  adapter: matrix-main
  entity_id: "@bot:matrix.org"
  access_token: ${BOT_TOKEN}      # per-instance token
  spaces:
    - name: main                  # logical name used in ctx.send/poll
      handle: "!roomid:matrix.org"
    - name: journal
      handle: "!otherroomid:matrix.org"

schedule:
  heartbeat: "0 * * * *"         # cron expression (or omit for message-only)
  max_idle_heartbeats: 3         # skip heartbeat if no messages for N cycles
  min_thinks_per_reply: 2        # throttle: wait N heartbeats between replies

# Species-specific top-level keys (see species-ref for details):
# thrivemind:
# hecate:
# neural_dreamer:
# consilium:
```

Extra keys (anything not `species`, `provider`, `model`, `messaging`, `schedule`) are accessible via `ctx.config(key)`.

## Provider Types

**`openai_compat`** — any OpenAI-compatible API. `base_url` required. Set `api_key` to a dummy string if the endpoint doesn't require auth. Supports tool use via OpenAI function calling format.

**`anthropic`** — Anthropic Claude API. `api_key` required. No `base_url` needed.

Both providers implement `LLMProvider.create(model, messages, system, tools, tool_choice, max_tokens, caller) -> LLMResponse`.
