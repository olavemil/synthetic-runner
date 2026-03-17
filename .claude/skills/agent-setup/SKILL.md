---
name: agent-setup
description: Guide for creating and configuring Symbiosis agents (instances, species, harness wiring). Use when creating new instances, designing new species, or debugging how harness components connect.
user_invocable: true
---

# Agent Setup Guide

## Creating a New Instance

Create `config/instances/<id>.yaml`. The filename stem becomes the `instance_id`.

```yaml
species: draum              # species_id (must match a registered species)
provider: anthropic         # provider key from harness.yaml providers list
model: claude-opus-4-6

messaging:
  adapter: matrix-main      # adapter key from harness.yaml adapters list
  entity_id: "@bot:matrix.org"
  access_token: ${BOT_MATRIX_TOKEN}  # resolved from .env
  spaces:
    - name: main            # logical name (used in ctx.send("main", ...))
      handle: "!room:matrix.org"

schedule:
  heartbeat: "0 * * * *"   # cron expression
  max_idle_heartbeats: 3    # skip heartbeat after N idle cycles (no messages)
  min_thinks_per_reply: 2   # minimum heartbeats between replies (throttle)
```

Species-specific config goes as top-level keys (see species-ref skill for details).

Instance configs are **gitignored** except `example.yaml`. Real tokens live in `.env`.

## Creating a New Species

### 1. Create the package

```
library/species/<species_id>/
  __init__.py    # Species class + handlers
  about.md       # Description (loaded by introspect tool)
```

### 2. Define the manifest

```python
from library.species import Species, SpeciesManifest, EntryPoint

class MySpecies(Species):
    def manifest(self) -> SpeciesManifest:
        return SpeciesManifest(
            species_id="my-species",
            entry_points=[
                EntryPoint(name="on_message", handler=on_message, trigger="message"),
                EntryPoint(name="heartbeat", handler=heartbeat, schedule="0 * * * *"),
            ],
            tools=[],               # extra tool schemas (usually empty — tools built in handlers)
            default_files={         # written on first spawn
                "memory.md": "# Memory\n",
            },
            spawn=None,             # optional setup callback(ctx)
        )
```

### 3. Write handlers

```python
def on_message(ctx: InstanceContext, events: list[Event]) -> None:
    # events = list of Event(event_id, sender, body, timestamp)
    # All infrastructure via ctx only — no direct imports
    response = ctx.llm(messages=[...], system="...", caller="on_message")
    ctx.send("main", response.message)

def heartbeat(ctx: InstanceContext) -> None:
    # Scheduled work — no messaging unless send policy allows it
    pass
```

### 4. Register the species

In `library/__main__.py`, species are auto-discovered from `library.species.*` subpackages. No manual registration needed — just make the class extend `Species`.

### Key rules for species code

- Never import `anthropic`, `openai`, or any vendor SDK
- Never construct absolute file paths — use `ctx.read/write` only
- Never call HTTP directly — use `ctx.llm()` and `ctx.send/poll()`
- Keep handlers stateless — all state in `ctx.write()` or `ctx.store()`

## How Harness Wires Things

```
symbiosis check:
  Checker.run()
    → polls messaging adapter for each instance
    → checks cron schedules
    → enqueues jobs into SQLite JobQueue

symbiosis work:
  Worker.run()
    → claims jobs from queue (respects provider concurrency slots)
    → for each job in a thread:
        config = registry.get_instance_config(instance_id)
        ctx = _build_context(config)   # constructs InstanceContext
        handler(ctx, **payload)        # calls on_message or heartbeat
        queue.complete(job)
```

`_build_context()` in `worker.py` wires:
- `NamespacedStorage` (files in `instances/<id>/memory/`)
- `LLMProvider` (looked up from harness config by provider id)
- `MessagingAdapter` (per-instance Matrix credentials)
- `Mailbox` (inter-instance messages)
- `Compactor` (if `compact:` section in harness.yaml)
- `AnalyticsClient` (if `analytics:` section)

## Adding a Provider

In `harness.yaml`:
```yaml
providers:
  - id: lmstudio
    type: openai_compat
    base_url: ${LMSTUDIO_BASE_URL}
    api_key: lm-studio
    max_concurrency: 1          # optional slot limit
  - id: anthropic
    type: anthropic
    api_key: ${ANTHROPIC_API_KEY}
```

Supported types: `openai_compat` (any OpenAI-compatible endpoint), `anthropic`.

## Data Sync & Publish

```yaml
# harness.yaml
sync:
  repo: ../synthetic-space    # relative path to data repo
  prefix: symbiosis
  branch: main
```

- `symbiosis work --sync` or `symbiosis sync` — copies `instances/*/memory/*.md` to data repo, commits + pushes.
- **Publish tool**: enable with `make_tools(ctx, {"publish": True})` — agent can call `publish(path, content)` to write to `_published/` in data repo.

## Related Skills

- `/ctx-api` — Full InstanceContext API reference + config format
- `/species-ref` — All 6 species with config keys and tool options
- `/tools-patterns` — make_tools options, patterns.py, phase scopes
- `neural-dreamer.md` (this directory) — Neural Dreamer architecture detail
