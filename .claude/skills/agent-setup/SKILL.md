---
name: agent-setup
description: Guide for creating and configuring Symbiosis agents (instances, species, harness wiring).
user_invocable: true
---

# Agent Setup Guide

## Three-Layer Architecture

Symbiosis uses three layers with strict separation:

1. **Harness** (`library/harness/`) — Infrastructure: config, storage, SQLite store, LLM providers, messaging adapters, job queue, checker, worker, scheduler. Knows nothing about agent behavior.

2. **Species** (`library/species/`) — Stateless behavior code. Defines what an agent type *does*. Exports a `SpeciesManifest` with entry points, tools, default files, and a spawn function. Handlers receive `(ctx: InstanceContext, ...)`.

3. **Instance** — Pure data: a YAML config file + namespaced file storage. No code. Each instance declares which species it uses and how it connects to infrastructure.

## Creating a New Instance

Create a YAML file in `config/instances/<id>.yaml`:

```yaml
species: draum              # Which species to use
provider: anthropic         # LLM provider key from harness.yaml
model: claude-opus-4-6      # Model ID

messaging:
  adapter: matrix-main      # Adapter key from harness.yaml
  entity_id: "@bot:matrix.org"
  access_token: ${BOT_MATRIX_TOKEN}  # Env var reference
  spaces:
    - name: main             # Logical space name (used in ctx.send("main", ...))
      handle: "!room:matrix.org"  # Platform-specific handle

schedule:
  heartbeat: "0 * * * *"    # Cron expression for heartbeat entry point
  max_idle_heartbeats: 3    # Skip heartbeat after N idle cycles
```

Species-specific config goes as top-level keys (e.g., `thrivemind:`, `hecate:`).

## Data Sync & Publish

Instance memory can be synced to a separate data repository. Configure in `harness.yaml`:

```yaml
sync:
  repo: ../synthetic-space    # path to data repo
  prefix: symbiosis           # subdirectory within the repo
  branch: main
```

- **Sync**: `symbiosis work --sync` or `symbiosis sync` copies `.md` files to `<repo>/<prefix>/<instance_id>/`, commits, and pushes.
- **Publish tool**: Agents can publish files to `_published/` in their data repo space. Enable with `make_tools(ctx, {"publish": True})`.
- **Post-heartbeat rendering**: Neural Dreamer auto-renders graph HTML and map PNG/GIF after heartbeat.
- OS schedule files auto-include `--sync` when sync is configured.

## Available Species

### Draum (`species: draum`)
Persistent memory agent with gut-plan-compose response pipeline.

**Entry points:** `on_message` (reactive), `heartbeat` (scheduled)
**Default files:** `thinking.md`, `project.md`, `sessions.md`, `scratchpad.md`, `sensitivity.md`, `intentions.md`, `subconscious.md`
**Config keys:** None beyond standard

### Thrivemind (`species: thrivemind`)
Colony-based deliberation. A colony of individuals with personality dimensions proposes, votes, and converges on responses. Individuals have evocative adjective-adjective-noun names, track age (sessions since spawn), and receive relative age context during reflection.

**Entry points:** `on_message` (reactive), `heartbeat` (scheduled, constitution updates + spawn cycle)
**Default files:** `constitution.md`, `sessions.md`
**Config keys:**
```yaml
thrivemind:
  min_colony_size: 8         # colony grows from this size
  max_colony_size: 16        # colony is trimmed to this size
  suggestion_fraction: 0.5
  approval_threshold: 3
  consensus_threshold: 0.6
  suggestion_model: ""       # provider/model for suggestions
  writer_model: ""           # provider/model for final composition
  voice_space: main
```

**Voting:** Each voter ranks candidates; voters who placed the winner in their **top 2** picks give +1 approval to the winner. Voters whose top pick was not the winner lose -1. This dual-pick system makes positive approval more achievable.

**Spawn cycle:** When colony is below max, eligible parents survive and offspring are added. At max, eligible parents are replaced by offspring. Colony is always kept within min/max bounds.

### Hecate (`species: hecate`)
Multi-voice deliberation with named identity voices that think, vote, and compose responses.

**Entry points:** `on_message` (reactive), `heartbeat` (scheduled)
**Config keys:** `hecate.voices` (list of voice definitions with name, model, personality)

### Subconscious Dreamer (`species: subconscious_dreamer`)
Three-phase thinking (active thinking → subconscious → dreaming) and three-phase response (intuition → worry → action). Uses declarative YAML pipelines.

**Entry points:** `on_message` (reactive), `heartbeat` (scheduled)
**Default files:** `thinking.md`, `dreams.md`, `concerns_and_ideas.md`
**Config keys:** None beyond standard

### Neural Dreamer (`species: neural_dreamer`) — In Progress
Extends Subconscious Dreamer with dual neural networks (fast + slow) that control prompt segment selection and variable injection. Adds semantic graph and activation map memory tools.

**Entry points:** `on_message` (fast cycle: gut → suggest → reply → review), `heartbeat` (slow cycle: think + sleep)
**Default files:** `thinking.md`, `dreams.md`, `concerns_and_ideas.md`, net checkpoints (binary)
**Config keys:** `neural_dreamer.fast_net`, `neural_dreamer.slow_net`, `neural_dreamer.graph`, `neural_dreamer.activation_map`

See `neural-dreamer.md` in this skill directory for full architecture details.

## How Species Define Behavior

A species is a class extending `Species` that returns a `SpeciesManifest`:

```python
from library.species import Species, SpeciesManifest, EntryPoint

class MySpecies(Species):
    def manifest(self) -> SpeciesManifest:
        return SpeciesManifest(
            species_id="my-species",
            entry_points=[
                EntryPoint(name="on_message", handler=on_message, trigger="message"),
                EntryPoint(name="heartbeat", handler=heartbeat, schedule="*/15 * * * *"),
            ],
            tools=[],
            default_files={"memory.md": "# Memory\n"},
            spawn=self._spawn,
        )
```

**Key rules:**
- Species never import vendor SDKs, construct absolute paths, or call HTTP directly
- All infrastructure access goes through `InstanceContext` (the `ctx` parameter)
- Species are stateless — all state lives in instance storage or SQLite store

## How Harness Wires Them

1. **Config loading** (`harness/config.py`): Reads `harness.yaml` for providers/adapters, scans `config/instances/` for instance YAML files
2. **Checker** (`harness/checker.py`): Polls messaging adapters + checks cron schedules, enqueues jobs into SQLite queue
3. **Worker** (`harness/worker.py`): Drains job queue, constructs `InstanceContext` for each job, calls species handler
4. **InstanceContext** (`harness/context.py`): The only interface between species and harness — provides `ctx.read/write/list/exists`, `ctx.llm()`, `ctx.send/poll()`, `ctx.store()`, `ctx.get_space_context()`, `ctx.get_all_space_contexts()`

## InstanceContext API Summary

| Method | Description |
|--------|-------------|
| `ctx.read(path)` / `ctx.write(path, content)` | Namespaced file storage |
| `ctx.list(prefix)` / `ctx.exists(path)` | File listing and existence |
| `ctx.llm(messages, ...)` | LLM call (returns `LLMResponse`) |
| `ctx.send(space, message)` / `ctx.poll(space)` | Messaging via logical space names |
| `ctx.send_to(target_id, message)` / `ctx.read_inbox()` | Inter-instance mailboxes |
| `ctx.store(namespace)` / `ctx.shared_store(namespace)` | SQLite key-value store |
| `ctx.get_space_context(space)` | Room metadata (name, topic, members) |
| `ctx.get_all_space_contexts()` | All rooms' metadata |
| `ctx.config(key)` | Read instance config values |
