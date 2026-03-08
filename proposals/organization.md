# Repository Organization Proposal

## Overview

This proposal reorganizes the Symbiosis repository to reflect and emphasize the three-layer architecture described in `ARCHITECTURE.md`: **Harness** (infrastructure), **Species** (behavior), and **Instance** (state/data).

Currently, code is organized functionally within `symbiosis/`:
```
symbiosis/
  harness/        # Layer 1: Infrastructure
  species/        # Layer 2: Behavior (species implementations)
  toolkit/        # Layer 2: Shared behavioral patterns and tools
```

The proposed reorganization does two things:

1. **Clarifies layer separation** by renaming `symbiosis/` → `library/` and establishing explicit divisions within each layer
2. **Organizes species templates** with consistent structure (docs, prompts, pipelines, tests)
3. **Segregates instance state** (configs + memory) into a separate top-level `instances/` directory

---

## Proposed Structure

### Top Level

```
library/              # Layer 1+2: Shared infrastructure + behavioral toolkit
  harness/           # Layer 1: Infrastructure (read CLAUDE.md)
  tools/             # Layer 2: Reusable behavioral patterns and tools
  species/           # Layer 2: Species definitions (templates)

config/              # Configuration files (gitignored except templates)
  harness.yaml
  instances/         # Instance YAML configs (read-only at runtime)
    example.yaml     # Template, committed
    custom.yaml      # Instance-specific, gitignored

instances/           # Layer 3: Instance state (runtime data + memory)
  {instance-id}/
    inbox/
    memory/
    storage.db       # Instance-scoped SQLite (optional, future)

tests/               # Test suite for all layers
  unit/              # Harness and toolkit unit tests
  integration/       # Species tests, end-to-end
  fixtures/          # Test data and mock configs

docs/                # Project documentation
  architecture.md    # (move from root)
  implementation.md  # (move from root)
```

### Layer 1: Harness (`library/harness/`)

Infrastructure for all species. Currently flat; propose organizing by concern:

```
library/harness/
  __init__.py
  config.py          # YAML loading, env var resolution
  context.py         # InstanceContext (main Species↔Harness interface)
  registry.py        # Species loader and manifest registry
  
  storage/           # File storage abstraction
    __init__.py
    local.py         # Namespaced local file storage
    # future: cloud storage backends
  
  providers/         # LLM provider abstraction (normalize Anthropic, OpenAI, local)
    __init__.py
    anthropic.py
    openai_compat.py
    # future: more providers
  
  adapters/          # Messaging platform adapters (Matrix, Slack, local, etc.)
    __init__.py
    matrix.py
    local_file.py
    # future: more adapters
  
  queue/             # Job queue + scheduling
    __init__.py
    jobqueue.py      # SQLite-backed job queue
    checker.py       # Poll adapters + check schedules → enqueue
    worker.py        # Drain queue with provider concurrency control
    scheduler.py     # Legacy continuous loop runner
  
  store/             # Structured key-value store (SQLite)
    __init__.py
    store.py
    mailbox.py       # Inter-instance mailbox logic
  
  # Core orchestration
  scheduler.py       # Legacy main loop (keep for `symbiosis run`)
```

### Layer 2a: Toolkit (`library/tools/`)

Shared behavioral patterns and tools used by multiple species. Currently scattered in `symbiosis/toolkit/`.

```
library/tools/
  __init__.py
  make_tools()       # Tool registry and dispatcher
  
  patterns/          # Response pipeline patterns
    __init__.py
    response.py      # gut_response, plan_response, compose_response
    memory.py        # distill_memory, run_session, etc.
  
  prompts/           # Prompt segment registry, selection vectors
    __init__.py
    segments.py      # Prompt segment system with variables
  
  memory/            # Non-textual memory representations
    __init__.py
    graph.py         # Semantic graph (persistent directed graph)
    activation_map.py # 2D activation heatmap (attention/affect)
  
  neural/            # Fast/slow dual neural nets (opt-in, future)
    __init__.py
    neural.py        # PyTorch nets, checkpoint save/load
    encoding.py      # Graph/map encoding for nets
  
  rendering/         # Visualization tools
    __init__.py
    graph_render.py  # Graph → HTML
    map_render.py    # Activation map → PNG/GIF
  
  pipeline/          # YAML-driven pipeline runner
    __init__.py
    pipeline.py      # Pipeline executor
  
  # Behavioral utilities (species-specific, but here for visibility)
  deliberate.py      # Multi-agent deliberation patterns
  voting.py          # Consensus voting
  identity.py        # Identity and relationship tracking
```

### Layer 2b: Species Templates (`library/species/`)

Each species is a complete behavior template with documentation, prompts, and tests.

```
library/species/
  __init__.py
  base.py            # SpeciesManifest, EntryPoint (move from context.py)
  
  draum/             # Persistent memory agent (gut→plan→compose)
    __init__.py
    species.py       # DraumSpecies class + entry point handlers
    about.md         # Draum overview and design doc
    entries/
      on_message.yaml
      heartbeat.yaml
    prompts/
      gut.md
      plan.md
      compose.md
  
  hecate/            # Multi-voice deliberation
    __init__.py
    species.py
    about.md
    entries/
      on_message.yaml
      heartbeat.yaml
    prompts/
      vote.md
      voice_*.md      # Per-voice prompts
  
  thrivemind/        # Colony-based consensus
    __init__.py
    species.py
    about.md
    entries/
      on_message.yaml
      heartbeat.yaml
    prompts/
      colony.md
  
  subconscious_dreamer/  # Three-phase thinking + response
    __init__.py
    species.py
    about.md
    entries/
      on_message.yaml
      heartbeat.yaml
    prompts/
      active_thinking.md
      subconscious.md
      dreaming.md
  
  neural_dreamer/    # NN-driven segment selection (future)
    __init__.py
    species.py
    about.md
    entries/
      on_message.yaml
      heartbeat.yaml
    prompts/
      segments/       # Pre-written prompt segments
        identity.md
        state.md
        task.md
        etc.
```

---

## Mapping: Current → Proposed

| Current Path | Proposed Path | Notes |
|--------------|---------------|-------|
| `symbiosis/` | `library/` | Rename to emphasize "shared library" |
| `symbiosis/harness/` | `library/harness/` | Reorganized by concern (storage, provider, adapter, queue) |
| `symbiosis/harness/adapters/` | `library/harness/adapters/` | Same location, clearer parent structure |
| `symbiosis/harness/providers/` | `library/harness/providers/` | Same location, clearer parent structure |
| `symbiosis/toolkit/` | `library/tools/` | Reorganized: patterns/, prompts/, memory/, neural/, rendering/ |
| `symbiosis/species/` | `library/species/` | Each species gets `about.md`, `entries/`, `prompts/` substructure |
| `instances/` | `instances/` | Same top-level location; state (inbox, memory) already here |
| `config/` | `config/` | Already top-level; proposal adds `config/harness.yaml` organization |
| `.env` | `.env` | Unchanged; secrets only |

---

## Benefits

1. **Clearer Layer Separation**: `library/` clearly holds infrastructure (harness) + behavior (tools, species). `config/` and `instances/` are distinct.

2. **Species as Templates**: Each species in a subdirectory with standardized `about.md`, `entries/`, `prompts/` structure makes it easy to:
   - Understand what a species does (`about.md`)
   - See entry point handlers and pipelines (`entries/`)
   - Reuse and adapt prompts (`prompts/`)
   - Add species-specific tests in `tests/integration/`

3. **Fewer Namespace Collisions**: `toolkit/graph.py`, `toolkit/segments.py` etc. are now under `tools/memory/` and `tools/prompts/`, avoiding confusion with species-specific modules like `species/thrivemind.py`.

4. **Easier Navigation**: A new contributor can:
   - Read `docs/architecture.md` from top-level docs (clearer than root ARCHITECTURE.md)
   - Find infrastructure code in `library/harness/`
   - Find behavior patterns in `library/tools/`
   - See all species templates in `library/species/`
   - Find tests matching the structure (`tests/unit/`, `tests/integration/`)

5. **Scalability**: Adding a new species is straightforward: create `library/species/my-species/` with standard subdirectories.

---

## Implementation Notes

- **Backwards compatibility**: The `symbiosis/` package still imports from the right places. Internal module paths can be updated gradually.
- **Entry point discovery**: Species loader in `library/species/__init__.py` or a new `loader.py` can auto-discover species subdirectories.
- **Tests**: Move test organization to mirror source structure (`tests/unit/harness/`, `tests/unit/tools/`, `tests/integration/species/`).
- **Documentation**: Move root-level architecture docs to `docs/` subdirectory; update all cross-references.
