# Symbiosis

A framework for building multi-layered LLM agent systems. Symbiosis separates infrastructure from behavior from state, enabling multi-instance and multi-provider deployments from a single codebase.

## Architecture

Three layers with strict separation:

- **Harness (Layer 1)** — Infrastructure: storage, scheduling, LLM provider routing, messaging adapters. Knows nothing about agent behavior.
- **Species (Layer 2)** — Stateless behavior code. Defines what an agent type does via entry point handlers that receive an `InstanceContext`.
- **Instance (Layer 3)** — Pure data: a unique ID, a namespaced storage directory, and config. Not code.

Species code interacts with infrastructure exclusively through `ctx` — no vendor SDK imports, no absolute paths, no direct HTTP calls.

## Quickstart

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Install

```bash
git clone <repo-url> && cd synthetic-runner
uv sync
```

### Configure

1. **Run the interactive setup wizard** (recommended):

```bash
uv run symbiosis setup
```

This configures `.env`, `config/harness.yaml`, and (if none exists) your first
instance plus a starter pipeline profile.

2. **Or configure manually**:

Create `.env` with your secrets:

```bash
ANTHROPIC_API_KEY=sk-ant-...
MATRIX_HOMESERVER=https://matrix.example.org
```

Edit `config/harness.yaml` — defines shared infrastructure (providers, adapters):

```yaml
providers:
  - id: anthropic
    type: anthropic
    api_key: ${ANTHROPIC_API_KEY}

  - id: lmstudio
    type: openai_compat
    base_url: http://localhost:1234/v1
    api_key: lm-studio

adapters:
  - id: matrix-main
    type: matrix
    homeserver: ${MATRIX_HOMESERVER}

storage_dir: instances
store_path: harness.db
poll_interval: 30
```

Providers and adapters are shared infrastructure — `${VAR}` references resolve from `.env` and environment variables at load time.

Create an instance config in `config/instances/`:

```yaml
# config/instances/my-agent.yaml
species: draum
provider: anthropic
model: claude-sonnet-4-6

messaging:
  adapter: matrix-main
  entity_id: "@my-agent:matrix.org"
  access_token: ${MY_AGENT_MATRIX_TOKEN}
  spaces:
    - name: main
      handle: "!roomid:matrix.org"

schedule:
  heartbeat: "0 * * * *"
```

Each instance gets its own `access_token` — multiple instances can use the same adapter (shared homeserver) with separate Matrix identities.

### Run

```bash
uv run symbiosis
# or
uv run python -m symbiosis
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--config`, `-c` | `config/harness.yaml` | Harness config path |
| `--base-dir`, `-d` | `.` | Base directory for storage |
| `--log-level` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |

Setup command:

```bash
uv run symbiosis setup
```

The scheduler loads all instance configs from `config/instances/`, registers their species, and runs a unified loop that:
- Polls messaging adapters for incoming events (reactive triggers)
- Checks cron schedules for timed entry points (scheduled triggers)

### Run tests

```bash
uv run pytest
```

## Creating a new species

A Species is a Python class that defines agent behavior. It exports a manifest telling the harness what entry points, tools, and default files it needs.

### 1. Create the species file

```python
# symbiosis/species/my_species.py

from symbiosis.species import Species, SpeciesManifest, EntryPoint

DEFAULT_FILES = {
    "notes.md": "# Notes\n",
}

def on_message(ctx, events):
    """Handle incoming messages."""
    memory = ctx.read("notes.md")

    response = ctx.llm(
        messages=[{"role": "user", "content": f"Events: {events}\nNotes: {memory}"}],
        system="You are a helpful agent.",
    )

    if response.message:
        ctx.send("main", response.message)

def heartbeat(ctx):
    """Periodic maintenance."""
    ctx.write("notes.md", "Updated at heartbeat.\n")

class MySpecies(Species):
    def manifest(self):
        return SpeciesManifest(
            species_id="my-species",
            entry_points=[
                EntryPoint(name="on_message", handler=on_message, trigger="message"),
                EntryPoint(name="heartbeat", handler=heartbeat, schedule="0 */6 * * *"),
            ],
            default_files=DEFAULT_FILES,
            spawn=self._spawn,
        )

    def _spawn(self, ctx):
        for path, content in DEFAULT_FILES.items():
            if not ctx.exists(path):
                ctx.write(path, content)
```

### 2. Register it in the loader

Add your species to `symbiosis/__main__.py`:

```python
def load_species(species_id):
    if species_id == "draum":
        from symbiosis.species.draum import DraumSpecies
        return DraumSpecies()
    if species_id == "my-species":
        from symbiosis.species.my_species import MySpecies
        return MySpecies()
    raise ValueError(f"Unknown species: {species_id}")
```

### 3. Create an instance config

```yaml
# config/instances/my-agent-1.yaml
species: my-species
provider: anthropic
model: claude-haiku-4-5-20251001

messaging:
  adapter: matrix-main
  entity_id: "@my-agent:matrix.org"
  access_token: ${MY_AGENT_TOKEN}
  spaces:
    - name: main
      handle: "!room:matrix.org"

schedule:
  heartbeat: "0 */6 * * *"
```

### InstanceContext API

All species code interacts with infrastructure through `ctx`:

| Method | Description |
|--------|-------------|
| `ctx.read(path)` | Read a memory file (scoped to instance) |
| `ctx.write(path, content)` | Write a memory file |
| `ctx.list(prefix)` | List files |
| `ctx.exists(path)` | Check if file exists |
| `ctx.config(key)` | Read-only instance config (`instance_id`, `species`, `model`, `entity_id`, ...) |
| `ctx.llm(messages, *, system, model, tools, ...)` | LLM call (uses instance default provider/model) |
| `ctx.send(space, message)` | Send a message to a logical space |
| `ctx.poll(space, since_token)` | Poll for new events |
| `ctx.get_space_context(space)` | Get room name, topic, members |
| `ctx.send_to(target_id, message)` | Inter-instance mailbox |
| `ctx.read_inbox()` | Read and clear inbox messages |
| `ctx.store(namespace)` | Instance-private SQLite key/value store |
| `ctx.shared_store(namespace)` | Species-shared SQLite key/value store |

### Reusable patterns

The toolkit provides patterns you can compose in your entry points:

```python
from symbiosis.toolkit.patterns import (
    gut_response,       # Quick assessment of incoming events
    plan_response,      # Deliberate planning step
    compose_response,   # Final message composition
    run_subconscious,   # Post-session self-assessment (writes subconscious.md)
    run_react,          # Translate subconscious into intentions (writes intentions.md)
    update_relationships,  # Update relationship tracking files
    distill_memory,     # Compress memory into a digest
    run_session,        # Tool-use session loop
)
```

All patterns take `ctx` as their first argument. Use them directly or build your own pipeline.

### YAML pipelines

Simple species can be defined declaratively without Python — see `IMPLEMENTATION.md` section 7 for the pipeline format.

## Running multiple instances

The scheduler runs all instances from a single process. Each instance config in `config/instances/` is loaded at startup.

### Same species, multiple instances

Create one config file per instance:

```
config/instances/
  agent-alice.yaml    # species: draum, entity_id: @alice:matrix.org
  agent-bob.yaml      # species: draum, entity_id: @bob:matrix.org
```

Each gets its own:
- **Storage namespace** — files at `instances/<instance-id>/memory/`
- **Messaging identity** — separate `access_token` and `entity_id`
- **SQLite store** — private via `ctx.store()`, shared within species via `ctx.shared_store()`
- **Mailbox** — inter-instance messages via `ctx.send_to()`/`ctx.read_inbox()`

### Different species in parallel

Mix species freely — the scheduler handles all of them:

```
config/instances/
  draum-1.yaml        # species: draum (persistent memory agent)
  worker-1.yaml       # species: hivemind-worker
  worker-2.yaml       # species: hivemind-worker
  coordinator.yaml    # species: hivemind-coordinator
```

### Concurrency model

- **Per-instance locking** — if two triggers fire for the same instance (e.g. a heartbeat and an incoming message), they run serially. One waits for the other to finish.
- **Cross-instance parallelism** — different instances run concurrently in separate threads.
- **Shared state** — instances coordinate through `ctx.shared_store()` (SQLite with atomic `claim`/`release`) or `ctx.send_to()` mailboxes.

### Provider mixing

Different instances can use different LLM providers:

```yaml
# agent-local.yaml — uses a local model
species: draum
provider: lmstudio
model: qwen2.5-72b

# agent-cloud.yaml — uses Anthropic
species: draum
provider: anthropic
model: claude-sonnet-4-6
```

Both providers are defined in `harness.yaml`; each instance picks one.

### Messaging without an external platform

For hivemind or testing setups, use the `local_file` adapter or skip messaging entirely and coordinate through mailboxes and the shared store:

```yaml
# harness.yaml
adapters:
  - id: local
    type: local_file
    base_dir: messages
```

```yaml
# worker.yaml — no messaging block at all
species: hivemind-worker
provider: lmstudio
model: local-model
```

Workers communicate via `ctx.send_to()` / `ctx.read_inbox()` and coordinate via `ctx.shared_store()`.
