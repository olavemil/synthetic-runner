# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Symbiosis toolkit** — a framework for building multi-layered LLM agent systems. The implementation is complete and tested.

Key documents:
- `docs/architecture.md` — Three-layer architecture specification
- `docs/implementation.md` — Technical implementation roadmap
- `proposals/neural_nets/` — Hybrid NN/LLM extension proposals (architecture + memory tools)

## Commands

```bash
uv run pytest                  # run all tests
uv run pytest tests/test_X.py  # run specific test file
uv run symbiosis check         # poll adapters, enqueue jobs (lightweight)
uv run symbiosis work          # drain job queue run-to-empty
uv run symbiosis work --sync   # drain queue, then sync memory to data repo
uv run symbiosis tick          # check + work (one cycle, for testing)
uv run symbiosis sync          # sync instance memory to data repo
uv run symbiosis schedule      # print OS scheduler config files
uv run symbiosis setup         # interactive setup wizard
uv run symbiosis run           # legacy continuous loop
```

## Architecture

Three layers with strict separation:

- **Harness** (`library/harness/`) — Infrastructure: config, storage, SQLite store, LLM providers, messaging adapters, job queue, checker, worker, scheduler.
- **Species** (`library/species/`) — Stateless behavior. Exports a `SpeciesManifest` with `entry_points`, `tools`, `default_files`, `spawn`. Handlers take `(ctx: InstanceContext, ...)`.
- **Instance** — Pure data: a YAML config + namespaced file storage. No code.

**InstanceContext** is the only interface between Species and Harness:
- `ctx.read/write/list/exists` — namespaced file storage
- `ctx.read_binary/write_binary` — binary file storage (PyTorch checkpoints, etc.)
- `ctx.llm(messages, ...)` → `LLMResponse`
- `ctx.send/poll(space, ...)` — messaging via logical space names
- `ctx.send_to/read_inbox` — inter-instance mailboxes
- `ctx.store/shared_store(namespace)` → `NamespacedStore` (SQLite)
- `ctx.get_space_context(space)` — room metadata
- `ctx.list_spaces()` — available logical space names

**Species never** import vendor SDKs, construct absolute paths, or call HTTP directly.

## Scheduling Architecture

The preferred deployment model uses OS scheduling (`launchd`/`systemd`/`cron`) to run two short-lived commands:

- **`check`** (every ~5 min): polls messaging adapters + checks cron schedules → enqueues jobs into SQLite queue. No LLM calls.
- **`work`** (every ~1 min): drains job queue run-to-empty using threads. Respects provider concurrency limits.

State is persisted in SQLite (`harness.db`):
- Sync tokens per instance/space
- Schedule next-fire times
- Idle heartbeat counts (`max_idle_heartbeats` in instance YAML throttles heartbeats when no messages)
- Job queue with instance guards (prevents duplicate concurrent jobs per instance)
- Provider slots for concurrency control (`max_concurrency` in harness.yaml)

Generate OS schedule files with `symbiosis schedule` (reads `scheduler.check_interval` / `scheduler.work_interval` from `harness.yaml`).

## Config Structure

```
config/
  harness.yaml          # providers (with max_concurrency), adapters, scheduler, sync settings
  instances/
    <id>.yaml           # species, provider, model, messaging (with access_token), schedule
.env                    # secrets: API keys, Matrix tokens
```

- `config/instances/example.yaml` is committed; real instance configs are gitignored.
- Per-instance Matrix `access_token` lives in instance YAML (not shared adapter config).
- `schedule.max_idle_heartbeats: N` limits heartbeat runs when no messages received.

## Data Sync & Publish

Instance memory files (`.md`) can be automatically synced to a separate data repository for external visibility (e.g. GitHub Pages).

Configure in `harness.yaml`:

```yaml
sync:
  repo: ../synthetic-space    # path to data repo (can be relative)
  prefix: symbiosis           # subdirectory within the data repo
  branch: main
```

- **Sync** (`library/sync.py`): Copies `.md` files from `instances/*/memory/` to `<repo>/<prefix>/<instance_id>/`. Commits and pushes automatically. Triggered by `symbiosis sync` or `symbiosis work --sync`.
- **Publish** (`library/publish.py`): Agents can publish rendered artefacts (reports, graphs, maps) to `_published/` within their data repo space, separate from memory files. Enabled via `make_tools(ctx, {"publish": True})`.
- **Post-heartbeat rendering**: Neural Dreamer automatically renders `graph.html`, `map.png`, and `map_session.gif` after heartbeat and publishes them to the data repo.
- Generated OS schedule files (`symbiosis schedule`) include `--sync` when sync is configured.

## Key Files

| File | Purpose |
|------|---------|
| `library/harness/config.py` | YAML loading with `${VAR}` env resolution |
| `library/harness/context.py` | `InstanceContext` — main Species interface |
| `library/harness/store.py` | `StoreDB` / `NamespacedStore` (SQLite, `claim`/`release` for atomic ops) |
| `library/harness/jobqueue.py` | `JobQueue` — enqueue/claim/complete with instance guards |
| `library/harness/checker.py` | `Checker` — poll + schedule check → enqueue |
| `library/harness/worker.py` | `Worker` — run-to-empty with provider slot management |
| `library/harness/scheduler.py` | Legacy continuous loop (kept for `symbiosis run`) |
| `library/scheduling.py` | OS schedule file generation (launchd/systemd/crontab) |
| `library/tools/patterns.py` | `gut_response`, `plan_response`, `compose_response`, etc. |
| `library/tools/tools.py` | `make_tools()` / `handle_tool()` |
| `library/tools/pipeline.py` | YAML pipeline runner |
| `library/tools/segments.py` | Prompt segment registry, selection vectors, variable injection |
| `library/tools/graph.py` | Semantic graph tools (persistent directed weighted graph) |
| `library/tools/activation_map.py` | 2D activation map tools (attention/affect heatmap) |
| `library/tools/neural.py` | Fast/slow neural net management, checkpoints, signal encoding |
| `library/tools/rendering.py` | Graph→HTML, Map→PNG/GIF visualization |
| `library/sync.py` | Instance memory sync to data repo |
| `library/publish.py` | Publish artefacts to data repo `_published/` + post-heartbeat rendering |
| `library/species/draum/` | Draum species: reactive + heartbeat |
| `library/species/subconscious_dreamer/` | Subconscious Dreamer: three-phase thinking + response pipelines |
| `library/species/neural_dreamer/` | Neural Dreamer: NN-driven prompt configuration + memory tools |
| `library/__main__.py` | CLI entry point |
| `library/setup_wizard.py` | Interactive setup wizard |

## Neural Net Extension

Spec: `proposals/neural_nets/architecture-technical.md` + `proposals/neural_nets/tools-memory-representations.md`

Adds a dual neural network layer and non-textual memory tools on top of the existing pipeline architecture. All new code is **tools modules** (opt-in by species), with one small harness addition (`read_binary`/`write_binary`).

### Key design decisions

- **Tools not harness:** NN, graph, map, segments are behavioral capabilities, not infrastructure
- **Binary storage:** Small nets (<64 hidden units) stored as binary files via `ctx.read_binary`/`write_binary`
- **Two nets:** Fast net (shallow, updates per message review) encodes session affect; slow net (deeper, updates at sleep) encodes accumulated disposition
- **Segment selection:** NN outputs control which pre-written prompt segments are active and in what order — fast net controls state/relational/meta segments, slow net controls identity/temporal/task segments
- **Variable injection:** Named floats (`tone_warmth`, `verbosity`, `confidence`, etc.) injected into active segments for continuous modulation
- **Graph + map tools:** Available to any species via `make_tools(ctx, {"graph": True, "activation_map": True})`; graph feeds slow net, map feeds fast net
