# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Symbiosis toolkit** — a framework for building multi-layered LLM agent systems. The implementation is complete and tested.

Key documents:
- `ARCHITECTURE.md` — Three-layer architecture specification
- `IMPLEMENTATION.md` — Technical implementation roadmap
- `proposals/neural_nets/` — Hybrid NN/LLM extension proposals (architecture + memory tools)

## Commands

```bash
uv run pytest                  # run all tests
uv run pytest tests/test_X.py  # run specific test file
uv run symbiosis check         # poll adapters, enqueue jobs (lightweight)
uv run symbiosis work          # drain job queue run-to-empty
uv run symbiosis tick          # check + work (one cycle, for testing)
uv run symbiosis schedule      # print OS scheduler config files
uv run symbiosis setup         # interactive setup wizard
uv run symbiosis run           # legacy continuous loop
```

## Architecture

Three layers with strict separation:

- **Harness** (`symbiosis/harness/`) — Infrastructure: config, storage, SQLite store, LLM providers, messaging adapters, job queue, checker, worker, scheduler.
- **Species** (`symbiosis/species/`) — Stateless behavior. Exports a `SpeciesManifest` with `entry_points`, `tools`, `default_files`, `spawn`. Handlers take `(ctx: InstanceContext, ...)`.
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
  harness.yaml          # providers (with max_concurrency), adapters, scheduler settings
  instances/
    <id>.yaml           # species, provider, model, messaging (with access_token), schedule
.env                    # secrets: API keys, Matrix tokens
```

- `config/instances/example.yaml` is committed; real instance configs are gitignored.
- Per-instance Matrix `access_token` lives in instance YAML (not shared adapter config).
- `schedule.max_idle_heartbeats: N` limits heartbeat runs when no messages received.

## Key Files

| File | Purpose |
|------|---------|
| `symbiosis/harness/config.py` | YAML loading with `${VAR}` env resolution |
| `symbiosis/harness/context.py` | `InstanceContext` — main Species interface |
| `symbiosis/harness/store.py` | `StoreDB` / `NamespacedStore` (SQLite, `claim`/`release` for atomic ops) |
| `symbiosis/harness/jobqueue.py` | `JobQueue` — enqueue/claim/complete with instance guards |
| `symbiosis/harness/checker.py` | `Checker` — poll + schedule check → enqueue |
| `symbiosis/harness/worker.py` | `Worker` — run-to-empty with provider slot management |
| `symbiosis/harness/scheduler.py` | Legacy continuous loop (kept for `symbiosis run`) |
| `symbiosis/scheduling.py` | OS schedule file generation (launchd/systemd/crontab) |
| `symbiosis/toolkit/patterns.py` | `gut_response`, `plan_response`, `compose_response`, etc. |
| `symbiosis/toolkit/tools.py` | `make_tools()` / `handle_tool()` |
| `symbiosis/toolkit/pipeline.py` | YAML pipeline runner |
| `symbiosis/toolkit/segments.py` | Prompt segment registry, selection vectors, variable injection |
| `symbiosis/toolkit/graph.py` | Semantic graph tools (persistent directed weighted graph) |
| `symbiosis/toolkit/activation_map.py` | 2D activation map tools (attention/affect heatmap) |
| `symbiosis/toolkit/neural.py` | Fast/slow neural net management, checkpoints, signal encoding |
| `symbiosis/toolkit/rendering.py` | Graph→HTML, Map→PNG/GIF visualization |
| `symbiosis/species/draum.py` | Draum species: reactive + heartbeat |
| `symbiosis/species/subconscious_dreamer/` | Subconscious Dreamer: three-phase thinking + response pipelines |
| `symbiosis/species/neural_dreamer/` | Neural Dreamer: NN-driven prompt configuration + memory tools |
| `symbiosis/__main__.py` | CLI entry point |
| `symbiosis/setup_wizard.py` | Interactive setup wizard |

## Neural Net Extension (In Progress)

Spec: `proposals/neural_nets/architecture-technical.md` + `proposals/neural_nets/tools-memory-representations.md`

Adds a dual neural network layer and non-textual memory tools on top of the existing pipeline architecture. All new code is **toolkit modules** (opt-in by species), with one small harness addition (`read_binary`/`write_binary`).

### Implementation order

1. Add `ctx.read_binary`/`write_binary` to harness (`context.py`, `storage.py`)
2. `toolkit/segments.py` — segment registry + selection (manual weights first, NN-driven later)
3. `toolkit/graph.py` — semantic graph tools, JSON storage via `ctx.write`
4. `toolkit/activation_map.py` — 2D map tools, JSON storage
5. Register graph/map tools in `tools.py`
6. Create `neural_dreamer` species with pipelines, manual segment selection
7. `toolkit/neural.py` — PyTorch fast/slow nets, checkpoint save/load
8. Wire NN→segment selection + variable injection
9. Add review phase to fast cycle; wire LLM→NN structured signals
10. `toolkit/rendering.py` — graph HTML, map PNG/GIF
11. NN encoding of graph/map (GNN for graph→slow net, CNN/flatten for map→fast net)

### Key design decisions

- **Toolkit not harness:** NN, graph, map, segments are behavioral capabilities, not infrastructure
- **Binary storage:** Small nets (<64 hidden units) stored as binary files via `ctx.read_binary`/`write_binary`
- **Two nets:** Fast net (shallow, updates per message review) encodes session affect; slow net (deeper, updates at sleep) encodes accumulated disposition
- **Segment selection:** NN outputs control which pre-written prompt segments are active and in what order — fast net controls state/relational/meta segments, slow net controls identity/temporal/task segments
- **Variable injection:** Named floats (`tone_warmth`, `verbosity`, `confidence`, etc.) injected into active segments for continuous modulation
- **Graph + map tools:** Available to any species via `make_tools(ctx, {"graph": True, "activation_map": True})`; graph feeds slow net, map feeds fast net
