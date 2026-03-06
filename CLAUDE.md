# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Symbiosis toolkit** — a framework for building multi-layered LLM agent systems. The implementation is complete and tested.

Key documents:
- `ARCHITECTURE.md` — Three-layer architecture specification
- `IMPLEMENTATION.md` — Technical implementation roadmap

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
| `symbiosis/species/draum.py` | Draum species: reactive + heartbeat |
| `symbiosis/__main__.py` | CLI entry point |
| `symbiosis/setup_wizard.py` | Interactive setup wizard |
