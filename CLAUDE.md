# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Symbiosis toolkit** — a framework for building multi-layered LLM agent systems with 6 species, all fully implemented and tested (~800 tests).

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
```

## Architecture

Three layers with strict separation:

- **Harness** (`library/harness/`) — Infrastructure: config, storage, SQLite store, LLM providers, adapters, job queue, checker, worker.
- **Species** (`library/species/`) — Stateless behavior. Exports a `SpeciesManifest` with `entry_points`, `tools`, `default_files`, `spawn`. Handlers take `(ctx: InstanceContext, ...)`.
- **Instance** — Pure data: a YAML config + namespaced file storage. No code.

**Species never** import vendor SDKs, construct absolute paths, or call HTTP directly.

## InstanceContext API (ctx)

The only interface between Species and Harness:

| Method | Description |
|--------|-------------|
| `ctx.read(path)` / `ctx.write(path, content)` | Namespaced text file storage |
| `ctx.list(prefix)` / `ctx.exists(path)` | File listing / existence check |
| `ctx.read_binary(path)` / `ctx.write_binary(path, data)` | Binary storage (PyTorch checkpoints) |
| `ctx.compact_file(content, path=None)` | Compact via devstral if over threshold; returns compacted str or None |
| `ctx.llm(messages, *, model, system, tools, max_tokens, caller)` | LLM call → `LLMResponse` |
| `ctx.send(space, message, reply_to)` / `ctx.poll(space)` | Messaging via logical space names |
| `ctx.send_to(target_id, msg)` / `ctx.read_inbox()` | Inter-instance mailboxes |
| `ctx.store(ns)` / `ctx.shared_store(ns)` | `NamespacedStore` (SQLite, per-instance / per-species) |
| `ctx.get_space_context(space)` / `ctx.list_spaces()` | Room metadata / available space names |
| `ctx.config(key)` | Read instance config values |
| `ctx.configure_send_policy(allow_send, max_sends, reason)` | Messaging throttle (set by Worker) |

## Config Structure

```
config/
  harness.yaml          # providers (+ max_concurrency), adapters, scheduler, sync, compact
  instances/
    <id>.yaml           # species, provider, model, messaging, schedule, species-specific keys
.env                    # secrets: API keys, Matrix tokens
```

Key harness.yaml sections: `providers`, `adapters`, `scheduler`, `sync`, `analytics`, `compact`.

`compact` config (auto-compacts large files via devstral):
```yaml
compact:
  provider: lmstudio
  model: mistralai/devstral-small-2-2512
  threshold_chars: 2048
```

## Scheduling

- **`check`** (~5 min): polls adapters + cron schedules → enqueues jobs. No LLM calls.
- **`work`** (~1 min): drains queue run-to-empty with threads. Respects `max_concurrency`.
- State in SQLite (`harness.db`): sync tokens, schedule next-fire, idle counts, job queue, provider slots.

## Species (6 total)

| Species | Entry Points | Notable tools |
|---------|-------------|---------------|
| `draum` | on_message, heartbeat | base only |
| `subconscious_dreamer` | on_message, heartbeat | base only, YAML pipelines |
| `neural_dreamer` | on_message, heartbeat | graph, activation_map, neural, organize, creative |
| `hecate` | on_message, heartbeat | graph, activation_map, multi-voice deliberation |
| `thrivemind` | on_message, heartbeat | graph, activation_map, colony deliberation |
| `consilium` | on_message, heartbeat | graph, activation_map, 5-persona pipeline |

Full per-species config reference: `/invoke species-ref`

## Finding Things Quickly

**To find species behavior:** `library/species/<species_id>/__init__.py` + `about.md`

**To find tool definitions:** `library/tools/tools.py` (schemas + dispatch), then module in `library/tools/`

**To find harness flow:** `checker.py` → `jobqueue.py` → `worker.py` → `context.py`

**To find prompt patterns:** `library/tools/patterns.py`, `library/tools/prompts.py`

**To find test mocks:** any `tests/test_<species>.py` — all have `make_mock_ctx()` with `compact_file = MagicMock(return_value=None)`

**Config loading:** `library/harness/config.py` — `load_harness_config()` / `load_instance_config()`

## Key Files

| File | Purpose |
|------|---------|
| `library/harness/config.py` | YAML loading with `${VAR}` env resolution; `HarnessConfig`, `InstanceConfig`, `CompactConfig` |
| `library/harness/context.py` | `InstanceContext` — full species interface |
| `library/harness/compactor.py` | `Compactor` — devstral-based file compaction |
| `library/harness/store.py` | `StoreDB` / `NamespacedStore` (SQLite) |
| `library/harness/jobqueue.py` | `JobQueue` — enqueue/claim/complete with instance guards |
| `library/harness/checker.py` | `Checker` — poll + cron → enqueue |
| `library/harness/worker.py` | `Worker` — run-to-empty, builds `InstanceContext`, provider slots |
| `library/harness/providers/` | `AnthropicProvider`, `OpenAICompatProvider` |
| `library/harness/adapters/` | `MatrixAdapter`, `LocalFileAdapter` |
| `library/tools/tools.py` | `make_tools()` schemas + `handle_tool()` dispatch |
| `library/tools/patterns.py` | High-level patterns: `gut_response`, `run_session`, `thinking_session`, `run_organize_phase`, etc. |
| `library/tools/phases.py` | Phase-scoped tool filtering: `get_tools_for_phase(phase)` |
| `library/tools/prompts.py` | Prompt assembly: `read_memory()`, `format_events()`, `format_memory_context()`, etc. |
| `library/tools/pipeline.py` | YAML pipeline runner (`run_pipeline`) |
| `library/tools/organize.py` | Knowledge org tools (archive, topic CRUD) |
| `library/tools/graph.py` | `SemanticGraph` — persistent directed weighted graph |
| `library/tools/activation_map.py` | `ActivationMap` — 2D attention/affect heatmap |
| `library/tools/neural.py` | `FastNet` / `SlowNet`, checkpoint management |
| `library/tools/rendering.py` | Graph→HTML, Map→PNG/GIF |
| `library/tools/segments.py` | Segment registry, selection vectors, variable injection |
| `library/tools/deliberate.py` | `generate_with_identity()`, `deliberate()`, `recompose()` |
| `library/tools/identity.py` | `Identity` dataclass, `format_persona()` |
| `library/tools/hecate.py` | Hecate voice management toolkit |
| `library/tools/thrivemind.py` | Colony management, spawn, voting |
| `library/tools/consilium.py` | Five-persona pipeline stages |
| `library/tools/voting.py` | `borda_tally()`, `approval_tally()` |
| `library/sync.py` | Memory sync to data repo |
| `library/publish.py` | Artifact publish to `_published/` |
| `library/scheduling.py` | OS schedule file generation |
| `library/__main__.py` | CLI entry point |

## Available Skills

| Skill | Invoke | Contents |
|-------|--------|---------|
| agent-setup | `/agent-setup` | Creating instances + new species, harness wiring |
| ctx-api | `/ctx-api` | Full InstanceContext API + config format reference |
| species-ref | `/species-ref` | All 6 species: entry points, default files, config keys |
| tools-patterns | `/tools-patterns` | make_tools options, phase scopes, patterns.py, prompts.py |
