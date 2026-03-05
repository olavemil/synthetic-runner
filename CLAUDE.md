# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Status

This is the **Symbiosis toolkit** — a framework for building multi-layered LLM agent systems. The project is currently in the **design/planning phase**. No source code exists yet; the repo contains only architecture and implementation documents.

Key documents:
- `ARCHITECTURE.md` — Three-layer architecture specification with reference configs
- `IMPLEMENTATION.md` — Technical implementation roadmap with interfaces and patterns

## Architecture Overview

The system uses three layers with strict separation:

- **Harness (Layer 1)** — Infrastructure: storage, scheduling, LLM provider routing, tool dispatch, messaging adapters. Knows nothing about agent behavior or memory semantics.
- **Species (Layer 2)** — Stateless behavior code. Exports a manifest (`species_id`, `entry_points`, `tools`, `default_files`, `spawn`). Entry point handlers receive an `InstanceContext` as their only argument.
- **Instance (Layer 3)** — Pure data: a unique ID, a namespaced storage directory, and config files. Not code.

## Key Design Principles

**InstanceContext** is the core abstraction — Species code only interacts with infrastructure through `ctx`:
- `ctx.read/write/list` — namespaced file storage (auto-scoped to instance)
- `ctx.llm(messages, ...)` — provider-agnostic LLM calls returning normalized `LLMResponse`
- `ctx.send/poll(space, ...)` — messaging via logical space names (no platform details)
- `ctx.send_to/read_inbox` — harness-mediated inter-instance mailboxes
- `ctx.store/shared_store(namespace)` — SQLite-backed typed key/value store

**Species never** import vendor SDKs, construct absolute paths, or call HTTP directly.

## Config Structure (Target Layout)

```
config/
  harness.yaml          # providers, adapters, scheduler settings
  instances/
    <instance-id>.yaml  # per-instance: species, model, messaging identity
.env                    # secrets only (safe to omit from commits)
```

Config files use `${VAR}` references resolved from `.env` at load time.

## Pattern Library

Reusable patterns all take `(ctx, ...)` and operate via `ctx`:

| Pattern | Purpose |
|---------|---------|
| `gut_response` | Initial gut-check guidance from events |
| `plan_response` | Deliberate planning step |
| `compose_response` | Final message composition |
| `run_subconscious` | Post-session meta-evaluation (writes `subconscious.md`) |
| `run_react` | Translates subconscious into intentions (writes `intentions.md`) |
| `distill_memory` | Recursive memory compression |
| `update_relationships` | Structured entity relationship tracking |

## Suggested Build Order

Per `IMPLEMENTATION.md` section "Build Order":

1. Config system (everything depends on it)
2. Instance context
3. LLM provider abstraction (`OpenAICompatProvider`, `AnthropicProvider`)
4. Messaging adapter abstraction (`MatrixAdapter`, `LocalFileAdapter`)
5. Pattern library (formalize existing scripts as `ctx`-taking functions)
6. Scheduler (unified dispatch, replacing launchd)
7. YAML pipeline support (lowest priority)

## Package Structure (Proposed)

Single package `symbiosis/` with subpackages: `harness/`, `toolkit/`, `species/`.

## Open Implementation Questions

See `IMPLEMENTATION.md` § "Open Questions" for unresolved decisions around: multi-host secrets, Anthropic `tool_choice` differences, reactive polling interval, YAML pipeline expressiveness limits, and concurrency (lock-per-instance for simultaneous entry points).
