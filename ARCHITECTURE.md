# Three-Layer LLM Agent Architecture

## Overview

This document describes the Symbiosis agent system as a three-layer architecture: **Harness**, **Species**, and **Instance**. The goal is to separate infrastructure from behavior from state — so that new agent types can be defined without touching core machinery, and multiple instances can run in parallel without interference.

The existing system already has this structure informally: launchd is the Harness, the Python scripts and prompts are the Species, and the `memory/` directory is the Instance. This document names and hardens those boundaries.

---

## Layer 1: Harness (Host)

The Harness is infrastructure. It knows nothing about agents, personas, or memory semantics.

**Responsibilities:**

- **Storage**: Namespaced key/value file storage. Reads and writes text files. Does not interpret contents.
- **Structured store**: Typed key/value store (backed by SQLite) for runtime variables, coordination data, and cross-instance state. Distinct from file storage — queryable, supports atomic operations.
- **Scheduling**: Runs registered entry points on a schedule. Accepts cron-style or interval expressions. Does not know what those entry points do.
- **LLM access**: Wraps one or more LLM providers (LMStudio, Anthropic, OpenAI). Accepts a model ID, messages, and a tool schema. Returns a normalized response. Provider and model are caller-specified; the Harness routes accordingly.
- **Tool dispatch**: When an LLM response includes a tool call, the Harness routes it to the registered handler and feeds the result back. Tool handlers are registered by the Species at load time.
- **Messaging abstraction**: Platform-agnostic send/receive over spaces. The Harness holds the platform adapter (Matrix, Slack, local files, etc.) and maps logical space identifiers to platform handles. Species code calls `ctx.send(space, message)` and receives events as `(space, sender, body)` — no platform-specific concepts leak through. Registering an adapter is optional; Species that communicate only via inter-instance mailboxes need none.

**What the Harness does NOT do:**

- It does not define prompts, personas, or memory structure.
- It does not know the difference between a heartbeat and a distillation run.
- It does not manage inter-instance relationships.
- It does not decide what to remember or what to say.
- It does not enforce any schema on stored files — that is the Species's concern.
- It does not know what Matrix is. Platform details are encapsulated in the adapter.

---

## Layer 2: Species (Flavor)

A Species is a class definition for agent behavior. It is stateless code that defines what an agent of this type does.

Species responsibilities divide into two levels: **core** (shared across all species) and **flavor** (implementation choices specific to this species).

### Core responsibilities

These are universal — every Species must address them, though how is up to the Species:

- **React to external triggers**: When a message or event arrives, decide whether and how to respond.
- **Run on a schedule**: Perform unsolicited activity at regular intervals (thinking, memory maintenance, etc.).
- **Maintain persistent memory**: Read from and write to the Instance's storage to accumulate state across sessions.
- **Declare a file schema**: Specify which files an Instance owns and their initial contents.
- **Support spawning**: Provide a way to create a new Instance with a blank slate.

### Flavor details

These are implementation choices — one Species does them this way; another might not do them at all:

- **Gut → plan → compose pipeline**: One strategy for producing a response. Another species might respond directly.
- **Relationship tracking**: Saving structured notes about entities encountered. A task-oriented species might not care about relationships.
- **Subconscious assessment**: A meta-evaluation pass after each session. Optional.
- **Memory distillation**: Recursive merging of memory files to manage context size. One approach; others exist.
- **React phase / intentions**: Translating the subconscious assessment into forward-looking notes. Flavor-specific.

The Harness doesn't know or care which of these a Species implements. It only cares about the manifest interface.

### Harness–Species contract

The Species exports a manifest that the Harness reads at load time. This is the minimal surface the Harness requires:

```
{
  species_id: string,
  entry_points: [{ name, schedule, handler }],
  tools: [{ name, schema, handler }],
  default_files: { path: initial_content },
  spawn: fn(instance_id) -> void
}
```

The manifest says nothing about memory structure, relationship tracking, or response strategy — those are flavor details internal to the Species's handlers. The Harness calls each entry point handler with a single argument: an **instance context** that scopes all storage and LLM calls to that instance.

---

## Layer 3: Instance

An Instance is the state of one running entity. It is not code — it is data.

**What it is:**

- A unique ID (e.g. `draum-1`)
- A namespace in the Harness storage
- All configuration and memory files within that namespace
- A record of which Species it instantiates

**State visibility:**

Instance state falls into three categories from the perspective of the running consciousness:

- **Infrastructure (config)** — Platform handles, credentials, space mappings. Always read-only: the instance never writes its own config. Whether config is *visible* to the instance depends on the deployment — a single persistent agent may read its own entity identity (e.g., to populate a self-relationship file); a hivemind worker may treat config as fully invisible. In either case, config is not injected into LLM context by default; the Species must explicitly request it.
- **Immutable** — State written by parallel or post-session processes. The main session cannot modify it. `subconscious.md` is the canonical example: produced by a separate evaluation pass, read-only from the consciousness's own perspective. The consciousness observes its subconscious; it does not write it.
- **Mutable** — State the instance can explicitly read and write during a session. `thinking.md` is the canonical example: accumulated, revised, and pruned through explicit tool calls.

This distinction matters because it reflects how a real cognitive architecture works: some state is infrastructure, some is introspective signal from a process that runs separately, and some is the working surface of active thought.

**Messaging identity:**

An Instance's config declares its messaging identity and space memberships. The Harness reads this config and binds it to the platform adapter — neither the Species code nor the memory files contain any platform-specific identifiers. Species code refers to spaces by logical name (`ctx.send("main", message)`). Swapping an instance to a different platform means updating the config; the Species code is unchanged.

**Sandboxing:**

All Harness operations performed through an instance context are automatically scoped to that instance's namespace. An Instance cannot accidentally read or write another Instance's files through normal operation. Cross-instance access is explicit and harness-mediated.

---

## Spawn Lifecycle

1. Caller invokes `spawn(new_instance_id)`.
2. The Harness creates a new namespace for the instance.
3. Species default files are copied into it.
4. If the Species defines an `on_spawn` entry point, the Harness invokes it with the new instance context — allowing the Species to write initial identity or configuration.
5. The new instance is registered with the scheduler.

Spawning is the only way to create an instance. There is no other mutation of instance identity after creation.

---

## Scheduling and Dispatch

The Harness maintains a registry of instances, entry points, and their schedules. When a schedule fires — or a reactive trigger arrives — the Harness constructs an instance context scoped to that instance and calls the Species handler. The Species handler assembles context, calls the LLM, handles tool responses, and writes results back — all through the instance context, all sandboxed.

Two trigger modes are supported: **scheduled** (cron or interval) and **reactive** (incoming message events, delivered via the messaging adapter). Both are equivalent from the Species's perspective; the difference is a deployment concern.

---

## Inter-Instance Communication

The canonical primitive is **harness-mediated mailboxes**: one instance addresses a message to another instance by ID; the Harness delivers it to that instance's inbox; the receiving instance reads its inbox on its next scheduled run. No external broker is required.

For human-facing communication, the messaging adapter (Matrix, Slack, etc.) remains available as a Species-level tool. The two channels are independent: mailboxes for instance-to-instance coordination, the messaging adapter for external presence.

---

## Reference Configurations

The architecture is designed to accommodate varied deployments without changes to the Harness or Species interface. Two configurations illustrate the range:

---

### Configuration A: Persistent Memory Agent

> Anthropic/Claude + Matrix · Single instance · Spontaneous thinking sessions

| Layer | Choices |
| ----- | ------- |
| Harness | LLM: Anthropic API; Messaging adapter: Matrix |
| Species | Draum: gut→plan→compose, subconscious, react, relationships, memory distillation |
| Instance | Single instance; config holds Matrix identity and room memberships |

Config is accessible — the instance reads its own identity to populate its self-relationship file. Spontaneous sessions are scheduled heartbeats; reactive sessions fire when the Matrix adapter delivers an incoming message.

---

### Configuration B: Multi-Instance Hivemind

> Local hosted model + Local file messaging · Multiple instances · Peer coordination

| Layer | Choices |
| ----- | ------- |
| Harness | LLM: LMStudio (local); Messaging adapter: none |
| Species | Hivemind: task claiming, result sharing; no subconscious, no relationship tracking |
| Instance | N instances spawned from one seed; coordination via mailboxes and shared store |

Config is invisible — instances identify each other by instance ID only. All coordination happens through harness mailboxes and the structured store. Spawning is instance-initiated, subject to Harness-level authorization.

---

**What stays the same across both:**

- The Harness interface is unchanged — only the LLM provider and messaging adapter differ.
- Species manifests follow the same contract.
- Storage namespacing and sandboxing are identical.
- The Harness schedules and dispatches entry points in both cases.
