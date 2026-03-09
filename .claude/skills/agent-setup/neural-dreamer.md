# Neural Dreamer Species — Setup & Architecture

## Overview

Neural Dreamer extends the Subconscious Dreamer pattern with a dual neural network layer that controls prompt configuration and accumulates non-textual state. It adds:

- **Two neural nets** (fast + slow) that modulate LLM behavior via segment selection and variable injection
- **Semantic graph** — persistent directed weighted graph for relational/associative memory
- **Activation map** — 2D float grid for attentional/affective state
- **Review phase** — post-send self-evaluation that feeds structured signals to the fast net
- **Sleep phase** — session-end consolidation that feeds the slow net

Spec documents: `proposals/neural_nets/architecture-technical.md`, `proposals/neural_nets/tools-memory-representations.md`

## Processing Cycles

### Slow cycle (heartbeat — once per session)

| Phase | Input | Output | NN interaction |
|-------|-------|--------|----------------|
| Think | previous thinking, chat history, slow net state | free-form reflection → `thinking.md` | Reads slow net state |
| Sleep | thinking output, session reviews, both net states | coherence assessment, emotional characterisation, updated self-description | Writes → slow net update |

### Fast cycle (on_message — per incoming message)

| Phase | Input | Output | NN interaction |
|-------|-------|--------|----------------|
| Gut feeling | message, fast net state, state store | unfiltered associations | Reads fast net state |
| Suggestion | gut output, slow net state, segments | structured response candidates | Reads both net states |
| Reply | suggestions, fast net feedback, context | final sent message | — |
| Review | full cycle trace, reply, reaction | coherence/effort/surprise scores | Writes → fast net update |

## Neural Nets

### Fast net
- Architecture: 2–3 layers, moderate width, higher learning rate
- Updates: after every fast cycle review
- Encodes: session momentum, immediate affect, context sensitivity
- Controls: state segments, relational segments, meta/phase segments
- Variables: `tone_warmth`, `verbosity`, `risk_tolerance`, `self_disclosure`, `confidence`

### Slow net
- Architecture: 4–6 layers, lower learning rate, stronger regularisation
- Updates: during sleep phase only
- Encodes: accumulated disposition, stable preferences, characteristic tendencies
- Controls: identity segments, temporal segments, baseline task framing
- Variables: `identity_salience`, `temporal_weight`, `relational_depth`, `reflection_depth`

Both stored as PyTorch checkpoints via `ctx.read_binary`/`write_binary`. Rolling checkpoint history (minimum 10 sessions).

## Segment System

Pre-written prompt segments in `segments/registry.yaml`, organised by category:

- **Identity** — who the instance is (multiple variants)
- **State** — current internal condition (populated from sleep output)
- **Relational** — chat environment and participants
- **Task** — constraints and objectives
- **Temporal** — history vs present vs future orientation
- **Meta** — phase-specific instructions

NN outputs a selection vector determining which segments are active and their order.

## Memory Tools

### Semantic graph
Tools: `graph_add_node`, `graph_add_edge`, `graph_remove_node`, `graph_remove_edge`, `graph_query`, `graph_describe`, `graph_snapshot`

Stored as JSON. Encoded via GNN/positional encoding → slow net input.

### Activation map
Tools: `map_define`, `map_set`, `map_set_region`, `map_get`, `map_describe`, `map_clear`, `map_snapshot`

Stored as JSON (16x16 to 32x32 grid). Encoded via CNN/flatten → fast net input.

### Cross-tool
Tools: `representation_summary`, `representation_compare`

## LLM → NN Signals

### Fast cycle review → fast net

| Signal | Type | Description |
|--------|------|-------------|
| `success` | float 0–1 | Task/interaction completion |
| `coherence` | float 0–1 | Agreement between phases |
| `effort` | float 0–1 | Normalised cost |
| `surprise` | float -1–1 | Outcome vs prediction |
| `unresolved` | float 0–1 | Deferred processing |
| `external_valence` | float -1–1 | External reaction tone |
| `novelty` | float 0–1 | Contrast with recent history |

### Sleep phase → slow net

| Signal | Type | Description |
|--------|------|-------------|
| `session_coherence` | float 0–1 | Cross-session consistency |
| `identity_drift` | float -1–1 | Self-description divergence |
| `accumulated_effort` | float 0–1 | Session-level cost |
| `emotional_characterisation` | vector | Embedding of retrospective state |
| `intention_alignment` | float 0–1 | Behavior vs intention match |
| `consolidation_items` | vector | Embedding of retention flags |

## Instance Config

```yaml
species: neural_dreamer
provider: anthropic
model: claude-sonnet-4-6

neural_dreamer:
  fast_net:
    layers: 3
    hidden_units: 32
    learning_rate: 0.01
  slow_net:
    layers: 5
    hidden_units: 64
    learning_rate: 0.001
  graph: true                  # Enable semantic graph tools
  activation_map: true         # Enable activation map tools
  segment_registry: segments/registry.yaml

messaging:
  # ... standard messaging config
```

## File Structure

```
library/species/neural_dreamer/
  __init__.py           # Species class + handlers
  heartbeat.yaml        # Slow cycle pipeline (think + sleep)
  on_message.yaml       # Fast cycle pipeline (gut + suggest + reply + review)
  segments/
    registry.yaml       # Segment definitions by category
    identity/           # Identity segment variants (.md files)
    state/              # State segment variants
    relational/         # Relational segment variants
    task/               # Task segment variants
    temporal/           # Temporal segment variants
    meta/               # Phase-specific instruction variants
  prompts/
    think.md            # Thinking phase prompt
    sleep.md            # Sleep/consolidation phase prompt
    gut.md              # Gut feeling phase prompt
    suggest.md          # Suggestion phase prompt
    reply.md            # Reply composition prompt
    review.md           # Post-send review prompt
```

## Toolkit Dependencies

All in `library/tools/`:

| Module | Purpose | Used by |
|--------|---------|---------|
| `segments.py` | Segment registry + selection + variable injection | neural_dreamer handlers |
| `graph.py` | Semantic graph tools + JSON storage | Any species via `make_tools` |
| `activation_map.py` | Activation map tools + JSON storage | Any species via `make_tools` |
| `neural.py` | FastNet/SlowNet classes, checkpoint management | neural_dreamer handlers |
| `rendering.py` | Graph→HTML, Map→PNG/GIF | Post-session hooks |

## Implementation Order

Build incrementally — each step is independently testable:

1. `ctx.read_binary`/`write_binary` (harness change)
2. `toolkit/segments.py` with manual weights
3. `toolkit/graph.py` with tool registration
4. `toolkit/activation_map.py` with tool registration
5. `neural_dreamer` species with pipelines, manual segment selection
6. `toolkit/neural.py` with PyTorch
7. Wire NN → segments + variables
8. Add review phase + LLM → NN signals
9. `toolkit/rendering.py`
10. NN encoding of graph/map tensors
