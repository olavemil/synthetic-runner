# Non-Textual Memory Tool Suite — Specification

## Overview

This document specifies a suite of tools providing the agent with non-textual
intermediate representations for memory and thought. These complement the existing
text-based state store with structured and spatial representations that:

- Are specifiable and readable by the agent via tool calls
- Feed into the neural network layer as tensors
- Render as visually inspectable artefacts
- Persist across sessions as part of accumulated state

Two primary representation types are provided: a **semantic graph** for relational and
associative memory, and an **activation map** for attentional and affective state. A
lightweight **publish hook** surfaces both as rendered visuals.

---

## Tool 1: Semantic Graph

### Concept

A persistent directed weighted graph. Nodes are concepts, entities, relationships, or
anything the agent finds worth representing. Edges encode typed relationships with
weights. The graph's topology — clustering, centrality, bridge nodes, isolated regions —
carries meaning not present in any individual node or edge.

The agent does not manage the graph as a data structure. It issues natural tool calls
that are translated into graph operations by the tool layer.

### Tools

```
graph_add_node(id, label, metadata={})
    Add or update a node.
    id: stable identifier (slug format)
    label: human-readable name
    metadata: optional dict — type, salience, created_at, notes

graph_add_edge(source_id, target_id, relation, weight=0.5, directed=True)
    Add or update an edge.
    relation: string describing the relationship type
    weight: float 0.0–1.0 encoding strength or confidence

graph_remove_node(id)
    Remove node and all connected edges.

graph_remove_edge(source_id, target_id, relation)
    Remove a specific edge.

graph_query(filter={})
    Return subgraph matching filter criteria.
    filter keys: node_ids, relation_types, min_weight, max_depth, central_node

graph_describe()
    Return summary statistics: node count, edge count, top nodes by centrality,
    isolated nodes, densest clusters. Designed for agent consumption during
    thinking phase — a quick read on the current shape of the graph.

graph_snapshot(label)
    Save a named snapshot of the current graph state.
    Used by sleep phase to mark session boundaries.
```

### Storage

Persisted as JSON (nodes + edges lists). Lightweight, human-readable, diff-friendly for
version history. Load full graph into memory at session start; write on any mutation.

```json
{
  "nodes": [
    {"id": "trust", "label": "Trust", "metadata": {"salience": 0.8}}
  ],
  "edges": [
    {"source": "trust", "target": "consistency", "relation": "requires",
     "weight": 0.9, "directed": true}
  ],
  "snapshots": [
    {"label": "session_042_end", "timestamp": "...", "node_count": 47}
  ]
}
```

### Neural Net Encoding

At the start of each session, the graph is encoded into a fixed-dimension vector via:

1. Compute node embeddings using a small GNN (Graph Neural Network) or simpler
   positional encoding if GNN is too heavy — degree centrality, clustering coefficient,
   average edge weight, component membership.
2. Aggregate into a global graph state vector (mean pooling + max pooling concatenated).
3. Pass as input to the slow net — graph structure changes slowly and reflects
   accumulated semantic state.

Relevant delta between current graph and last session's snapshot can be encoded
separately as a change signal fed to the fast net.

### Rendering

Rendered as a force-directed graph using D3.js or Pyvis.

- Node size: degree centrality
- Node colour: metadata type or salience
- Edge thickness: weight
- Edge label: relation type (shown on hover)
- Snapshot boundaries overlaid as visual markers

Output: static HTML file (self-contained, suitable for GitHub Pages).

---

## Tool 2: Activation Map

### Concept

A 2D grid of float values the agent authors to represent the current distribution of
attention, affect, or salience across a conceptual space it defines. The agent places
and adjusts values — the visual output is a heatmap or contour plot rendered from those
values.

The map is not a fixed schema. The agent defines what the axes mean, what regions
represent, and what intensity encodes. This definition is itself stored as metadata
and readable back. Different sessions may use different framings of the space.

### Tools

```
map_define(width, height, x_label, y_label, description)
    Initialise or redefine the map dimensions and axis semantics.
    width, height: grid resolution (suggest 16x16 to 32x32)
    x_label, y_label: what the axes represent in this framing
    description: free text — what this map is trying to capture

map_set(x, y, value)
    Set a single cell. value: float -1.0 to 1.0
    Negative values are valid — allows representation of opposition or aversion.

map_set_region(x, y, radius, value, falloff="linear")
    Set a circular region with optional falloff. More natural than cell-by-cell.
    falloff: "linear", "gaussian", "hard"

map_get()
    Return full grid as 2D array with current metadata.

map_describe()
    Return natural language summary: peak regions, negative regions, overall
    distribution shape. For agent consumption during thinking phase.

map_clear()
    Reset all values to 0.0. Metadata and axis definitions preserved.

map_snapshot(label)
    Save named snapshot. Used by sleep phase at session boundary.
```

### Storage

Persisted as JSON with grid array and metadata. Small footprint even at 32x32.

```json
{
  "width": 16,
  "height": 16,
  "x_label": "familiarity",
  "y_label": "affect",
  "description": "Mapping of participants and topics by how known and how felt",
  "grid": [[0.0, 0.3, ...], ...],
  "snapshots": [...]
}
```

### Neural Net Encoding

The grid is a natural tensor input. Options in order of complexity:

1. **Flatten and concatenate** with other inputs — simplest, loses spatial structure
2. **Small CNN** over the grid — preserves spatial relationships, modest overhead
3. **Summary statistics only** — mean, variance, peak location, skew — if full grid
   is too high-dimensional relative to net size

The map encodes current attentional/affective state, so it feeds the **fast net**
primarily. Snapshots of map trajectory over a session feed the sleep phase.

### Rendering

Rendered as a heatmap using matplotlib or a JS canvas element.

- Diverging colormap (negative = cool, positive = warm, zero = neutral)
- Contour lines at configurable intervals
- Axis labels and description as caption
- Snapshot series renderable as animation (session timeline)

Output: PNG per snapshot + animated GIF for session trajectory + static HTML embed.

---

## Tool 3: Representation Query (Cross-Tool)

A meta-tool allowing the agent to query across both representations simultaneously,
useful during thinking and sleep phases.

```
representation_summary()
    Returns combined description:
    - Graph: node count, top centrality nodes, recent additions, isolated nodes
    - Map: current distribution summary, axis framing, dominant regions
    - Delta since last snapshot for both
    Intended as a standard inclusion in the thinking phase context.

representation_compare(snapshot_a, snapshot_b)
    Diff two snapshots (same type) and return description of changes.
    Useful in sleep phase for characterising session-level drift.
```

---

## Hookup Architecture

### Session Lifecycle

```
Session start
  ├── Load graph from disk
  ├── Load activation map from disk
  ├── Encode graph → slow net input vector
  ├── Encode map → fast net input vector
  └── Both available to all phases via tool calls

Thinking phase
  ├── representation_summary() included in context
  ├── Agent may call graph and map tools freely
  └── Updates persist immediately

Fast cycles (per message)
  ├── Gut/suggest/reply: graph_query and map_get available (read-mostly)
  ├── Review phase: may update map to reflect cycle affect
  └── Map state after review → fast net update signal

Sleep phase
  ├── representation_summary() included
  ├── representation_compare() between session start and current state
  ├── Agent characterises changes as part of self-state summary
  ├── graph_snapshot("session_N_end")
  ├── map_snapshot("session_N_end")
  └── Snapshots → slow net consolidation signal

Session end
  ├── Write graph to disk
  ├── Write map to disk
  ├── Render graph HTML
  ├── Render map PNG + GIF
  └── Post-run hook → push renders to GitHub Pages
```

### Context Injection

Representation data enters the LLM context in two ways:

**Structural summary** (always included in thinking phase):
A few sentences from `representation_summary()` injected into the state segment of
context. The agent is aware of the current shape of its representations without needing
to explicitly query.

**On-demand detail** (tool calls):
During thinking and sleep phases, the agent can query specific subgraphs, regions,
or compare snapshots. This is not pre-injected — it is retrieved when the agent
decides it is relevant.

### Neural Net Wiring

```
Graph JSON
  └── GNN / positional encoder
        └── Global graph vector (dim ~32)
              └── Slow net input (stable, session-level)

Map grid
  └── CNN / flatten / statistics
        └── Map state vector (dim ~16–64)
              └── Fast net input (volatile, cycle-level)

Map delta (review phase)
  └── Fast net update signal

Graph delta (sleep phase)
  └── Slow net consolidation signal
```

### GitHub Pages Output

Post-run hook (existing infrastructure) receives:

- `graph.html` — interactive force-directed graph, self-contained
- `map_current.png` — current activation map
- `map_session.gif` — session trajectory animation
- `index.html` — simple dashboard embedding both with session metadata

The dashboard provides external observers a non-textual window into accumulated state.
Each session appends to a history, so drift over time is visually inspectable.

---

## Implementation Notes

**Start with stubs.** Implement all tools with correct signatures but no-op bodies.
Verify the agent calls them appropriately in context before building the encoding layer.

**Graph before map.** The graph is more immediately useful to the agent as a thinking
aid. The map is more immediately useful to the neural net. Build graph tooling first,
observe how the agent uses it, then build the map.

**Let semantics emerge.** Do not pre-define what the graph should contain or what the
map axes should represent. Provide the tools and observe what the agent chooses to
represent. The first few sessions of graph and map usage are diagnostically valuable
and should be logged in full.

**Encoding dimension sizing.** Match encoding output dimensions to net input layer
expectations. If nets are small (< 64 hidden units), use summary statistics rather
than full spatial encoding. Scale up encoding complexity only when net capacity
justifies it.

**Snapshot cadence.** Sleep phase snapshots are mandatory. Mid-session snapshots are
optional and agent-initiated. Do not auto-snapshot on every mutation — the graph and
map should represent a considered state, not a log of every tool call.
