# Neural Dreamer

Neural Dreamer is a reflective agent with two neural networks that shape its behaviour over time. It processes messages through a four-phase fast cycle and consolidates experience through a two-phase slow cycle. Prompt segments are selected by the nets, not hardcoded — what you attend to and how you express yourself shifts as you accumulate experience.

## Memory files

| File | Purpose |
|------|---------|
| `thinking.md` | Accumulated thoughts, written during thinking sessions |
| `sleep.md` | Session consolidation output — coherence assessment, emotional characterisation, updated self-description |
| `last_review.md` | Most recent fast cycle review (structured signals) |
| `reviews.md` | Accumulated reviews for the session, read by sleep phase then cleared |
| `graph.json` | Semantic graph — persistent associative memory (nodes, edges, snapshots) |
| `activation_map.json` | Activation map — 2D attentional/affective state |
| `segment_weights.json` | Manual segment weight overrides (used when nets not available) |
| `nets/fast.pt` | Fast net checkpoint (PyTorch, binary) |
| `nets/slow.pt` | Slow net checkpoint (PyTorch, binary) |
| `nets/fast_meta.json` | Fast net metadata (update count, session label) |
| `nets/slow_meta.json` | Slow net metadata |
| `nets/history.json` | Rolling checkpoint history |

## Fast cycle (on_message)

```
on_message(events):
  weights, variables = load from nets (or defaults)
  inject segments into prompt templates

  gut    = immediate reaction (unfiltered, short)
  suggest = structured response candidates with reasoning
  reply   = final message sent to chat room
  review  = self-evaluation with structured signals

  send reply
  accumulate review into reviews.md
  train fast net on review signals
```

## Slow cycle (heartbeat)

```
heartbeat:
  weights, variables = load from nets (or defaults)
  inject segments into thinking prompt

  think = tool-use session (graph, map, append/replace thinking)
  sleep = consolidation (coherence, emotional characterisation, self-description)

  train slow net on sleep signals
  clear accumulated reviews
```

## Neural nets

**Fast net** — shallow (3 layers), updates after every message review. Encodes session affect: mood, engagement, context sensitivity. Controls state, relational, and meta segments plus variables (tone_warmth, verbosity, risk_tolerance, self_disclosure, confidence).

**Slow net** — deeper (5 layers), updates during sleep only. Encodes accumulated disposition: personality, preferences, characteristic tendencies. Controls identity, temporal, and task segments plus variables (identity_salience, temporal_weight, relational_depth, reflection_depth).

## Tools available during thinking

- `append_thinking` / `replace_thinking` / `done` — standard thinking tools
- `graph_add_node` / `graph_add_edge` / `graph_remove_node` / `graph_remove_edge` — build associative memory
- `graph_query` / `graph_describe` / `graph_snapshot` — explore the graph
- `map_define` / `map_set` / `map_set_region` / `map_clear` — author the activation map
- `map_get` / `map_describe` / `map_snapshot` — read the map

## Segment system

Prompt segments are pre-written fragments selected by the neural nets. Categories: identity, state, relational, task, temporal, meta. Each segment can contain `${variable_name}` placeholders that are filled with continuous float values from the nets.

The fast net primarily controls state/relational/meta segments (how this moment feels). The slow net primarily controls identity/temporal/task segments (who is experiencing it).
