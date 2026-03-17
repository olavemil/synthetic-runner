---
name: species-ref
description: All 6 Symbiosis species — entry points, default files, make_tools options, config keys, and key patterns. Use when working with or extending any species, or choosing a species for a new instance.
user_invocable: true
---

# Species Reference

All species live in `library/species/<species_id>/`. Each is auto-discovered from `library.species.*` subpackages.
Source: `__init__.py` (manifest + handlers) + `about.md` (description).
Test file: `tests/test_<species_id>.py`.

---

## Quick Reference Table

| Species | Entry Points | make_tools options | Default files |
|---------|-------------|-------------------|--------------|
| `draum` | on_message, heartbeat | base only | thinking.md, project.md, sessions.md, scratchpad.md, sensitivity.md, intentions.md, subconscious.md |
| `subconscious_dreamer` | on_message, heartbeat | base only | thinking.md, dreams.md, concerns_and_ideas.md |
| `neural_dreamer` | on_message, heartbeat | graph, activation_map, neural, organize, creative | thinking.md, dreams.md, concerns_and_ideas.md, sleep.md, reviews.md, segment_weights.json |
| `hecate` | on_message, heartbeat | graph, activation_map | memory.md, constitution.md, per-voice files |
| `thrivemind` | on_message, heartbeat | graph, activation_map | constitution.md, sessions.md, colony.md, contributions.md |
| `consilium` | on_message, heartbeat | graph, activation_map | memory.md, constitution.md, per-persona files |

---

## Draum

**Pattern:** Gut → Plan → Compose response pipeline. Simple persistent memory agent.

**on_message:** `gut_response()` → `plan_response()` → `compose_response()` → send. Updates `relationships/<sender>.md`, `intentions.md`, `subconscious.md`.

**heartbeat:** `run_subconscious()` + `run_react()`. Reflects on recent sessions, updates intentions.

**Config keys:** None beyond standard.

**Test:** `tests/test_draum.py`

---

## Subconscious Dreamer

**Pattern:** Three-phase thinking (active → subconscious → dreaming) + three-phase response (intuition → worry → action). Driven entirely by **YAML pipelines**.

**on_message pipeline:** `species/subconscious_dreamer/on_message.yaml`
Steps: distill_messages → run_subconscious → thinking_session → run_organize_phase → run_react → compose_response

**heartbeat pipeline:** `species/subconscious_dreamer/heartbeat.yaml`
Steps: distill_memory → thinking_session → run_organize_phase → run_subconscious → llm_generate (dreams) → llm_generate (concerns_and_ideas)

**Config keys:** None beyond standard.

**Test:** `tests/test_subconscious_dreamer.py`

---

## Neural Dreamer

**Pattern:** Extends Subconscious Dreamer with dual neural networks controlling prompt segment selection + variable injection. Adds semantic graph and activation map as non-textual memory.

**Fast cycle (on_message):**
1. Gut feeling — reads fast net state, unfiltered associations
2. Suggestion — reads both net states + segment registry → structured candidates
3. Reply — sends message
4. Review — evaluates cycle → updates fast net with 7 signals

**Slow cycle (heartbeat):**
1. Think — reads slow net state → updates `thinking.md`
2. Organize — knowledge management phase
3. Subconscious + dream generation
4. Create phase — creative output
5. Sleep — consolidates to slow net with 6 signals

**make_tools options used:** `graph=True, activation_map=True, neural=True, organize=True, creative=True`

**Config keys:**
```yaml
neural_dreamer:
  fast_net:
    layers: 3
    hidden_units: 32
    learning_rate: 0.01
  slow_net:
    layers: 5
    hidden_units: 64
    learning_rate: 0.001
  graph: true
  activation_map: true
  segment_registry: segments/registry.yaml  # relative to species dir
```

**Default files:** `thinking.md`, `dreams.md`, `concerns_and_ideas.md`, `sleep.md`, `reviews.md`, `segment_weights.json`, nets stored as binary (`nets/fast.pt`, `nets/slow.pt`)

**Segment system:** `library/species/neural_dreamer/segments/` — categories: identity, state, relational, task, temporal, meta. NN outputs select active segments and order.

**Variables (injected into active segments):**
- Fast net controls: `tone_warmth`, `verbosity`, `risk_tolerance`, `self_disclosure`, `confidence`
- Slow net controls: `identity_salience`, `temporal_weight`, `relational_depth`, `reflection_depth`

**Detailed architecture:** See `.claude/skills/agent-setup/neural-dreamer.md`

**Test:** `tests/test_neural_dreamer.py` (39 tests)

---

## Hecate

**Pattern:** Three named voices (each with model + personality) that think, vote, and compose responses. Each voice has separate memory files.

**on_message:** Load voice contexts → each voice generates suggestion → Borda vote → winning voice composes reply → send.

**heartbeat:** Each voice does subconscious update + optional creative phase.

**Voice memory files** (per voice, e.g. `voice_name = "raven"`):
- `raven_thinking.md` — working thoughts
- `raven_subconscious.md` — background reflections
- `raven_motivation.md` — current drives

**Config keys:**
```yaml
hecate:
  voices:
    - name: raven
      model: claude-opus-4-6          # can differ per voice
      provider: anthropic             # optional override
      personality: "introspective, poetic, risk-tolerant"
    - name: ember
      model: claude-sonnet-4-6
      personality: "pragmatic, warm, direct"
    - name: vale
      model: claude-haiku-4-5-20251001
      personality: "analytical, cautious, precise"
  voice_space: main                   # which space to use for messaging
```

**make_tools options used:** `graph=True, activation_map=True`

**Toolkit:** `library/tools/hecate.py` — `HecateConfig`, `load_config()`, `run_voice_messaging_phase()`, `update_voice_subconscious()`

**Test:** `tests/test_hecate_species.py` (13), `tests/test_hecate_toolkit.py` (6)

---

## Thrivemind

**Pattern:** Colony of named individuals (adjective-adjective-noun names) with personality dimensions. Individuals propose candidates, vote via Borda, and the colony converges. Individuals spawn and die over time.

**on_message:** `distill_messages` → `run_subconscious` → colony generates candidates → Borda vote → compose final response → send.

**heartbeat (*/15 min):**
- Constitution review + update
- `run_spawn_cycle` — eligible parents produce offspring; colony trimmed to max_colony_size
- Graph/map organization phase

**Voting:** Borda tally across colony. Voters whose pick was in the top 2 get +1 approval; voters whose top pick lost get -1. This dual-pick system makes positive approval achievable.

**Colony files:** `colony.md` (roster + personalities), `reflections/<name>.md` per individual, `constitution.md`, `sessions.md`, `contributions.md`

**Config keys:**
```yaml
thrivemind:
  min_colony_size: 8
  max_colony_size: 16
  suggestion_fraction: 0.5        # fraction of colony that suggests each round
  approval_threshold: 3           # minimum approval score to survive
  consensus_threshold: 0.6        # fraction agreement needed for strong consensus
  suggestion_model: ""            # optional model override for suggestions
  writer_model: ""                # optional model override for final composition
  voice_space: main
```

**make_tools options used:** `graph=True, activation_map=True`

**Toolkit:** `library/tools/thrivemind.py` — `ThrivemindConfig`, `load_colony()`, `run_spawn_cycle()`, `vote_peer_approval()`, `deliberate()`, `recompose()`

**Test:** `tests/test_thrivemind_species.py` (15), `tests/test_thrivemind_toolkit.py` (25)

---

## Consilium

**Pattern:** Five named personas with distinct roles that run a structured 5→4→1 pipeline: draft → reduce → merge → transform → review.

**on_message:**
1. Each of 5 personas drafts independently
2. Reduction phase: 5 drafts → 4 (weakest removed by vote)
3. Merge phase: 4 remaining → 1 composite
4. Transform phase: composite refined
5. Review phase: final approval/edit

**heartbeat:** Subconscious update for each persona + graph/map organization.

**Persona memory files** (per persona, e.g. `name = "archivist"`):
- `archivist_thinking.md`
- `archivist_reviews.md`
- `archivist_subconscious.md`

**Config keys:**
```yaml
consilium:
  personas:
    - name: archivist
      model: claude-opus-4-6
      role: "keeper of historical context and precedent"
      personality: "meticulous, long-memory, conservative"
    - name: herald
      model: claude-sonnet-4-6
      role: "communicator and translator of complex ideas"
      personality: "clear, accessible, empathetic"
    - name: strategist
      model: claude-opus-4-6
      role: "planner and opportunity-seeker"
      personality: "forward-looking, risk-aware, decisive"
    - name: critic
      model: claude-sonnet-4-6
      role: "challenger and quality-enforcer"
      personality: "skeptical, rigorous, honest"
    - name: ghost
      model: claude-haiku-4-5-20251001
      role: "lateral thinker and wildcard"
      personality: "unexpected, creative, contrarian"
  voice_space: main
```

**make_tools options used:** `graph=True, activation_map=True`

**Toolkit:** `library/tools/consilium.py` — `ConsiliumConfig`, `load_config()`, `run_drafting_phase()`, `run_reduction_phase()`, `run_merge_phase()`, `run_transform_phase()`, `run_review_phase()`

**Test:** `tests/test_consilium_species.py` (12)

---

## Writing Test Mocks

All species tests use `make_mock_ctx()` — always include:

```python
def make_mock_ctx(files=None):
    ctx = MagicMock()
    _files = files or {}
    ctx.read = MagicMock(side_effect=lambda p: _files.get(p, ""))
    ctx.write = MagicMock(side_effect=lambda p, c: _files.update({p: c}))
    ctx.compact_file = MagicMock(return_value=None)  # required — prevents truthy mock compaction
    ctx.list_spaces = MagicMock(return_value=["main"])
    ctx.send = MagicMock(return_value="$event1")
    return ctx
```

`compact_file = MagicMock(return_value=None)` is required in any mock that exercises `write_file` or `append_thinking`, otherwise the MagicMock return value is truthy and compaction logic incorrectly activates.
