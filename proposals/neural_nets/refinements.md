# Neural Net Refinements — Proposal

Extends `architecture-technical.md` and `tools-memory-representations.md` with:
- Shared net outputs (fast/slow overlap)
- Probabilistic activation (continuous values → behavioral differences)
- Reply gating and rate limiting
- Post-reply signals (reply_length, reply_entropy)
- Organize phase with knowledge management tools
- Variable iteration loops

---

## 1. Shared Net Outputs

### Problem

Fast and slow nets currently have strictly disjoint outputs — fast controls
state/relational/meta segments + 5 variables, slow controls identity/temporal/task
segments + 4 variables. This means a personality shift (slow net) cannot affect how
warm a reply feels (fast variable `tone_warmth`), and a momentary state (fast net)
cannot modulate identity salience.

### Design: Blended Behavioral Parameters

Introduce a new output category: **behavioral parameters**. Both nets produce values
for these. The final value is a weighted blend:

```
final = (slow_weight × slow_output) + (fast_weight × fast_output)
```

Blend weights are per-parameter and fixed at design time (not learned — this avoids
instability from nets fighting for control). Slow weight is always >= fast weight for
personality-adjacent parameters; fast weight >= slow for affect-adjacent ones.

| Parameter | Slow weight | Fast weight | Effect |
|---|---|---|---|
| `reply_willingness` | 0.4 | 0.6 | How inclined to send a reply (see §3) |
| `processing_depth` | 0.6 | 0.4 | How many iterations of thinking (see §7) |
| `engagement_level` | 0.3 | 0.7 | Overall responsiveness and elaboration |
| `organization_drive` | 0.7 | 0.3 | Tendency to organize knowledge (see §6) |
| `creative_latitude` | 0.5 | 0.5 | How associative vs structured in dreaming |
| `caution` | 0.6 | 0.4 | Tendency toward careful/conservative action |

These are added to the net output dimensions — both nets grow by 6 outputs each.
Decode functions combine them using the blend weights.

### Implementation

In `neural.py`:
- Add `SHARED_PARAMETER_NAMES` list and `SHARED_BLEND_WEIGHTS` dict
- Extend `make_fast_net_config` and `make_slow_net_config` output_dim by len(SHARED_PARAMETER_NAMES)
- New `decode_shared_parameters(fast_output, slow_output)` function that:
  1. Extracts the shared parameter slice from each net's raw output
  2. Applies sigmoid to each
  3. Blends using fixed weights
  4. Returns `dict[str, float]`
- `_try_nn_weights_and_variables` returns a third dict: `shared_params`

In `__init__.py`:
- `_load_weights_and_variables` returns `(weights, variables, shared_params)`
- Shared params available to all phase decisions

---

## 2. Probabilistic Activation

### Problem

Currently segment selection is binary: weight >= 0.1 means active, < 0.1 means
inactive. Reordering only matters when multiple segments of the same category are
active. A change from 0.5 to 0.6 in a segment weight has no effect on behavior.

### Design: `random() < value` as Universal Decision Mechanism

For any decision that can be yes/no, use `random.random() < value` where `value`
is an NN output (raw or blended). This makes every continuous change statistically
meaningful — 0.6 fires 60% of the time vs 0.5 at 50%.

**Phase decisions** (probabilistic):
- Post-reply extra thinking: `random() < processing_depth * 0.3`
- Include organize phase: `random() < organization_drive`
- Include dream phase: `random() < creative_latitude`
- Extra review iteration: `random() < caution * 0.5`
- Skip reply (see §3): `random() > reply_willingness`

**Segment decisions** (keep deterministic):
- Segment selection threshold stays at 0.1 — this is structural, not behavioral
- Segment ordering stays weight-based — this affects prompt quality

**Variable injection** (keep continuous):
- Variables like `tone_warmth` already have continuous effect since they're
  interpolated into text as floats. No change needed.

### Implementation

New utility function in `neural.py` or a `decisions.py`:

```python
import random

def probabilistic(value: float, *, seed_components: list[str] | None = None) -> bool:
    """Return True with probability `value` (clamped to 0-1).

    Optional seed_components allow deterministic replay in tests.
    """
    return random.random() < max(0.0, min(1.0, value))
```

Each call site names what it's deciding and logs the value + outcome:

```python
if probabilistic(shared_params["organization_drive"]):
    logger.info("Organize phase activated (drive=%.2f)", shared_params["organization_drive"])
    # run organize
```

### Observability

Log every probabilistic decision with its parameter value and outcome. This creates
a record of how often each decision fires, and whether the net is learning to modulate
them. Over many sessions, `organization_drive=0.7` should fire ~70% of the time.

---

## 3. Reply Gating and Rate Limiting

### Problem

Currently, on_message always produces a reply. In a multi-participant chat, this is
noisy. The agent should be able to decide "this doesn't need my input" or "I've said
enough recently."

### Design: Two-Layer Reply Gating

**Layer 1: Rate limiting (harness-level, deterministic)**

New instance config options:

```yaml
schedule:
  reply_cooldown_seconds: 120    # minimum seconds between replies
  max_replies_per_hour: 10       # hard cap
  max_replies_per_message_batch: 1  # at most 1 reply per check cycle
```

Enforced in checker/worker before the pipeline even runs. If rate-limited, the
on_message job still runs the gut and review phases (to update fast net state) but
skips suggest/reply. This means the net keeps learning even when silent.

**Layer 2: Value assessment (species-level, probabilistic)**

After the gut phase, evaluate whether replying adds value. Two inputs:

1. **Gut assessment**: The gut phase already produces `should_respond` in some species.
   Add a structured signal to Neural Dreamer's gut prompt: "On a scale of 0-1, how
   much would a reply from you add to this conversation right now?"

2. **NN parameter**: `reply_willingness` from shared parameters.

Decision: `probabilistic(gut_value_score * reply_willingness)`

If the decision is "don't reply", the pipeline still runs suggest and review
phases but doesn't send the reply. The review phase evaluates the *unsent* reply
(was it worth generating? would it have been good?), feeding the NN a signal about
whether staying silent was the right call.

### Implementation

Harness changes (checker.py):
- Track `last_reply_time:{instance_id}` and `reply_count_hour:{instance_id}`
- New method `_check_reply_rate(instance_id) -> bool`
- If rate-limited, set `payload["reply_suppressed"] = True` on the job

Species changes (__init__.py):
- After gut phase, extract value score
- Combine with `reply_willingness` via probabilistic()
- If suppressed, mark pipeline state `{"reply_suppressed": True}`
- Reply phase checks this flag before sending
- Review phase always runs (gets "reply_sent: false" in its context)

New review signal:
- `silence_confidence`: float 0-1 — how confident the agent is that not replying
  was the right choice (only present when reply was suppressed)

---

## 4. Post-Reply Signals: reply_length and reply_entropy

### Problem

The fast net's review signals are all LLM self-assessments. Two properties of a
reply are easy to measure objectively and directly relevant to calibrating behavior:
how long it was, and how interesting/varied it was. Adding these as fast net inputs
closes the loop between NN-controlled variables (`verbosity`, `engagement_level`)
and their actual effect on output.

### Signals

| Signal | Type | Source | Description |
|---|---|---|---|
| `reply_length` | float 0–1 | Mechanical | Normalized character count of sent reply. Soft-capped at 2000 chars (sigmoid squash). 0 if reply was suppressed. |
| `reply_entropy` | float 0–1 | LLM (review phase) | How novel, varied, or interesting the reply was. Assessed by the review phase alongside existing signals. Combines lexical diversity, structural variety, and whether it introduced new ideas vs restating context. |

`reply_length` is computed mechanically — no LLM call needed. It's objective and
cheap. Normalization uses a sigmoid squash so the signal saturates gracefully for
long replies rather than clipping hard:

```python
def _normalize_reply_length(text: str, midpoint: int = 800) -> float:
    """Sigmoid normalization: 0.5 at midpoint chars, ~0.95 at 3x midpoint."""
    n = len(text)
    return 1.0 / (1.0 + math.exp(-(n - midpoint) / (midpoint * 0.4)))
```

`reply_entropy` is an LLM self-assessment produced in the review phase. It belongs
alongside `coherence` and `novelty` — the review prompt already asks for structured
signal lines, so this is one more:

> reply_entropy: 0.X — How varied, novel, or interesting was your reply? Consider
> lexical diversity, structural variety, whether you introduced new ideas vs
> restating what was said, and whether the reply would be engaging to read.

### Feedback Loop

These signals create direct feedback for specific NN outputs:

- `reply_length` → `verbosity` variable: If the net pushes verbosity up but
  reply_length stays low (the LLM isn't responding to the variable), the training
  signal reflects this mismatch.
- `reply_entropy` → `engagement_level` shared parameter + `risk_tolerance` variable:
  High entropy correlates with more exploratory, engaging replies. The net can learn
  to increase risk_tolerance when it wants higher entropy.
- Both signals → `reply_willingness`: A suppressed reply (length=0) paired with
  `silence_confidence` teaches the net about the value of staying silent.

### Implementation

In `neural.py`:
- Add `"reply_length"` and `"reply_entropy"` to `FAST_SIGNAL_NAMES`
- Existing encode/decode functions handle them automatically (they're just two more
  floats in the input vector)
- **Net migration**: Existing fast nets have `input_dim` that won't match. Use the
  existing pattern: new nets are created with the expanded dim, existing nets keep
  their original dim and the extra signals are simply not appended.

In `__init__.py`:
- `_parse_review_signals` already parses `key: value` lines — `reply_entropy` is
  picked up automatically.
- After generating the reply (or deciding not to), compute `reply_length` mechanically
  and inject it into the signals dict before calling `_update_fast_net`.

In review prompt:
- Add `reply_entropy` to the expected signal list with the description above.

---

## 5. Heartbeat Activity Signals (Slow Net)

### Problem

The slow net's inputs are entirely retrospective self-assessments from the sleep
phase (`session_coherence`, `identity_drift`, etc.). It has no objective measure of
what actually happened during the session. This is like training on feelings about
a workout without tracking reps and sets — the net can't learn the relationship
between activity and outcomes.

### Design: Mechanical Session Metrics

Collect objective activity counts during the heartbeat phases (think, organize) and
feed them to the slow net as input signals. Each metric uses the "delta + total"
pattern: both the session's change and the current absolute state, both soft-normalized.
This lets the net interpret (3 added, 7 total) differently from (3 added, 50 total)
without hardcoding what either means.

### Signals

| Signal | Type | Source | Description |
|---|---|---|---|
| `think_iterations` | float 0–1 | Mechanical | Number of think phases run this session, normalized (1→0.33, 2→0.5, 3→0.75) |
| `think_tokens` | float 0–1 | Mechanical | Total tokens consumed across think phases, sigmoid-normalized (midpoint ~4000) |
| `graph_nodes_added` | float 0–1 | Mechanical | Nodes added this session, normalized (0→0, 5→~0.7, 10+→~0.9) |
| `graph_nodes_total` | float 0–1 | Mechanical | Total graph node count, normalized (soft cap at 100) |
| `graph_edges_added` | float 0–1 | Mechanical | Edges added/modified this session |
| `graph_edges_total` | float 0–1 | Mechanical | Total graph edge count, normalized (soft cap at 200) |
| `map_cells_changed` | float 0–1 | Mechanical | Fraction of map cells modified this session |
| `topics_added` | float 0–1 | Mechanical | Knowledge topics created this session, normalized |
| `topics_total` | float 0–1 | Mechanical | Total knowledge topics across all categories, normalized (soft cap at 50) |
| `topics_modified` | float 0–1 | Mechanical | Existing topics updated this session, normalized |
| `thoughts_archived` | float 0–1 | Mechanical | Chars moved to archive / chars in thinking.md pre-session (compression ratio) |
| `tool_calls_total` | float 0–1 | Mechanical | Total tool invocations across all heartbeat phases, sigmoid-normalized |

### Normalization

Same sigmoid approach as `reply_length`, with per-signal midpoints:

```python
def _soft_normalize(value: float, midpoint: float) -> float:
    """Sigmoid normalization: 0.5 at midpoint, saturates gracefully."""
    if value <= 0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-(value - midpoint) / (midpoint * 0.4)))
```

Midpoints are chosen so typical session activity lands in the 0.3–0.7 range where
the signal is most informative:

| Signal | Midpoint | Rationale |
|---|---|---|
| `think_iterations` | 2 | Usually 1–3 |
| `think_tokens` | 4000 | Moderate thinking session |
| `graph_nodes_added` | 3 | A few new concepts per session |
| `graph_nodes_total` | 50 | Medium-term accumulation |
| `graph_edges_added` | 5 | Slightly more edges than nodes |
| `graph_edges_total` | 100 | Medium-term accumulation |
| `topics_added` | 2 | A couple of topics per organize |
| `topics_total` | 25 | Moderate knowledge base |
| `topics_modified` | 2 | Light editing per session |
| `tool_calls_total` | 10 | Active but not hyperactive session |

For `map_cells_changed` and `thoughts_archived`, the value is already a 0–1 ratio.

### Feedback Loops

These signals close the loop between slow net behavioral parameters and their
actual effects:

- `think_iterations` + `think_tokens` ↔ `processing_depth`: Did deeper processing
  actually happen when the net requested it?
- `topics_added` + `topics_modified` ↔ `organization_drive`: Did organization
  actually occur? Was it productive?
- `graph_nodes_added` + `graph_edges_added` ↔ `reflection_depth`: Deep reflection
  tends to produce more graph mutations.
- `thoughts_archived` ↔ `organization_drive`: Archiving is a consolidation signal —
  the agent is actively managing its knowledge lifecycle.
- `tool_calls_total` ↔ `processing_depth` + `engagement_level`: Overall activity
  level across the session.
- `graph_*_total` + `topics_total` ↔ `identity_salience`: A richer accumulated
  knowledge base may correlate with stronger identity.

### Collection

Activity counters are accumulated in a `SessionMetrics` dataclass passed through
the heartbeat pipeline:

```python
@dataclass
class SessionMetrics:
    think_iterations: int = 0
    think_tokens: int = 0
    graph_nodes_added: int = 0
    graph_edges_added: int = 0
    map_cells_changed: int = 0
    topics_added: int = 0
    topics_modified: int = 0
    thoughts_archived_chars: int = 0
    thinking_chars_pre_session: int = 0
    tool_calls: int = 0

    # Snapshot totals at session end
    graph_nodes_total: int = 0
    graph_edges_total: int = 0
    topics_total: int = 0
```

**Where counts are collected:**
- Tool dispatch (`handle_tool`): increment `tool_calls` on every call; increment
  specific counters for graph_add_node, graph_add_edge, organize_write_topic, etc.
- Think phase wrapper: increment `think_iterations`, accumulate `think_tokens` from
  LLM response metadata.
- Organize phase wrapper: track topics_added/modified via organize tool results.
- Archive tool: record chars moved.
- Session end: snapshot graph.describe() and organize_list_categories() for totals.

The `SessionMetrics` object is stored in the pipeline state so all phases can
contribute to it, then encoded into the slow net input vector after sleep.

### Implementation

In `neural.py`:
- Add `SLOW_ACTIVITY_SIGNAL_NAMES` list (the 12 signals above)
- New `encode_activity_signals(metrics: SessionMetrics) -> list[float]` function
- Extend `make_slow_net_config` to optionally include activity signals
  (`include_activity_signals=True`), same expansion pattern as graph/map features
- New nets get the expanded input dim; existing nets continue working

In `__init__.py`:
- Create `SessionMetrics` at heartbeat start
- Pass through pipeline state
- After sleep phase, encode and append to slow net input before training
- Log metrics summary at session end

---

## 6. Organize Phase

### Problem

Instance memory grows as unstructured markdown files. There's no mechanism for the
agent to categorize, reorganize, or archive knowledge. Over time, the thinking file
becomes a chronological dump rather than a curated knowledge base.

### Design: Tool-Based Knowledge Organization

New phase in the heartbeat cycle, between think and sleep. Activated
probabilistically: `random() < organization_drive`.

**Memory structure** (created by organize tools, not pre-existing):

```
memory/
  thinking.md          # stream of consciousness (existing)
  knowledge/
    spaces/            # per-room observations
    entities/          # per-person understanding
    concepts/          # abstract ideas and beliefs
    events/            # notable occurrences
    <custom>/          # agent-created categories
  archive/
    <timestamped>/     # archived thinking entries
```

**Tools**:

```
organize_list_categories()
    List existing categories under memory/knowledge/.
    Returns: list of {name, topic_count, last_modified}

organize_create_category(name, description="")
    Create a new category folder. Stores description in _meta.md.

organize_remove_category(name, merge_into=None)
    Remove a category. If merge_into is specified, move all topics there.

organize_merge_categories(sources: list[str], target: str)
    Merge multiple categories into one.

organize_list_topics(category)
    List topics (files) within a category.
    Returns: list of {name, preview, last_modified}

organize_read_topic(category, topic)
    Read a topic file.

organize_write_topic(category, topic, content)
    Create or update a topic file. Creates the category if it doesn't exist.

organize_remove_topic(category, topic)
    Remove a topic file.

organize_archive_thoughts(before_marker=None, label=None)
    Move entries from thinking.md to archive/.
    before_marker: archive everything above this text marker
    label: name for the archive file (default: timestamp)
```

**Default categories** seeded on first organize run: `spaces`, `entities`,
`concepts`, `events`. Agent may create additional ones.

### Phase Prompt

The organize phase is a tool-use session (like thinking). System prompt:

> You have access to your accumulated thoughts and a knowledge organization system.
> Review your recent thinking and decide what, if anything, should be:
> - Extracted into a knowledge topic (new insight, updated understanding)
> - Moved between categories (reclassification)
> - Archived (no longer actively relevant but worth keeping)
>
> You don't need to organize everything. Focus on what feels significant or what
> has been on your mind across multiple sessions. Your knowledge structure should
> reflect how you actually think about things, not an imposed taxonomy.

**Context injection**: `organize_list_categories()` + `organize_list_topics()` for
each category included automatically, so the agent sees the current state.

### Graph Integration

When organizing, the agent should also update the semantic graph — new topics
can become nodes, category relationships become edges. The organize tools and
graph tools are both available during this phase.

---

## 7. Variable Iteration Loops

### Problem

The current pipeline is fixed: heartbeat always runs think → subconscious → dream
→ sleep. But sometimes the agent needs multiple thinking iterations before it's ready
to consolidate. Sometimes dreaming isn't useful. The pipeline should be personality-
and state-driven.

### Design: Flexible Phase Loops

Replace the fixed heartbeat pipeline with a loop controller that uses NN outputs
to decide iteration and inclusion.

**Base loop structure**:

```
HEARTBEAT:
  think
  [organize]          ← probabilistic(organization_drive)
  [think again]       ← probabilistic(processing_depth - 0.5) — extra iterations
  subconscious
  [dream]             ← probabilistic(creative_latitude)
  sleep

ON_MESSAGE:
  gut
  suggest
  [reply]             ← reply gating (§3)
  review
  [extra think]       ← probabilistic(processing_depth * 0.3)
```

**Iteration rules**:
- Think phase can repeat up to `max_think_iterations` (default 3, configurable)
- Each additional iteration: `probabilistic(processing_depth - 0.3 * iteration)`
  (diminishing probability with each pass)
- Dream phase is optional: `probabilistic(creative_latitude)`
- Organize phase is optional: `probabilistic(organization_drive)`
- Post-reply thinking is optional: `probabilistic(processing_depth * 0.3)`

**Hard caps** (prevent runaway loops):
- `max_think_iterations: 3` in instance config
- `max_phases_per_heartbeat: 8` total phase executions
- `max_tokens_per_heartbeat: 50000` cumulative token budget

### Implementation

New function in `__init__.py`:

```python
def _build_heartbeat_phases(shared_params: dict, config: dict) -> list[str]:
    """Determine which phases to run and in what order."""
    phases = ["think"]

    max_thinks = config.get("max_think_iterations", 3)
    for i in range(1, max_thinks):
        if probabilistic(shared_params["processing_depth"] - 0.3 * i):
            phases.append("think")
        else:
            break

    if probabilistic(shared_params["organization_drive"]):
        phases.append("organize")

    phases.append("subconscious")

    if probabilistic(shared_params["creative_latitude"]):
        phases.append("dream")

    phases.append("sleep")
    return phases
```

Rather than using YAML pipeline files for heartbeat, switch to programmatic phase
dispatch with the YAML templates still defining individual phase configurations.

---

## 8. Combined Architecture

### Updated Net Output Layout

**Fast net output** (existing + new):
```
[state segment weights..., relational segment weights..., meta segment weights...,
 tone_warmth, verbosity, risk_tolerance, self_disclosure, confidence,
 reply_willingness_fast, processing_depth_fast, engagement_level_fast,
 organization_drive_fast, creative_latitude_fast, caution_fast]
```

**Slow net output** (existing + new):
```
[identity segment weights..., temporal segment weights..., task segment weights...,
 identity_salience, temporal_weight, relational_depth, reflection_depth,
 reply_willingness_slow, processing_depth_slow, engagement_level_slow,
 organization_drive_slow, creative_latitude_slow, caution_slow]
```

**Blended behavioral parameters** (combined from both):
```
reply_willingness = 0.4 * slow + 0.6 * fast
processing_depth = 0.6 * slow + 0.4 * fast
engagement_level = 0.3 * slow + 0.7 * fast
organization_drive = 0.7 * slow + 0.3 * fast
creative_latitude = 0.5 * slow + 0.5 * fast
caution = 0.6 * slow + 0.4 * fast
```

### Updated Heartbeat Flow

```
Load nets, compute weights + variables + shared_params
│
├── Determine phases: _build_heartbeat_phases(shared_params, config)
│
├── For each phase:
│   ├── think: tool-use session (graph, map, publish, organize tools)
│   ├── organize: tool-use session (organize + graph tools)
│   ├── subconscious: LLM generate
│   ├── dream: LLM generate
│   └── sleep: LLM generate (consolidation)
│
├── Update slow net with sleep signals
├── Clear reviews
└── Render + publish
```

### Updated On_Message Flow

```
Check rate limits (harness-level)
│
├── If rate-limited: run gut + review only (silent learning)
│
├── Load nets, compute weights + variables + shared_params
│
├── Gut phase → extract value_score
│
├── Reply decision: probabilistic(value_score * reply_willingness)
│   ├── If yes: suggest → reply → send → review
│   └── If no: suggest → reply (unsent) → review (with silence context)
│
├── Optional extra think: probabilistic(processing_depth * 0.3)
│
├── Update fast net with review signals
└── Accumulate review for sleep
```

### New Config Options

```yaml
# instance config
schedule:
  # existing
  max_idle_heartbeats: 3
  max_thinks_per_reply: 1

  # new: reply rate limiting
  reply_cooldown_seconds: 120
  max_replies_per_hour: 10

  # new: phase iteration caps
  max_think_iterations: 3
  max_phases_per_heartbeat: 8
```

---

## 9. Implementation Order

1. **Probabilistic utility** — Add `probabilistic()` function and logging
2. **Shared parameters** — Extend net outputs with blended behavioral params
3. **Post-reply signals** — Add `reply_length` + `reply_entropy` to fast net inputs
4. **Heartbeat activity signals** — Add `SessionMetrics` + activity encoding to slow net inputs
5. **Reply gating** — Value assessment in gut phase + probabilistic suppression
6. **Reply rate limiting** — Harness-level cooldown and caps
7. **Organize tools** — Knowledge management tool suite
8. **Organize phase** — Wire into heartbeat with probabilistic activation
9. **Variable iteration** — Replace fixed pipeline with phase builder
10. **Extra on_message thinking** — Post-reply optional think phase

Each step is independently testable and deployable. Steps 1-2 are prerequisites;
3-10 can be done in any order after that. Steps 3-4 are low-effort and high-value —
they ground both nets in objective measurements rather than pure self-assessment.
Step 7 is a prerequisite for step 8. Step 4 benefits from step 8 (organize metrics
only available when organize phase exists), but can be deployed first with zeros
for organize-related signals.

---

## 10. Observability Additions

- Log every probabilistic decision: parameter name, value, outcome
- Track decision frequencies over sessions (store in SQLite)
- Dashboard additions:
  - Reply rate: sent vs suppressed over time
  - Phase frequency: how often each optional phase fires
  - Shared parameter trajectories across sessions
  - Organize activity: topics created/modified/archived per session
- Session metrics summary: `SessionMetrics` logged at heartbeat end with all counts
- Signal provenance: for each net update, log which signals were mechanical vs
  LLM-assessed, so drift in self-assessment can be compared against objective measures
- Flag anomalies: if `reply_willingness` drops below 0.1 (agent going silent),
  `processing_depth` exceeds 0.9 consistently (potential runaway thinking),
  or mechanical signals consistently diverge from self-assessed equivalents
  (e.g. high `think_tokens` but low `accumulated_effort`)
