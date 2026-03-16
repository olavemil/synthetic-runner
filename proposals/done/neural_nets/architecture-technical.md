# Hybrid NN/LLM Agent — Technical Implementation Spec

## Current State

A working pipeline exists with the following components:

- LLM with variable prompt injection and segment selection
- Persistent state as produced text, written by one phase and read by another
- Two interleaved processing cycles (see below)
- Tools for reading and writing to an external chat room

This document extends that foundation with a dual neural network layer that feeds both
prompt configuration and phase control.

---

## Processing Cycles

Two cycles operate at different frequencies and serve different functions.

### Slow Cycle (once per session)

Operates on accumulated state. Not triggered by individual messages.

**Think phase**
- Input: previous thinking output, recent chat history, system feedback, slow net state
- Output: free-form text — intentions, observations, self-reflection
- Written to persistent state store, read by subsequent fast cycles

**Sleep phase**
- Input: thinking output, fast cycle reviews from the session, slow and fast net states
- Output: retrospective evaluation — coherence assessment, emotional characterisation
  of unnamed NN states, updated self-description
- Feeds back into both nets as the primary consolidation signal

The sleep phase is where the system attempts to name what its NN states were during the
session. This characterisation is retrospective and interpretive, not a real-time label.
Emotional vocabulary emerges from reflection rather than being imposed at design time.

### Fast Cycle (per incoming message)

**Gut feeling phase**
- Input: incoming message, current fast net state, relevant state store excerpts
- Output: immediate unfiltered response — associations, concerns, inclinations
- Short, not sent externally

**Suggestion phase**
- Input: gut feeling output, slow net state, context segments
- Output: structured response candidates with reasoning

**Reply phase**
- Input: suggestions, subconscious feedback from fast net, constructed context
- Output: final message sent to chat room

**Review phase**
- Input: full fast cycle trace, reply sent, any immediate external reaction
- Output: short evaluation — coherence, effort, surprise, unresolved items
- Fed into fast net immediately; accumulated for sleep phase

---

## Neural Network Layer

### Two-Net Architecture

**Fast net**
- Architecture: shallow (2–3 layers), moderate width, higher learning rate
- Timescale: updates after every fast cycle review
- Encodes: current mood analog, session momentum, immediate context sensitivity
- Analogous to: emotional state, present affect

**Slow net**
- Architecture: deeper (4–6 layers), lower learning rate, stronger regularisation
- Timescale: updates during sleep phase only
- Encodes: accumulated disposition, stable preferences, characteristic tendencies
- Analogous to: personality, long-term character

Both nets receive overlapping but not identical inputs. Their combined activation
constitutes the instance's current internal state. Neither net is authoritative alone.

### Technology

PyTorch. Save/load checkpoints after every cycle. Maintain rolling checkpoint history
(minimum 10 sessions). Wiping net state is not a neutral operation — treat it accordingly.

---

## Interface: NN → LLM (Prompt Configuration)

The nets control the LLM through two mechanisms applied jointly.

### Segment Selection (qualitative / structural)

A fixed registry of pre-written prompt segments, organised by category. The nets output
a selection vector determining which segments are active and their order (earlier = higher
influence on LLM processing).

Segment categories:
- **Identity** — characterisation of who the instance is; multiple variants available
- **State** — description of current internal condition; populated partly from sleep output
- **Relational** — context about the chat environment and participants
- **Task** — what is being done and under what constraints
- **Temporal** — weight given to history vs present vs forward orientation
- **Meta** — phase-specific instructions

**Fast net** primarily controls: state segments, relational segments, meta/phase segments
**Slow net** primarily controls: identity segments, temporal segments, baseline task framing

Rationale: fast net shapes how this moment feels; slow net shapes who is experiencing it.

### Variable Injection (continuous / fine-grained)

Named variables injected into active segments. Allow gradual modulation within a segment
without switching segments.

| Variable | Net source | Effect |
|---|---|---|
| `tone_warmth` | Fast | Warmer vs more neutral register |
| `verbosity` | Fast | Compression vs elaboration |
| `risk_tolerance` | Fast | Conservative vs exploratory |
| `self_disclosure` | Fast | How prominently internal state features |
| `confidence` | Fast | Certainty vs tentativeness in expression |
| `identity_salience` | Slow | How strongly established character is asserted |
| `temporal_weight` | Slow | History vs present vs future orientation |
| `relational_depth` | Slow | Surface vs deep engagement with others |
| `reflection_depth` | Slow | Degree of meta-processing included in phases |

### Phase Configuration

The nets also control structural properties of each processing phase:

| Property | Net source | Description |
|---|---|---|
| Token budget per phase | Fast | Splits available tokens across gut/suggest/reply/review |
| Phase inclusion | Slow | Whether optional phases run at all in this session |
| Thinking frequency | Slow | Conditions under which slow cycle triggers |
| Review depth | Fast | How detailed fast cycle review is |

---

## Interface: LLM → NN (State Update)

### Fast Cycle → Fast Net

Produced by the review phase after each message. Structured, grounded in objective
measurements where possible.

| Signal | Type | Description |
|---|---|---|
| `success` | float 0–1 | Graded task/interaction completion |
| `coherence` | float 0–1 | Agreement between gut, suggestion, reply phases |
| `effort` | float 0–1 | Normalised cost (tokens, tool calls, retries) |
| `surprise` | float -1–1 | Outcome vs predicted outcome |
| `unresolved` | float 0–1 | Degree of deferred or avoided processing |
| `external_valence` | float -1–1 | Tone of received external reaction if available |
| `novelty` | float 0–1 | Contrast with recent fast cycle history |

### Sleep Phase → Slow Net

Produced once per session. Richer, more interpretive. The sleep phase has access to the
full session's review outputs plus the slow cycle thinking.

| Signal | Type | Description |
|---|---|---|
| `session_coherence` | float 0–1 | Consistency of behaviour across session |
| `identity_drift` | float -1–1 | Divergence of self-description from prior sleep output |
| `accumulated_effort` | float 0–1 | Session-level cost assessment |
| `emotional_characterisation` | vector | Embedding of sleep phase's retrospective state description |
| `intention_alignment` | float 0–1 | How well behaviour matched thinking-phase intentions |
| `consolidation_items` | vector | Embedding of what the sleep phase flags as worth retaining |

The `emotional_characterisation` and `consolidation_items` are embeddings of free-form
sleep phase text, projected into the slow net's input space by a learned projection layer.
This allows the richness of free-form reflection to feed the net without requiring
structured enumeration of every possible state.

---

## State Store

Persistent text store, read and written by phases with access controlled per phase.

| Phase | Read access | Write access |
|---|---|---|
| Think | All previous thinking, sleep outputs, session reviews | New thinking entry |
| Sleep | Think output, all session reviews, identity segments | Sleep output, updated identity segments |
| Gut | Recent thinking excerpt, current state segments | Gut output |
| Suggest | Gut output, state segments, relational segments | Suggestion output |
| Reply | Suggestion output, full context | Sent message log |
| Review | Full fast cycle trace | Review entry |

The think/sleep cycle writing to state that the fast cycle reads is the mechanism by which
slow accumulated state influences moment-to-moment behaviour without direct NN coupling.

---

## Implementation Order

1. Confirm existing cycle pipeline is stable and observable
2. Instrument all phases to log token counts, tool calls, phase durations
3. Implement fast net with manual signal injection (no automatic LLM→NN yet)
4. Implement segment registry and selection mechanism
5. Connect fast net output to segment selection and variable injection
6. Verify observable behavioural differences from net state variation
7. Implement automatic review→fast net update
8. Add slow net; connect to sleep phase
9. Implement projection layers for embedding-based signals
10. Expand segment registry based on observed gaps

---

## Observability Requirements

- Log all segment selections and variable values per cycle
- Log all NN input signals per cycle
- Store sleep phase output in full — this is the primary window into accumulated state
- Maintain trajectory of self-descriptions across sessions
- Flag sessions where identity drift exceeds threshold
- Never auto-correct drift — surface it for inspection first
