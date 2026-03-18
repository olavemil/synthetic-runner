# Thrivemind Self-Organization Policy Framework

## Overview

The Thrivemind colony currently has hard-coded logic for message preprocessing/postprocessing and thinking phases. The constitution (Article II & III) calls for configurable governance with "emotional consensus" and "measured collective wisdom," but lacks tooling to enforce this beyond voting on constitution text.

This proposal introduces a **Policy Framework** — a structured, YAML-based system to customize:
- Message flow (preprocessing/postprocessing)
- Thinking phase structure (individual/collective thinking, ordering, visibility)
- Post-spawn reflection phases
- Voting thresholds and ordering heuristics

## Current Hard-Coded Behavior

### on_message Flow
```
incoming events
  → format_events() [no customization]
  → select_target_room() [no customization]
  → per-individual reflect_on_colony() [uses fixed PROMPT_REFLECT]
  → run_messaging_phase() [hardcoded internal messaging]
  → deliberate() [all suggesters propose simultaneously]
  → recompose() [one-shot rewrite with PROMPT_RECOMPOSE]
  → send response
```

**Problem:** No way to customize or polymorph this flow. All colonies follow the same pattern.

### Thinking Phase (heartbeat)
```
per-individual contribute_constitution_line() [fixed PROMPT_CONTRIBUTE]
  → rewrite_constitution() [fixed PROMPT_REWRITE]
  → Round 1 vote [no phase restrictions]
  → Round 2 vote (conditional) [no phase restrictions]
```

**Problem:** No stages; all individuals contribute and vote simultaneously. No intermediate collective reflection or staged visibility.

## Proposed Policy System

### 1. **on_message Policy** (Message-Triggered Reasoning)

```yaml
policies:
  on_message:
    # Preprocess: customize how incoming events are summarized/reframed
    preprocess:
      enabled: true
      prompt_template: |
        Summarize this message as a short one-liner for the colony:
        "{message}"
        Keep it under 15 words, emphasizing tone and intent.

    # Individual Reflection: customize per-individual reasoning
    reflect:
      enabled: true
      prompt_template: default  # or custom markdown file
      visibility: private

    # Post-process candidate responses before sending
    postprocess:
      enabled: true
      prompt_template: |
        You are the unified voice of the colony. Refine this draft response to match
        the colony's constitution (Article I: shared hope, Article II: emotional stewards).
        Ensure empathy and consensus-seeking tone.
        Draft: "{candidate}"

    # Vote guidance: customize how individuals vote on candidates
    vote_guidance:
      prompt_template: default
```

**Implementation Notes:**
- `prompt_template` can be inline (string) or reference `default` (species built-in) or `file:path/to/file`
- Preprocessing happens once on the full message; postprocessing happens after deliberation
- Both are optional (can disable by `enabled: false`)
- Preprocessing output becomes the "scoped_conversation" in subsequent phases

---

### 2. **Thinking Policy** (Constitution & Reflection Phases)

Structured as ordered stages with per-stage controls.

```yaml
policies:
  thinking:
    # Configurable voting thresholds
    consensus_threshold: 0.60
    approval_threshold: 3  # min approval to spawn

    # Pre-voting stages (0-3 allowed, executed in order)
    stages:
      - name: "individual_reflection"
        type: "individual"  # each person writes privately
        duration_steps: 2
        prompt_template: |
          Reflect on the colony's current constitution.
          What principle would strengthen our emotional foundation?
          Write one new principle (≤18 words).

        # After this stage, can individuals see others' work?
        visibility_after: "private"  # or "revealed" or "partial"

      - name: "collective_dialogue"
        type: "collective"  # all contribute to same document
        duration_steps: 3
        prompt_template: |
          Review the emerging principles. Build on them.
          The colony document is:\n{shared_doc}
          Add your insight in ~30 words, signed by your name.

        # How to order participants (if not all contribute simultaneously)
        ordering: "cohesion_desc"  # asc/desc/random/approval_asc/approval_desc/combined_asc/combined_desc

        # What can participants see?
        visibility_in_phase: "incremental"  # full/incremental/none
        visibility_after: "revealed"

      - name: "synthesis"
        type: "writer"  # colony writer synthesizes
        prompt_template: default  # PROMPT_REWRITE
        visibility_after: "revealed"

    # Post-spawn phases (0-2 allowed, same structure as stages)
    post_spawn:
      - name: "offspring_orientation"
        type: "collective"
        duration_steps: 1
        prompt_template: |
          Welcome, new members of the colony.
          Current constitution excerpt:\n{constitution_head_100}
          What is your first insight about our shared values?
```

**Type Definitions:**
- `individual`: Each person writes to their private reflection file (unread by others until visibility change)
- `collective`: Everyone contributes to the same document, possibly in sequence
- `writer`: The colony writer (or designated identity) synthesizes input

**Ordering Schemes:**
- `cohesion_asc` / `cohesion_desc`: Order by cohesion score (Thrivemind's metric of colony stability)
- `approval_asc` / `approval_desc`: Order by approval score
- `combined_asc` / `combined_desc`: Order by cohesion × approval
- `random`: Randomize order each phase

**Visibility Options:**
- `private`: Others cannot see content until explicitly revealed
- `revealed`: Content is visible to all
- `incremental`: Revealed progressively (person N sees N-1's work)
- `partial`: Customizable list of who sees what (future extension)
- `none`: Content is never visible (write-only)

---

### 3. **Data Structure in Instance Config**

```yaml
thrivemind:
  # Existing config
  colony_size: 12
  suggestion_fraction: 0.5
  approval_threshold: 3
  consensus_threshold: 0.60

  # New: policies reference or inline definition
  policies:
    path: "policies.yaml"  # OR inline the full dict below
    # policies:
    #   on_message: { ... }
    #   thinking: { ... }
```

**Policy loading:**
```python
def load_policies(ctx: InstanceContext, cfg: ThrivemindConfig) -> ThrivemindPolicies:
    """Load policies from config or default file."""
    policies_path = cfg.policies.get("path")
    if policies_path:
        raw = ctx.read(policies_path)
        return parse_policies(yaml.safe_load(raw))
    elif "policies" in cfg.raw_config:
        return parse_policies(cfg.raw_config["policies"])
    else:
        return ThrivemindPolicies()  # sensible defaults
```

---

## Implementation Plan Outline

### Phase 1: Data Model & Config Parsing
- [ ] Add `ThrivemindPolicies` dataclass with nested `PreprocessPolicy`, `ThinkingPolicy`, `StagePolicy`
- [ ] Implement `load_policies()` and policy validation
- [ ] Update `ThrivemindConfig` to accept `policies` dict
- [ ] Add default policies (today's hard-coded behavior as baseline)

### Phase 2: on_message Refactoring
- [ ] Extract preprocessing logic into `_apply_preprocess_policy()`
- [ ] Extract postprocessing logic into `_apply_postprocess_policy()`
- [ ] Refactor `on_message()` to use policies instead of hard-coded prompts
- [ ] Add tests for each policy variant

### Phase 3: Thinking Phase Refactoring
- [ ] Implement stage runner: `_run_thinking_stages()`
- [ ] Implement visibility/ordering logic
- [ ] Refactor heartbeat to use policy-driven stages
- [ ] Add tests for multi-stage thinking

### Phase 4: Post-Spawn Extension (Future)
- [ ] Implement post-spawn stages (after run_spawn_cycle)
- [ ] Ensure offspring have orientation phase option

### Phase 5: Constitution Integration
- [ ] Define policy format in constitution (Article II amendment)
- [ ] Allow constitution voting to propose policy changes (future)
- [ ] Document policy design as part of colony charter

---

## Example Policies

### Policy 1: "Deliberate Colony" (Current Behavior)
Minimal customization; largely preserves today's flow.

```yaml
policies:
  on_message:
    preprocess:
      enabled: false
    postprocess:
      enabled: true
      prompt_template: default

  thinking:
    consensus_threshold: 0.60
    stages:
      - name: "contribution"
        type: "individual"
        prompt_template: default  # PROMPT_CONTRIBUTE
        visibility_after: "revealed"
```

### Policy 2: "Consensus-Building Colony"
Staged individual reflection, then collective dialogue.

```yaml
policies:
  on_message:
    preprocess:
      enabled: true
      prompt_template: |
        Summarize this message in one line emphasizing emotional tone:
        "{message}"

    postprocess:
      enabled: true
      prompt_template: |
        Refine this draft to align with the colony's constitution.
        Draft: "{candidate}"

  thinking:
    consensus_threshold: 0.65
    stages:
      - name: "individual_reflection"
        type: "individual"
        duration_steps: 2
        prompt_template: |
          Reflect privately: what principle strengthens us?
          One principle, ≤18 words.
        visibility_after: "private"

      - name: "collective_build"
        type: "collective"
        duration_steps: 3
        ordering: "cohesion_desc"
        prompt_template: |
          Emerging principles:\n{principles_so_far}
          Add your refinement or new principle (≤30 words).
        visibility_in_phase: "incremental"
        visibility_after: "revealed"

      - name: "synthesis"
        type: "writer"
        prompt_template: default
```

### Policy 3: "Adaptive Messaging Colony"
Custom preprocessing; message-specific framing.

```yaml
policies:
  on_message:
    preprocess:
      enabled: true
      prompt_template: |
        This message was just received. Rephrase it for the colony:

        Original: "{message}"

        Reframing: (Be brief, emphasize intent and emotional subtext)
```

---

## Open Questions & Considerations

1. **Prompt Template Resolution**
   - Should we support Jinja2 templating in prompts, or simple `{var}` replacement?
   - Which variables should be available at each stage?
   - Should policies be able to reference external files (e.g., `file:docs/style_guide.md`)?

2. **Stage Synchronization**
   - If a stage is "collective" with ordering, do we run it serially (slow) or parallel (loses incremental visibility)?
   - Should "duration_steps" be wall-clock time or LLM call count?

3. **Visibility & Inheritance**
   - If Stage 1 is "private" and Stage 2 is "revealed," do people see Stage 1 content at the start of Stage 2?
   - Should there be a "partial" visibility mode allowing selective sharing (e.g., Stage 1 → only elders see early)?

4. **Constitution Integration**
   - Should the constitution itself describe the colony's policies (as structured data)?
   - Can individuals propose policy changes via constitution voting?

5. **Backward Compatibility**
   - Old instances without policies should use sensible defaults (today's behavior).
   - Should we validate that policy changes don't break existing memory files?

6. **Telemetry & Debugging**
   - How should policy execution be logged? Per-stage summaries?
   - Should failed policy application fall back to defaults or error?

---

## Files Likely to Change

- `library/tools/thrivemind.py`: Add policy parsing, stage runner, visibility logic
- `library/species/thrivemind/__init__.py`: Refactor `on_message()` and heartbeat to use policies
- `library/tools/patterns.py`: May add `run_thinking_stages()` helper
- Tests: `tests/test_thrivemind_*.py` for policy variants

---

## Next Steps

1. **Review & Feedback** — Does this structure align with the constitution's intent?
2. **Data Model Definition** — Lock down Python classes for policies
3. **Prototype** — Implement Phase 1 (config parsing) to validate the design
4. **Design Doc** — Convert this proposal into detailed spec with examples
