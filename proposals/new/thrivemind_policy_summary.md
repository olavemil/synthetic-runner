# Thrivemind Policy Framework — Summary & Decision Framework

## Problem Statement

The Thrivemind species has a rich **constitution** describing emotional governance, collective wisdom, and staged decision-making (Articles II-III). However, the implementation is hard-coded:

- **on_message:** Fixed preprocessing → deliberation → postprocessing pipeline
- **heartbeat:** All individuals contribute and vote simultaneously; no staged thinking
- **Colony customization:** No way to enforce the constitution's intent (measured consensus, staged review, cooling-off periods)

**Goal:** Enable Thrivemind instances to self-organize via customizable **policies** that operationalize their constitution.

---

## Three Documents Created

### 1. **thrivemind_policy_framework.md**
**What:** The user's original proposal formalized into a structured specification.

**Contents:**
- Current hard-coded behavior analysis
- Proposed policy system for on_message (preprocessing/postprocessing)
- Proposed policy system for thinking phases (multi-stage with ordering & visibility)
- Example policies (Deliberate Colony, Consensus-Building Colony, Adaptive Messaging)
- Implementation plan outline
- Open questions & considerations

**Key insight:** Policies should be configurable via YAML/Python objects, with customizable:
- Preprocessing/postprocessing prompts
- Individual/collective thinking stages
- Ordering schemes (cohesion, approval, random)
- Visibility options (private, revealed, incremental)

---

### 2. **thrivemind_policy_alternatives.md**
**What:** Alternative design patterns, precedents, and implementation approaches.

**Contents:**
- Three precedents in existing codebase:
  1. **Subconscious Dreamer YAML pipelines** (declarative, extensible)
  2. **Consilium/Hecate config objects** (type-safe Python dataclasses)
  3. **Phase-based tool filtering** (existing capability to scope tools by phase)
- Three alternative approaches:
  - **Approach A:** Pure YAML pipelines (lightweight, fully declarative)
  - **Approach B:** Python dataclass policies (hybrid, type-safe)
  - **Approach C:** YAML pipelines + config objects (best of both)
- Comparison table
- **Recommendation:** Approach B (Phase 1) → Approach C (Phase 2, future)
- Testing strategy
- Integration with constitution

**Key insight:** Approach B is fastest to implement and reuses existing patterns (Consilium-like config). Approach C can be added later for full flexibility.

---

### 3. **thrivemind_policy_summary.md** (this document)
**What:** Decision framework and next steps.

---

## Recommended Implementation Path

### Phase 1: Hybrid Config Objects (Approach B) — **2-3 weeks estimated**

**Goal:** Enable customizable preprocessing, postprocessing, and multi-stage thinking via Python dataclasses.

**Key changes:**
1. Add `ThrivemindPolicies`, `StagePolicy`, `MessagePolicy` dataclasses to `library/tools/thrivemind.py`
2. Implement `load_policies()` from instance YAML config
3. Refactor `on_message()` to apply preprocessing & postprocessing policies
4. Refactor heartbeat to run thinking stages with ordering & visibility
5. Add comprehensive tests for policy variants

**Config format (in `instances/thrivemind.yaml`):**
```yaml
thrivemind:
  policies:
    on_message:
      preprocess:
        enabled: true
        prompt_template: |
          Summarize this message as a one-liner...
    thinking:
      consensus_threshold: 0.65
      stages:
        - name: individual_reflection
          type: individual
          prompt_template: |
            Reflect on the constitution...
          visibility_after: private
```

**Benefits:**
- Minimal framework changes (no new pipeline stages needed)
- Type-safe config with Python dataclasses
- Immediate value: customizable preprocessing, postprocessing, thinking stages
- Aligns with existing Consilium pattern

---

### Phase 2: YAML Pipelines (Approach C) — **Future, optional**

**Goal:** Full declarative workflow flexibility via YAML.

**Changes:**
- Implement Thrivemind-specific stages in `library/tools/pipeline.py`
- Extract policies to separate `policies/*.yaml` files
- Refactor `on_message()` and heartbeat to use `run_pipeline()`

**Benefits:**
- Fully declarative (no Python expertise needed)
- Composable stages (can repeat, disable, reorder)
- Reuses existing pipeline infrastructure
- Enables power users to design complex workflows

---

## Key Design Decisions (Phase 1)

### 1. Thinking Stages Structure

```python
@dataclass
class StagePolicy:
    name: str  # e.g., "individual_reflection"
    stage_type: str  # "individual" | "collective" | "writer"
    prompt_template: str  # Inline or reference to file
    duration_steps: int = 1  # How many LLM calls
    ordering: str = "cohesion_desc"  # Order execution
    visibility_after: str = "revealed"  # When is output visible?
    visibility_in_phase: str = "none"  # During execution (for collective)
```

### 2. Ordering Schemes

- `cohesion_asc` / `cohesion_desc` — Order by Thrivemind cohesion score
- `approval_asc` / `approval_desc` — Order by individual approval score
- `combined_asc` / `combined_desc` — Order by cohesion × approval
- `random` — Randomize order

**Use case:** Consensus-Building policy: run `cohesion_desc` first so stable members shape the dialogue.

### 3. Visibility Options

- `private` — Content unreadable until explicitly revealed
- `revealed` — Content readable by all
- `incremental` — Revealed progressively as subsequent members join
- `none` — Write-only (for drafts)

**Use case:** Individual reflection is `private`, collective dialogue is `incremental`, final synthesis is `revealed`.

### 4. Prompt Template Resolution

Prompts can be:
- **Inline:** `prompt_template: "Reflect on..."`
- **File reference:** `prompt_template: "file:policies/reflect.md"`
- **Default:** `prompt_template: "default"` (use today's hard-coded prompt)

**Variables available:**
- `{constitution}` — Current constitution text
- `{colony_snapshot}` — Current colony member list
- `{individual_name}` — Current individual's name
- `{event_summary}` — For on_message phases
- `{previous_contributions}` — For collective thinking

---

## Example Policies (Phase 1)

### Policy 1: Consensus-Building (Recommended)
Staged individual reflection → collective dialogue → synthesis.

```yaml
thrivemind:
  policies:
    thinking:
      consensus_threshold: 0.65
      stages:
        - name: individual_reflection
          type: individual
          prompt_template: |
            Reflect privately: what principle strengthens our colony?
            Write one new principle (≤18 words).
          visibility_after: private

        - name: collective_dialogue
          type: collective
          ordering: cohesion_desc
          prompt_template: |
            Review these principles so far:
            {previous_contributions}

            Add your refinement (≤30 words, signed).
          visibility_in_phase: incremental
          visibility_after: revealed

        - name: synthesis
          type: writer
          prompt_template: default
          visibility_after: revealed
```

### Policy 2: Rapid Consensus (Minimal Stages)
Single individual + writer synthesis (today's behavior).

```yaml
thrivemind:
  policies:
    thinking:
      stages:
        - name: contribution
          type: individual
          visibility_after: revealed

        - name: synthesis
          type: writer
```

### Policy 3: Cautious (Cooling-Off + Double-Check)
Multi-round voting with staged approval.

```yaml
thrivemind:
  policies:
    thinking:
      stages:
        - name: individual_reflection
          type: individual
          prompt_template: |
            Assess the proposed amendment.
            Is the colony ready? (Yes/No/Revise)
          visibility_after: private

        - name: first_vote
          type: vote
          consensus_threshold: 0.60

        - name: cooling_off
          type: pause
          duration_steps: 0  # Placeholder for wait

        - name: collective_reconsideration
          type: collective
          visibility_after: revealed

        - name: second_vote
          type: vote
          consensus_threshold: 0.65
```

---

## Implementation Checklist (Phase 1)

### Week 1: Data Models
- [ ] Add dataclasses: `StagePolicy`, `MessagePolicy`, `ThrivemindPolicies`
- [ ] Implement `ThrivemindPolicies.from_dict()` and validation
- [ ] Add defaults (return today's behavior if no policies specified)

### Week 2: on_message Refactoring
- [ ] Implement `_apply_preprocess_policy()`
- [ ] Implement `_apply_postprocess_policy()`
- [ ] Refactor `on_message()` to use policies
- [ ] Add tests for preprocessing/postprocessing variants

### Week 2.5: Heartbeat Refactoring
- [ ] Implement `_run_thinking_stages()` (main logic loop)
- [ ] Implement ordering logic (cohesion, approval, random)
- [ ] Implement visibility controls (private files, incremental reveal)
- [ ] Refactor heartbeat to use policies
- [ ] Add tests for multi-stage thinking

### Week 3: Integration & Testing
- [ ] Create test policies (Consensus-Building, Rapid, Cautious)
- [ ] Regression tests: default policies match today's behavior
- [ ] Documentation & examples
- [ ] Update constitution.md with policy recommendations

---

## Integration with Constitution

**Article II: Governance Structures**
→ Policies operationalize the "Council of Emotional Stewards" via multi-stage voting and collective dialogue.

**Article III: Amendment Procedure**
→ Policies enable the "72-hour emotional pulse pause" via staged thinking with cooling-off periods.

**Example:** Cautious policy implements Article III directly:
1. Individual reflection (pulse check)
2. First vote (emotional readiness)
3. Cooling-off (measured pause)
4. Collective reconsideration (new perspectives)
5. Second vote (final consensus)

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| **Backward compat:** Old instances without policies | Default policies load automatically, matching today's behavior |
| **Config validation:** Malformed policies break heartbeat | Validation in `ThrivemindPolicies.from_dict()` with clear error messages |
| **Performance:** Multi-stage thinking increases LLM calls | Make stages optional; default to minimal (individual + writer) |
| **Complexity:** Too many policy variants | Start with 2-3 recommended templates; document design patterns |

---

## Success Criteria

1. ✅ Default policies match today's behavior (regression test)
2. ✅ Preprocessing/postprocessing can be customized via config
3. ✅ Multi-stage thinking works with ordering & visibility controls
4. ✅ Constitution can be operationalized via policies
5. ✅ Clear examples show how to design policies for specific colonies
6. ✅ All existing tests pass; new tests cover policy variants

---

## Next Steps (User Decision)

**Option A: Proceed with Phase 1 Implementation**
- Start with dataclass design (Week 1)
- Refactor on_message first (Week 2)
- Refactor heartbeat (Week 2.5)
- Add tests & examples (Week 3)

**Option B: Request Changes to Proposal**
- Which design aspects need clarification?
- Should we prototype Phase 2 (YAML pipelines) first?
- Are there constitution requirements I missed?

**Option C: Gather More Data**
- Review existing instances' behavior (do any already customize prompts?)
- Interview imagined user personas ("What policy would YOUR colony want?")
- Prototype one policy variant (Consensus-Building) as proof-of-concept

---

## Deliverables Summary

| Document | Purpose |
|----------|---------|
| `thrivemind_policy_framework.md` | Main specification; what the proposal does |
| `thrivemind_policy_alternatives.md` | Design alternatives & precedents; how to implement |
| `thrivemind_policy_summary.md` (this) | Decision framework; when & how to proceed |

All three documents are in `proposals/new/` and ready for review.
