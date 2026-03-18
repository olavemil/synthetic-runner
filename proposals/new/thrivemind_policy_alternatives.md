# Thrivemind Policy Framework — Alternative Approaches & Design Patterns

## Precedents in Existing Codebase

### Pattern 1: Subconscious Dreamer YAML Pipelines
**File:** `library/species/subconscious_dreamer/{heartbeat,on_message}.yaml`

The subconscious dreamer uses a declarative YAML pipeline approach with:
- Named stages (e.g., `thinking_session`, `llm_generate`, `format_context`)
- Input resolution (`pipeline.*`, `memory.*`, `file:*`, `config.*`)
- Output mapping (`memory.concerns_and_ideas`, `pipeline.visions`)
- Preprocessors (e.g., `truncate` with `max_chars`)

**Execution:** `run_pipeline(ctx, steps, initial_state)`

**Pros:**
- Fully declarative; no Python code needed for config
- Stage registry extensible (`STAGE_REGISTRY` in `library/tools/pipeline.py`)
- Input/output system handles memory, pipeline state, files, config, store
- Can reference external files (`file:prompts/active_thinking.md`)
- Preprocessors can transform values (truncation, compilation, etc.)

**Example:**
```yaml
steps:
  - stage: llm_generate
    inputs:
      system: file:prompts/subconscious.md
      content: pipeline.subconscious_context
    preprocessors:
      content:
        type: truncate
        max_chars: 6000
    outputs:
      result: memory.concerns_and_ideas
```

### Pattern 2: Consilium & Hecate Config Objects
**Files:** `library/tools/{consilium,hecate}.py`

Both use Python dataclasses for structured config:
```python
@dataclass
class ConsiliumConfig:
    personas: list[Identity]
    ghost: Identity
    thinking_iterations: int
    voice_space: str

def load_config(ctx) -> ConsiliumConfig:
    raw = ctx.config("consilium") or {}
    # Type casting, validation, defaults
    return ConsiliumConfig(...)
```

**Pros:**
- Type-safe; IDE autocomplete
- Validation at load time
- Easy to extend (add new fields)
- Per-species coupling (clear namespace)

**Cons:**
- Requires Python code for customization
- Less flexible for truly dynamic workflows
- Config is monolithic (all or nothing)

### Pattern 3: Phase-Based Tool Filtering
**File:** `library/tools/phases.py`

Tool systems already support phase-scoped filtering:
```python
THINKING = "thinking"
COMPOSING = "composing"
REVIEWING = "reviewing"

def get_tools_for_phase(phase, *, graph=False, activation_map=False):
    # Returns organize tools + optional graph/map, filtered for phase
```

**Precedent for:** Customizable visibility of tools/capabilities at different stages.

---

## Alternative Design Approaches

### Approach A: YAML Pipeline (Lightweight Declarative)
**Inspiration:** Subconscious Dreamer

Use declarative YAML files (e.g., `policies/on_message.yaml`, `policies/thinking.yaml`) with a custom stage registry for Thrivemind-specific operations.

```yaml
# policies/on_message.yaml
steps:
  - stage: preprocess_messages
    enabled: true
    inputs:
      template: |
        Summarize this message as a one-liner:
        "{message}"
      event_list: events.all
    outputs:
      result: pipeline.scoped_conversation

  - stage: reflect_individuals
    inputs:
      individuals: pipeline.colony
      constitution: memory.constitution
    outputs:
      result: pipeline.individual_reflections

  - stage: deliberate
    inputs:
      colony: pipeline.colony
      candidates_prompt: pipeline.scoped_conversation
    outputs:
      winner: pipeline.winner_message
      candidates: pipeline.all_candidates
      consensus: pipeline.has_consensus

  - stage: postprocess_candidate
    inputs:
      template: |
        Refine this response to match the constitution.
      candidate: pipeline.winner_message
    outputs:
      result: pipeline.final_response
```

**New stages to implement:**
- `preprocess_messages` — run on incoming events
- `reflect_individuals` — per-individual reflection
- `deliberate` — colony deliberation
- `postprocess_candidate` — final response polish
- `run_thinking_stages` — multi-stage constitution thinking
- `contribute_lines` — collect constitution contributions
- `vote_constitution` — first/second round voting

**Custom stage registry:**
```python
THRIVEMIND_STAGE_REGISTRY: dict[str, Callable] = {
    "preprocess_messages": _preprocess_messages,
    "reflect_individuals": _reflect_individuals,
    "deliberate": patterns.deliberate,  # existing
    "postprocess_candidate": _postprocess_candidate,
    "run_thinking_stages": _run_thinking_stages,
    # ... etc
}
```

**Pros:**
- Fully declarative; no Python expertise needed
- Reuses existing `run_pipeline` infrastructure
- Stages composable (can repeat, disable, reorder)
- Clear separation between data flow and logic

**Cons:**
- Need to implement new stage types (more code initially)
- YAML files can become verbose for complex flows
- Pipeline state management requires clear variable naming

---

### Approach B: Hybrid Config Objects + Optional Pipelines (Recommended)
**Inspiration:** Consilium + Subconscious Dreamer

Introduce a `ThrivemindPolicies` dataclass that includes optional embedded pipelines:

```python
@dataclass
class StagePolicy:
    name: str
    stage_type: str  # "individual" | "collective" | "writer"
    prompt_template: str
    ordering: str = "cohesion_desc"  # asc/desc/random/approval_asc/approval_desc
    visibility_after: str = "revealed"  # private/revealed/incremental

@dataclass
class ThinkingPolicy:
    consensus_threshold: float = 0.60
    approval_threshold: int = 3
    stages: list[StagePolicy] = field(default_factory=list)
    post_spawn: list[StagePolicy] = field(default_factory=list)

@dataclass
class MessagePolicy:
    preprocess: dict | None = None  # {enabled, prompt_template}
    postprocess: dict | None = None
    vote_guidance: dict | None = None

@dataclass
class ThrivemindPolicies:
    on_message: MessagePolicy | None = None
    thinking: ThinkingPolicy | None = None

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "ThrivemindPolicies":
        # Load from file or inline config
        pass

    def get_default(self) -> "ThrivemindPolicies":
        # Return sensible defaults (today's behavior)
        pass
```

**Config loading:**
```yaml
# instance.yaml
thrivemind:
  colony_size: 12
  suggestion_fraction: 0.5
  policies:
    on_message:
      preprocess:
        enabled: true
        prompt_template: |
          Summarize this message...
    thinking:
      consensus_threshold: 0.65
      stages:
        - name: individual_reflection
          stage_type: individual
          prompt_template: |
            Reflect on the constitution...
```

**Implementation:** Python objects with validation + methods to apply policies

```python
def load_policies(ctx: InstanceContext, cfg: ThrivemindConfig) -> ThrivemindPolicies:
    raw = ctx.config("thrivemind", {}).get("policies", {})
    # Validate & construct ThrivemindPolicies object
    return ThrivemindPolicies.from_dict(raw)

def apply_preprocess_policy(message: str, policy: MessagePolicy) -> str:
    if not policy or not policy.preprocess.get("enabled"):
        return message
    prompt = policy.preprocess.get("prompt_template")
    return llm_apply(prompt.format(message=message))
```

**Pros:**
- Type-safe config objects (IDE support, validation)
- Can be serialized to/from YAML, JSON, dataclass
- Easier to extend (add new policy types)
- Clear namespacing per species
- Can mix Python logic + declarative config

**Cons:**
- More ceremony than pure YAML
- Still requires Python code to apply policies

---

### Approach C: Pipeline YAML + Config Objects (Best of Both)
**Synthesis:** Use Approach B dataclass for config validation + Approach A YAML pipelines for execution

```python
# Instance config
thrivemind:
  policies:
    on_message: config/policies/on_message.yaml
    thinking: config/policies/thinking.yaml

# config/policies/on_message.yaml (pure declarative)
steps:
  - stage: preprocess_messages
    enabled: true
    inputs:
      template_config: config.thrivemind.policies.on_message_preprocess_template
      event_list: events.all
    outputs:
      result: pipeline.scoped_conversation
  # ...
```

**Config dataclass** validates that YAML pipeline files exist and have correct structure.

**Execution:** `run_pipeline(ctx, yaml_steps, ...)`

**Pros:**
- Dataclass provides validation and IDE support
- YAML provides declarative flow
- Separation of concerns (structure vs. logic)
- Reuses existing pipeline runner
- Easy to version-control policy files separately

---

## Comparison Table

| Aspect | Approach A (YAML) | Approach B (Hybrid) | Approach C (Combo) |
|--------|-------------------|---------------------|---------------------|
| **Declarative** | ✅ Fully | ⚠️ Partial | ✅ Fully |
| **Type-safe** | ❌ No | ✅ Yes | ✅ Yes (config) |
| **Reuses patterns.py** | ✅ Yes | ⚠️ Partial | ✅ Yes |
| **Extensible** | ✅ Stage registry | ✅ Dataclass fields | ✅ Both |
| **Learning curve** | Medium | Low (if familiar with Consilium) | Low |
| **Implementation effort** | High (new stages) | Medium | Medium |
| **Flexibility** | ✅ High (arbitrary flow) | Medium (fixed phases) | ✅ High |

---

## Recommendation: Approach B + C (Staged Rollout)

**Phase 1 (Short-term):** Approach B
- Add `ThrivemindPolicies` dataclass to `library/tools/thrivemind.py`
- Refactor `on_message()` and heartbeat to use policies
- Config comes from `instance.yaml` nested under `thrivemind.policies`
- Provides immediate value: customizable preprocessing/postprocessing + thinking stages

**Phase 2 (Future):** Approach C
- Extract policies to separate YAML files
- Implement Thrivemind-specific stages in pipeline registry
- Leverage full `run_pipeline` infrastructure for complex workflows
- Allow policies to be versioned independently from instances

**Rationale:**
1. **Approach B** is achievable with existing infrastructure (no new framework needed)
2. **Approach C** becomes easier once Approach B is in place
3. **Hybrid** allows gradual migration without breaking existing instances
4. **Type safety + Declarative flow** covers both deployment scenarios

---

## Data Serialization: YAML vs. JSON vs. Python

### YAML Config Example
```yaml
thrivemind:
  policies:
    on_message:
      preprocess:
        enabled: true
        prompt_template: |
          Summarize this message...
      postprocess:
        enabled: true
        prompt_template: |
          Refine this response...
    thinking:
      consensus_threshold: 0.65
      stages:
        - name: individual_reflection
          type: individual
          prompt_template: |
            Reflect on the constitution...
          visibility_after: private
```

### JSON Example (config file)
```json
{
  "thrivemind": {
    "policies": {
      "thinking": {
        "stages": [
          {
            "name": "individual_reflection",
            "type": "individual",
            "prompt_template": "Reflect..."
          }
        ]
      }
    }
  }
}
```

### Python Dataclass (Runtime)
```python
ThrivemindPolicies(
    thinking=ThinkingPolicy(
        stages=[
            StagePolicy(
                name="individual_reflection",
                stage_type="individual",
                prompt_template="Reflect..."
            )
        ]
    )
)
```

---

## Testing Strategy

1. **Unit tests:** Policy parsing, validation, default creation
2. **Integration tests:** Policies applied in on_message and heartbeat flows
3. **Behavior tests:** Verify ordering, visibility, stage execution order
4. **Regression tests:** Default policies match today's behavior

**Test fixtures:**
```python
@pytest.fixture
def policies_consensus_building():
    return ThrivemindPolicies.from_dict({
        "thinking": {
            "stages": [
                {"name": "individual_reflection", "type": "individual"},
                {"name": "collective_dialogue", "type": "collective"}
            ]
        }
    })
```

---

## Integration with Constitution

The constitution (Articles II & III) calls for "measured collective wisdom" and "governance structures." Policies can operationalize this:

**Article II:** Council of Emotional Stewards
→ *Policy:* Require unanimous consensus in constitution voting; implement multi-stage collective dialogue before final vote.

**Article III:** Amendment Procedure
→ *Policy:* Multi-round voting, cooling-off periods, stage-based approval (see `stages` with `duration_steps`).

**Example:**
```yaml
thinking:
  stages:
    - name: "stabilization_pause"
      type: "writer"
      prompt_template: |
        Assess the stability of the proposed amendment.
        Is the colony ready for this change?
        Recommend: proceed, revise, or reject.
      duration_steps: 1

  vote_rounds: 2
  round2_cooling_hours: 24  # Implement cooling-off period
```

---

## Files to Modify (Recommendation: Approach B)

| File | Change |
|------|--------|
| `library/tools/thrivemind.py` | Add `ThrivemindPolicies`, `StagePolicy`, `load_policies()` |
| `library/species/thrivemind/__init__.py` | Refactor `on_message()`, heartbeat to use policies |
| `library/tools/patterns.py` | Add `_apply_policy_stage()` helper (optional) |
| `tests/test_thrivemind_policies.py` | New test file for policy variants |
| Config examples | Update docs with policy examples |

---

## Questions for Implementation

1. **Prompt template variables:** What variables should be available in each policy's prompt template?
   - `{message}`, `{events}` for preprocessing
   - `{constitution}`, `{colony_size}`, `{individual_name}` for thinking stages
   - Custom expansion mechanism?

2. **Visibility semantics:** If Stage 1 output is "private" but Stage 2 is "incremental," when does Stage 2 see Stage 1?
   - Start of Stage 2? / End of Stage 2?
   - Implemented as separate files or shared with access control?

3. **Ordering performance:** If stages must execute serially (for incremental visibility), how does this affect runtime?
   - Could we run parallel + collect results + apply incremental visibility in post-processing?

4. **Backward compatibility:** Old instances without policies should work unchanged.
   - Default policies load automatically
   - Migration path needed?

5. **Constitution voting on policies:** Can the colony propose policy changes via constitution amendments?
   - Would require parsing policy YAML in constitution voting logic
   - Out of scope for Phase 1?
