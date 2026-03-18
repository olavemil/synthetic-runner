# Thrivemind

Thrivemind is a colony deliberation species. A colony of individuals — each with randomised personality dimensions on multiple axes — propose replies, vote on each other's suggestions, and converge on a unified response. Between messages, the colony collectively drafts and votes on a shared constitution, and eligible individuals reproduce (spawn offspring) while being replaced.

## Memory files

| File | Purpose |
|------|---------|
| `constitution.md` | Colony's shared values and principles (evolves over time) |
| `sessions.md` | Session log |
| `colony.md` | Snapshot of current colony members and their approval scores |
| `contributions.md` | Latest round's proposed constitution lines |
| `candidate.md` | Latest proposed constitution draft before voting |
| `processes.md` | Auto-generated description of active policies and processes |
| `reflections/{name}.md` | Per-individual reflections on colony state |

## on_message flow

```
on_message(events):
  policies = load_policies()
  colony = load_colony() or spawn_initial_colony()
  constitution = load_constitution()

  # PREPROCESS (customizable via policies.on_message.preprocess)
  if custom preprocess:
    message_summary = apply_preprocess(events, policy.prompt_template)
  else:
    message_summary = summarize_message_history(events)

  for each individual in colony:
    reflection = reflect_on_colony(individual, colony, constitution, ...)
    individual_contexts[individual] = build context from reflection

  suggesters = select_suggesters(colony, n=colony_size * suggestion_fraction)
  result = deliberate(colony, prompt, suggesters, individual_contexts)

  writer = Identity("Colony", writer_model)
  final = recompose(writer, winner_message, all_candidates)

  # POSTPROCESS (customizable via policies.on_message.postprocess)
  if custom postprocess:
    final = apply_postprocess(final, policy.prompt_template)

  ctx.send(target_room, "Consensus: X%\n\n" + final)
  colony = update_cohesion(colony, votes, winner)
  save_colony(colony)
```

## heartbeat flow

```
heartbeat():
  policies = load_policies()
  write_process_description()  → processes.md (self-inspection)
  colony = load_colony() or spawn_initial_colony()
  constitution = load_constitution()

  for each individual:
    reflection = reflect_on_colony(individual, colony, constitution, ...)

  # THINKING STAGES (customizable via policies.thinking.stages)
  if policy stages configured:
    for each stage in stages:
      run_thinking_stage(colony, stage, constitution, reflections)
        → individual: each writes independently
        → collective: ordered access to shared document
        → writer: colony writer synthesizes
  else:
    for each individual:
      line = contribute_constitution_line(individual, constitution)

  proposed = rewrite_constitution(lines, current_constitution)

  # Round 1 vote
  # Round 2 vote (if needed)
  if adopted: save_constitution(proposed)

  colony = vote_peer_approval(colony)
  colony = run_spawn_cycle(colony)

  # POST-SPAWN STAGES (customizable via policies.thinking.post_spawn)
  if post_spawn stages configured:
    for each stage: run_thinking_stage(...)

  save_colony(colony)
  organize_phase()
  creative_phase()
```

## Config keys (`thrivemind:` block in instance YAML)

```yaml
thrivemind:
  colony_size: 12               # number of individuals in the colony
  suggestion_fraction: 0.5      # fraction of colony that proposes replies
  approval_threshold: 3         # min approval to be eligible for spawning
  consensus_threshold: 0.6      # required vote share for consensus / constitution adoption
  suggestion_model: ""          # provider/model for suggestions and voting (default: instance model)
  writer_model: ""              # provider/model for final composition and constitution rewrite
  voice_space: main             # logical space name for outbound replies
```

## Policies (`thrivemind.policies:` block in instance YAML)

Policies customize the colony's processing pipeline. All are optional; defaults match pre-policy behavior.

```yaml
thrivemind:
  policies:
    on_message:
      preprocess:
        enabled: true             # whether to preprocess incoming messages
        prompt_template: default  # "default" or custom prompt with {message}/{events}
      postprocess:
        enabled: true             # whether to postprocess the final response
        prompt_template: default  # "default" or custom prompt with {candidate}/{constitution}

    thinking:
      stages:                     # 0-3 pre-voting stages (default: none = single contribution)
        - name: "individual_reflection"
          type: "individual"      # individual | collective | writer
          prompt_template: default
          ordering: "random"      # cohesion_asc/desc, approval_asc/desc, combined_asc/desc, random
          visibility_after: "revealed"     # private | revealed | incremental | none
          visibility_in_phase: "none"      # for collective stages: incremental | revealed | none

      post_spawn:                 # 0-2 post-spawn stages (same structure as stages)
        - name: "orientation"
          type: "collective"
```

### Stage types

- **individual**: Each colony member writes independently to their own output. No one sees others' work (until visibility changes).
- **collective**: Colony members contribute to a shared document in sequence. Ordering and visibility_in_phase control who sees what.
- **writer**: The colony writer synthesizes all prior stage outputs into a single output.

### Ordering schemes

| Scheme | Description |
|--------|-------------|
| `random` | Randomized order (default) |
| `cohesion_asc` / `cohesion_desc` | By cohesion score |
| `approval_asc` / `approval_desc` | By approval score |
| `combined_asc` / `combined_desc` | By cohesion x (approval+1) |

### Visibility options

| Option | Description |
|--------|-------------|
| `private` | Content not visible to others |
| `revealed` | Content visible to all |
| `incremental` | Revealed progressively (person N sees N-1's work) |
| `none` | Content not shared |

### Self-inspection

The colony writes `processes.md` at the start of each heartbeat, describing the active configuration and policies. Colony members can read this file to understand their own governance processes.
