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
| `reflections/{name}.md` | Per-individual reflections on colony state |

## on_message flow

```
on_message(events):
  colony = load_colony() or spawn_initial_colony()
  constitution = load_constitution()
  message_summary = summarize_message_history(events)

  for each individual in colony:
    prior_reflection = load_reflection(individual)
    reflection = reflect_on_colony(individual, colony, constitution, prior_reflection, message_summary)
    save_reflection(individual, reflection)
    individual_contexts[individual] = build context from reflection

  suggesters = select_suggesters(colony, n=colony_size * suggestion_fraction)
    → weighted by approval score

  result = deliberate(colony, prompt, suggesters, individual_contexts)
    → each suggester generates a candidate reply
    → all colony members vote (Borda count)
    → returns winner + scores + consensus status

  writer = Identity("Colony", writer_model)
  if has_consensus:
    final = recompose(writer, winner_message, all_candidates)
  else:
    fallback = join top-scoring candidates until threshold coverage
    final = recompose(writer, fallback, selected_candidates)

  ctx.send(target_room, "Consensus: X%\n\n" + final)

  colony = update_approvals(colony, votes, winner)
    → winner's first-place voters gain +1 approval
    → losers' voters lose -1 approval
  save_colony(colony)
```

## heartbeat flow

```
heartbeat():
  colony = load_colony() or spawn_initial_colony()
  constitution = load_constitution()

  for each individual:
    reflection = reflect_on_colony(individual, colony, constitution, ...)
    save_reflection(individual, reflection)

  lines = []
  for each individual:
    line = contribute_constitution_line(individual, constitution)
      → proposes exactly one new principle (≤18 words)
    lines.append(line)

  save_contributions(lines)
  proposed = rewrite_constitution(lines, current_constitution)
    → ColonyWriter identity synthesises a new draft
  save_candidate(proposed)

  # Round 1 vote
  for each individual:
    accepted = vote_constitution(individual, current, proposed)
  if acceptance_ratio > consensus_threshold:
    adopted = True
  else:
    # Round 2 vote with round-1 context
    for each individual:
      accepted = vote_constitution(individual, current, proposed, round1_context)
    adopted = acceptance_ratio_round2 > consensus_threshold

  if adopted:
    save_constitution(proposed)

  colony = run_spawn_cycle(colony)
    → individuals with approval >= threshold reproduce and are replaced by offspring
    → colony size held constant

  save_colony(colony)
  save_colony_snapshot(colony)   → writes colony.md
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
