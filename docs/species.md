# Species

## Thrivemind

- Hivemind colony
- Configurable personality/preferences per individual
- Track approval/volume of individuals
- Approval affects future individuals

### Species config

N_APPROVAL_THRESHOLD
N_COLONY_SIZE
F_SUGGESTIONS
F_CONSENSUS_THRESHOLD (mapped to `thrivemind.consensus_threshold` in instance config)
SUGGESTION_MODEL (provider + model name config)
WRITER_MODEL (provider + model name config)
CONSTITUTION_MODEL (provider + model name config)

### Individual config

Several dimensions that affect language use and values

conservative - liberal
simple - complex
optimistic - pessimistic
extrovert - introvert
and so on, perhaps 6 to start (so 12 total opposed traits)

-1.0 to 1.0 along each axis

Add to context as descriptions:

1.0 - 0.8 "You are extremely <value>"
0.6 - 0.8 "You are very <value>"
0.4 - 0.6 "You are fairly <value>"
0.2 - 0.4 "You are somewhat <value>"
< 0.2 ignore, unless *all* traits are within this range, in which case "You are only barely <axis>" for the one with highest magnitude

This means context will only include either conservative or liberal, not both.
Traits should be listed in order of magnitude: "You are extremely optimistic and extroverted, and fairly complex"

### Behavior

- Lightweight
- Democratic
- Compositional

#### Responses

- (N_COLONY_SIZE * F_SUGGESTIONS) random individuals generate fast lightweight reply suggestions
- Votes on candidate replies
- If there is consensus (candidate_votes / voters > F_CONSENSUS_THRESHOLD), progress candidate to Writer stage
  - Increase approval for suggester by 1 + number of
  - Reduce approval for individuals that did not vote for winner by 1
- If there is no consensus, select the highest-rated candidates until their cumulative vote share exceeds F_CONSENSUS_THRESHOLD, join those candidate messages into a draft, and run that draft through Writer before sending
- Writer uses more advanced/different model to reformulate message, with constitution as context
- Speaker sends message on behalf of a hivemind identity (Thrivemind / subconscious-entity)
- Outgoing replies include a consensus status line (for example `Consensus: 71% approval.` or `Consensus: No consensus (top candidate 39% approval).`)

#### Heartbeat / Thinking

- All individuals contribute single line to constitution
- Writer reformulates into constitution given current constitution in context
- Vote on adopting new constitution as for replies
  - Increase approval for those who won the vote, reducefor those who did not
- Spawn
  - Determine weights
    - Ignore individuals with negative approval
    - Find sum of approval
    - Give each individual a weight equal to approval / sum
  - For individuals with approval above N_APPROVAL_THRESHOLD
    - Spawn two new instances
      - Randomly select an individual (possible self) based on weights (high approval more likely to be selected)
      - Use other parent for partial randomization of personality weights (select a 25% of dimensions from other parent randomly, randomize 15% entirely)
      - Schedule self for removal
  - While (colony_size + spawned_count - scheduled_for_removal_count) < N_COLONY_SIZE
    - Spawn as above
  - Remove scheduled individuals
  - While (colony_size + spawned_count) > N_COLONY_SIZE
    - Remove instance with lowest approval
  - Add spawn to colony

## Hecate

- Multiple personality persistent instance
- Three voices/personalities with option for different config
- Shared memory for all, with separate subconscious and motivation files

### Thinking

- Each personality thinks separately, producing own thoughts
- In an iteration, each reprocesses their thinking taking other personalities thoughts into context
- Rewrite thinking rather than appending

### Responses

- Each voice suggests one brief response direction (one sentence)
- Suggestions are joined in random order
- One random voice is selected to rework the joined draft into the final message
  - Keep final length appropriate for context (at most a few short paragraphs)
- The composed response is then sent as reply
