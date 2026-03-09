# Hecate

Hecate is a three-voice deliberation species. Three named identity voices — each with their own personality, model, and memory — propose replies independently, then one randomly-chosen voice synthesises a final reply. Between messages, all three voices think in parallel, each informed by the others' prior thoughts.

## Memory files

| File | Purpose |
|------|---------|
| `memory.md` | Shared memory across all voices |
| `constitution.md` | Shared values and behavioural guidelines |
| `{voice}_thinking.md` | Per-voice current thoughts (e.g. `aria_thinking.md`) |
| `{voice}_subconscious.md` | Per-voice subconscious evaluation |
| `{voice}_motivation.md` | Per-voice motivation notes |

## on_message flow

```
on_message(events):
  shared_memory = load memory.md + constitution.md
  context = format shared memory into prompt block

  for each voice:
    suggestion = generate_with_identity(voice, "Propose one brief reply", context)
    → returns a single sentence

  shuffle suggestions (anonymise ordering)
  composing_voice = random.choice(voices)

  final = generate_with_identity(
    composing_voice,
    "Rework these suggestions into one final reply",
    context,
  )
  ctx.send(target_room, final)

  for each voice:
    update voice's subconscious based on the conversation
    write {voice}_subconscious.md
```

## heartbeat flow

```
heartbeat():
  for iteration in range(thinking_iterations):
    shared_memory = load memory.md + constitution.md
    snapshot_context = load all voice thinking files

    for each voice:
      others_thinking = {other voices' thoughts from previous iteration}
      thought = think_with_context(voice, context, others_thinking, voice_memory)
      new_thoughts[voice] = thought

    write all thoughts to {voice}_thinking.md
    previous_thoughts = new_thoughts
```

## Config keys (`hecate:` block in instance YAML)

```yaml
hecate:
  voices:
    - name: Aria
      model: ""           # uses instance default if empty
      provider: ""
      personality: ""     # free-form personality description
    - name: Sable
      ...
    - name: Lune
      ...
  thinking_iterations: 2  # how many rounds of parallel thinking per heartbeat
  voice_space: main       # which space name to reply to when room is ambiguous
```

Exactly 3 voices are required.
