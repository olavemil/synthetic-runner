# Subconscious Dreamer

Subconscious Dreamer is a three-phase thinking, three-phase response species. Between messages it runs an introspective cycle — free-form thinking, subconscious concern extraction, and associative dreaming. When responding to messages it draws on this internal state through intuition, worry, and action phases before composing a reply.

## Memory files

| File | Purpose |
|------|---------|
| `thinking.md` | Accumulated free-form thoughts (written by active thinking session) |
| `concerns_and_ideas.md` | Current concerns, ideas, and curiosities (written by subconscious phase) |
| `dreams.md` | Associative dreams and images (written by dreaming phase) |

## heartbeat flow

```
heartbeat():
  thinking_context = load thinking.md + dreams.md + concerns_and_ideas.md

  # Phase 1: Active thinking (tool-use session)
  thinking_session(system=active_thinking.md, initial_message=thinking_context)
    → append_thinking / replace_thinking / done
    → writes thinking.md directly

  # Phase 2: Subconscious evaluation
  subconscious_context = format_context([thinking.md, dreams.md])
  concerns_and_ideas = llm_generate(system=subconscious.md, content=subconscious_context)
  write concerns_and_ideas.md

  # Phase 3: Dreaming
  dreaming_context = format_context([thinking.md, concerns_and_ideas.md])
  dreams = llm_generate(system=dreaming.md, content=dreaming_context)
  write dreams.md
```

## on_message flow

```
on_message(events):
  events_text = format_events(events)

  # Phase 1: Intuition (events + dreams → raw impressions)
  visions = llm_generate(system=intuition.md, content=events + dreams)

  # Phase 2: Worry (events + concerns/ideas + visions → approach)
  approach = llm_generate(system=worry.md, content=events + concerns_and_ideas, context=visions)

  # Phase 3: Action (events + thinking + approach → final reply)
  response = llm_generate(system=action.md, content=events + thinking, context=approach)

  ctx.send(target_room, response)
```

## Config keys

No species-specific config keys. All behaviour is governed by the harness schedule and the prompt files in `prompts/`.
