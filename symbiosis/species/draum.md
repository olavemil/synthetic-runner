# Draum

Draum is a persistent memory agent. It maintains a rich internal memory across sessions and responds to messages through a structured gut → plan → compose pipeline. Between messages it runs spontaneous thinking sessions to consolidate memory and update its sense of intentions.

## Memory files

| File | Purpose |
|------|---------|
| `thinking.md` | Accumulated thoughts, updated each session |
| `project.md` | Long-term project notes and goals |
| `sessions.md` | Session log summaries |
| `scratchpad.md` | Ephemeral working space |
| `sensitivity.md` | Sensitivity guidelines and content preferences |
| `intentions.md` | Forward-looking intentions (written by react phase) |
| `subconscious.md` | Meta-evaluation of recent sessions (written post-session, read-only from main session) |
| `relationships/<sender>.md` | Per-person relationship notes |

## on_message flow

```
on_message(events):
  memory = read all memory files
  relationships = load relationship notes for senders

  gut = gut_response(events, memory, relationships)
    → LLM decides: should_respond? rooms_to_respond? gut impression?

  if not should_respond:
    run post-session processes and return

  plan = plan_response(gut, messages_by_room, room_contexts, memory)
    → LLM produces per-room reply plans

  for each room in plan:
    message = compose_response(room_plan, room_context, relationships, memory)
      → LLM writes the final reply
    ctx.send(room, message)

  post-session:
    run_subconscious(ctx, "reactive")   → evaluates the session
    run_react(ctx, "reactive")          → updates intentions.md
    update_relationships(ctx, events)   → updates relationship files
    distill_memory(ctx)                 → compresses thinking.md if too long
```

## heartbeat flow

```
heartbeat():
  memory = read all memory files
  run_session(ctx)
    → LLM does a free-form thinking session, writes thinking.md
  run_subconscious(ctx, "heartbeat")    → evaluates the session
  run_react(ctx, "heartbeat")           → updates intentions.md
  distill_memory(ctx)                   → compresses if needed
```

## Config keys

No species-specific config keys. All behaviour is governed by the harness schedule and the LLM prompts inside the toolkit.
