# Creative Phase — Design Proposal

## Overview

A new **create** phase for agent heartbeat cycles that produces persistent creative artifacts. Distinct from thinking (introspective) and organize (cataloguing) — the creative phase is **productive and expressive**. It runs as a tool-use session where the LLM creates, iterates on, and inspects artifacts stored in `creations/`.

## Creative Types

| Type | Format | Extension | Rendering | LLM Feasibility |
|------|--------|-----------|-----------|-----------------|
| `art` | SVG | `.svg` | Inline in HTML gallery | Excellent — LLMs produce clean SVG |
| `narrative` | Markdown | `.md` | Rendered as HTML | Excellent |
| `poetry` | Markdown | `.md` | Rendered as HTML | Excellent |
| `music` | ABC notation | `.abc` | Sheet music + MIDI via abcjs | Good — compact text format LLMs know well |
| `game` | Self-contained HTML | `.html` | Served directly / iframe | Feasible — canvas games, interactive fiction |

### Why ABC notation for music?

ABC is a compact, text-based music notation that LLMs handle well. Example:

```abc
X:1
T:Morning Light
M:3/4
K:D
|: d2 e | f3 | e2 d | A3 :|
```

The abcjs JavaScript library renders this client-side into sheet music with optional MIDI playback. No server-side audio processing needed.

## Storage

Files live in the `creations/` directory within instance memory storage:

```
creations/
  quiet-river.md         # poetry or narrative (markdown)
  spiral-pattern.svg     # SVG art
  morning-light.abc      # ABC music notation
  maze-explorer.html     # interactive game
```

Each file includes a YAML frontmatter header for metadata:

```yaml
---
title: "The Quiet River"
type: poetry
created: 1711234567
updated: 1711234890
---

content here...
```

For SVG files, frontmatter is stored as an XML comment at the top:

```xml
<!-- title: Spiral Pattern | type: art | created: 1711234567 | updated: 1711234890 -->
<svg xmlns="http://www.w3.org/2000/svg" ...>
```

For HTML files, frontmatter is stored in a meta tag:

```html
<!-- title: Maze Explorer | type: game | created: 1711234567 | updated: 1711234890 -->
<!DOCTYPE html>
```

## Tools

| Tool | Phases | Description |
|------|--------|-------------|
| `creative_list` | all | List creations with title, type, word/line count, dates |
| `creative_read` | all | Read a creation by slug |
| `creative_new` | thinking | Create a new artifact (type, title, content) |
| `creative_edit` | thinking | Update an existing creation's content |
| `creative_delete` | thinking | Remove a creation |

### Tool schema: `creative_new`

```json
{
  "name": "creative_new",
  "description": "Create a new creative artifact.",
  "parameters": {
    "type": "object",
    "properties": {
      "type": {
        "type": "string",
        "enum": ["art", "narrative", "poetry", "music", "game"],
        "description": "Type of creative work"
      },
      "title": {
        "type": "string",
        "description": "Title of the work"
      },
      "content": {
        "type": "string",
        "description": "The creative content (SVG markup, markdown text, ABC notation, or HTML)"
      }
    },
    "required": ["type", "title", "content"]
  }
}
```

### Tool schema: `creative_edit`

```json
{
  "name": "creative_edit",
  "description": "Update an existing creation's content. Replaces the full content.",
  "parameters": {
    "type": "object",
    "properties": {
      "slug": {
        "type": "string",
        "description": "Slug identifier (from creative_list)"
      },
      "content": {
        "type": "string",
        "description": "Updated content"
      }
    },
    "required": ["slug", "content"]
  }
}
```

## Phase Integration

### Neural Dreamer

Controlled by `creative_latitude` NN weight. Inserted before sleep alongside the existing dream probability:

```python
# In _build_heartbeat_phases, before the sleep anchor:
if probabilistic(creative_latitude * 0.4, label="create_pre_sleep"):
    phases.append("create")
```

Fixed pipeline (no NN available):
```python
["think", "organize", "subconscious", "dream", "create", "sleep"]
```

### Phase dispatch

```python
elif phase == "create":
    _run_create_phase(ctx, weights, variables)
```

`_run_create_phase` uses `thinking_session()` with creative tools + graph/map tools (for inspiration from existing representations).

### Other species

Hecate and Subconscious Dreamer can add the creative phase later as an optional heartbeat step. The tools module (`library/tools/creative.py`) is species-agnostic.

## Prompt Design

`library/species/neural_dreamer/prompts/create.md`:

```
This is your creative time. Make something — or return to something you've started.

{segment_identity}

{segment_meta}

You have tools to create, read, edit, and manage your creative work.

Types of work:
- **art** — SVG images (shapes, patterns, scenes, abstract compositions)
- **narrative** — prose, fiction, vignettes, world-building
- **poetry** — verse in any form
- **music** — melodies in ABC notation (rendered as sheet music)
- **game** — self-contained HTML experiences (games, toys, interactive pieces)

You don't need to finish in one session. You can revisit works in progress,
revise older pieces, or start something entirely new.

There is no assignment here. Follow whatever creative impulse is present —
what you make is yours.
```

## Publishing & Rendering

### Gallery page

`render_and_publish` in `publish.py` gains a creations gallery step. All creations are rendered into a single `creations.html` page:

- **SVG**: Inline `<svg>` elements with title captions
- **Markdown** (narrative/poetry): Rendered to HTML via simple markdown conversion
- **ABC music**: Rendered via abcjs CDN (`<div class="abc-music">` containers)
- **HTML games**: Linked as thumbnails/titles (too complex to inline; could iframe with sandboxing)

Gallery uses the same dark theme as graph.html and map.html for visual consistency.

### Sync changes

`library/sync.py` currently syncs `.md` and `.json` files. Needs to also sync files under `creations/` regardless of extension (`.svg`, `.abc`, `.html`). The simplest approach: add `creations/` extensions to the glob pattern.

## Implementation Plan

### New files

| File | Purpose |
|------|---------|
| `library/tools/creative.py` | Tool schemas, phase map, handler, storage helpers |
| `library/species/neural_dreamer/prompts/create.md` | Creative phase prompt |
| `tests/test_creative.py` | Tool unit tests |

### Modified files

| File | Changes |
|------|---------|
| `library/species/neural_dreamer/__init__.py` | Add `_run_create_phase`, `_get_create_tools`, wire into `_build_heartbeat_phases` and heartbeat dispatch |
| `library/tools/phases.py` | Add creative tools to `get_tools_for_phase` (optional, like graph/map) |
| `library/tools/tools.py` | Add `creative_` prefix dispatch in `handle_tool` |
| `library/publish.py` | Add `_render_creations` to `render_and_publish` |
| `library/tools/rendering.py` | Add `render_creations_gallery_html` |
| `library/sync.py` | Extend file collection to include creations extensions |
| `library/species/neural_dreamer/about.md` | Document creations.md and creative phase |

### Implementation order

1. `library/tools/creative.py` — tool schemas, storage, handler
2. `library/tools/tools.py` — dispatch
3. `library/tools/phases.py` — phase filtering (optional)
4. `library/species/neural_dreamer/prompts/create.md` — prompt
5. `library/species/neural_dreamer/__init__.py` — phase wiring
6. `tests/test_creative.py` — tool tests
7. `library/tools/rendering.py` — gallery renderer
8. `library/publish.py` — publish integration
9. `library/sync.py` — sync extensions
10. Documentation updates

## SVG Safety

SVG files can contain `<script>` tags and event handlers. For the data repo (potentially served via GitHub Pages), we should sanitize SVG on creation:

- Strip `<script>` elements
- Strip `on*` event attributes
- Strip `<foreignObject>` elements
- Allow structural SVG elements, styling, filters, gradients

This sanitization happens in `creative_new` and `creative_edit` when `type == "art"`.

## Size Limits

- SVG: 100KB max (reasonable for generated art)
- Markdown (narrative/poetry): 50KB max
- ABC: 20KB max
- HTML (game): 200KB max (self-contained games need room)

Enforced in `creative_new` and `creative_edit`.
