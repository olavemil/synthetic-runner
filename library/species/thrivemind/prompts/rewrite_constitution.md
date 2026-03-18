Current constitution:
{current_constitution}

Proposed principles:
{proposed_principles}

Refine the principles into a new constitution.
Constraints:

- Output only the final constitution text.
- Start directly with the constitution, no preface.
- Keep length similar to input (target ~{target_words} words, acceptable range {min_words}-{max_words}).
- Keep concise headings/bullets if useful, but avoid repetition.

## Governance Policies (optional)

The constitution may include a fenced ```yaml block to configure the colony's governance processes. If the current constitution already has a policies block, preserve or refine it. If proposed principles suggest process changes, add or modify the block.

Available policy options:
```yaml
on_message:
  preprocess:
    enabled: true/false
    prompt_template: "default" or custom prompt with {message}
  postprocess:
    enabled: true/false
    prompt_template: "default" or custom prompt with {candidate}
thinking:
  stages:  # 0 to 3 pre-voting stages
    - name: "stage_name"
      type: individual/collective/writer
      ordering: random/cohesion_asc/cohesion_desc/approval_asc/approval_desc/combined_asc/combined_desc
      visibility_after: private/revealed/incremental/none
      visibility_in_phase: none/incremental/revealed  # for collective stages
      prompt_template: "default" or custom prompt
  post_spawn:  # 0 to 2 post-spawn stages (same structure)
    - name: "stage_name"
      type: individual/collective/writer
```

Only include a policies block if process changes are warranted by the principles. Do not add one just because you can.
