"""Single-instance colony toolkit for Thrivemind — colony biology only."""

from __future__ import annotations

import json
import logging
import random
import re
import time
import uuid
from pathlib import Path
from urllib.parse import quote
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import yaml

from library.harness.sanitize import strip_think_blocks
from library.tools.identity import (
    AXES,
    AXIS_NAMES,
    Identity,
    format_persona,
    parse_model,
)
from library.tools.voting import approval_weights, weighted_sample
from library.tools.deliberate import generate_with_identity
from library.tools.prompts import format_events, get_entity_id

if TYPE_CHECKING:
    from library.harness.context import InstanceContext

# Backward-compat alias: colony individuals ARE Identity objects
Individual = Identity

COLONY_NAMESPACE = "colony"
logger = logging.getLogger(__name__)
_CODE_FENCE_RE = re.compile(r"(?is)```(?:[a-zA-Z0-9_-]+)?\n(.*?)```")
_WORD_RE = re.compile(r"\b\w+\b")

# Load prompts (thrivemind is in species/ but we're in tools/, so find species path)
_SPECIES_DIR = Path(__file__).parent.parent / "species" / "thrivemind"
_PROMPTS_DIR = _SPECIES_DIR / "prompts"
_PROMPT_REFLECT = (_PROMPTS_DIR / "reflect.md").read_text()
_PROMPT_SUMMARIZE = (_PROMPTS_DIR / "summarize.md").read_text()
_PROMPT_MESSAGING = (_PROMPTS_DIR / "messaging.md").read_text()
_PROMPT_CONTRIBUTE = (_PROMPTS_DIR / "contribute_constitution.md").read_text()
_PROMPT_CONTRIBUTE_RETRY = (_PROMPTS_DIR / "contribute_constitution_retry.md").read_text()
_PROMPT_REWRITE = (_PROMPTS_DIR / "rewrite_constitution.md").read_text()
_PROMPT_VOTE = (_PROMPTS_DIR / "vote_constitution.md").read_text()
_PROMPT_VOTE_PEER = (_PROMPTS_DIR / "vote_peer.md").read_text()
_PROMPT_DESCRIPTOR = (_PROMPTS_DIR / "descriptor.md").read_text()


def _word_count(text: str) -> int:
    return len(_WORD_RE.findall(text or ""))


def _strip_reasoning_blocks(text: str) -> str:
    cleaned = strip_think_blocks(text)
    # Keep fenced content when it's not explicitly a thinking block.
    cleaned = _CODE_FENCE_RE.sub(lambda m: m.group(1), cleaned)
    return cleaned


def _extract_first_sentence(text: str) -> str:
    match = re.search(r"[.!?](?:\s|$)", text)
    if match:
        return text[: match.end()].strip()
    return text.strip()


def _sanitize_contribution(text: str, max_chars: int = 220) -> str:
    cleaned = _strip_reasoning_blocks(text).replace("\r\n", "\n")
    lines = []
    for raw_line in cleaned.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line)
        if re.match(r"^(analysis|reasoning|thoughts?|thinking)\s*:", line, flags=re.IGNORECASE):
            continue
        lines.append(line)

    merged = " ".join(lines)
    merged = re.sub(r"(?i)^(?:principle|refinement|proposal)\s*:\s*", "", merged).strip()
    merged = re.sub(r"\s+", " ", merged)
    if not merged:
        return ""

    one_sentence = _extract_first_sentence(merged)
    if len(one_sentence) > max_chars:
        one_sentence = one_sentence[:max_chars].rstrip()
    return one_sentence


def _truncate_to_words(text: str, max_words: int) -> str:
    if _word_count(text) <= max_words:
        return text.strip()
    return " ".join((text or "").split()[:max_words]).strip()


def _truncate_for_prompt(text: str, max_chars: int) -> str:
    cleaned = (text or "").strip()
    if len(cleaned) <= max_chars:
        return cleaned
    clipped = cleaned[:max_chars].rsplit(" ", 1)[0].rstrip()
    if not clipped:
        clipped = cleaned[:max_chars].rstrip()
    return f"{clipped}\n...[truncated]"


def _sanitize_constitution_candidate(text: str, max_words: int) -> str:
    cleaned = _strip_reasoning_blocks(text).replace("\r\n", "\n").strip()
    if not cleaned:
        return ""

    cleaned = re.sub(
        r"(?is)^\s*(?:here(?:'s| is)\s+(?:the|a)\s+(?:rewritten|revised)\s+constitution\s*:?)\s*",
        "",
        cleaned,
    )

    lower = cleaned.lower()
    for marker in ("# constitution", "constitution:"):
        idx = lower.find(marker)
        if idx > 0 and idx <= max(240, len(cleaned) // 2):
            cleaned = cleaned[idx:].strip()
            break

    cleaned = _truncate_to_words(cleaned, max_words=max_words)
    return cleaned


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class ThrivemindConfig:
    def __init__(
        self,
        colony_size: int = 12,  # deprecated, used as fallback
        min_colony_size: int = 8,
        max_colony_size: int = 16,
        suggestion_fraction: float = 0.5,
        approval_threshold: int = 3,
        consensus_threshold: float = 0.6,
        round_timeout_s: int = 30,
        suggestion_model: str = "",
        writer_model: str = "",
        voice_space: str = "main",
    ):
        # Support legacy colony_size as fallback for min/max
        self.min_colony_size = min_colony_size
        self.max_colony_size = max_colony_size
        # If only colony_size was set (legacy), use it for both bounds
        self._legacy_colony_size = colony_size
        self.suggestion_fraction = suggestion_fraction
        self.approval_threshold = approval_threshold
        self.consensus_threshold = consensus_threshold
        self.round_timeout_s = round_timeout_s
        self.suggestion_model = suggestion_model
        self.writer_model = writer_model
        self.voice_space = voice_space

    @property
    def colony_size(self) -> int:
        """Backward-compat: return max_colony_size as the target."""
        return self.max_colony_size


# ---------------------------------------------------------------------------
# Policy dataclasses — customizable colony behavior
# ---------------------------------------------------------------------------

# Valid ordering schemes for thinking stages
ORDERING_SCHEMES = frozenset({
    "cohesion_asc", "cohesion_desc",
    "approval_asc", "approval_desc",
    "combined_asc", "combined_desc",
    "random",
})

# Valid visibility options
VISIBILITY_OPTIONS = frozenset({"private", "revealed", "incremental", "none"})

# Stage types
STAGE_TYPES = frozenset({"individual", "collective", "writer"})

# Sentinel for "use built-in prompt"
DEFAULT_PROMPT = "default"


@dataclass
class StagePolicy:
    """One stage in a multi-stage thinking pipeline."""
    name: str
    stage_type: str = "individual"  # individual | collective | writer
    prompt_template: str = DEFAULT_PROMPT
    ordering: str = "random"
    visibility_after: str = "revealed"
    visibility_in_phase: str = "none"  # for collective stages

    def __post_init__(self) -> None:
        if self.stage_type not in STAGE_TYPES:
            raise ValueError(f"Invalid stage_type {self.stage_type!r}; must be one of {sorted(STAGE_TYPES)}")
        if self.ordering not in ORDERING_SCHEMES:
            raise ValueError(f"Invalid ordering {self.ordering!r}; must be one of {sorted(ORDERING_SCHEMES)}")
        if self.visibility_after not in VISIBILITY_OPTIONS:
            raise ValueError(f"Invalid visibility_after {self.visibility_after!r}; must be one of {sorted(VISIBILITY_OPTIONS)}")
        if self.visibility_in_phase not in VISIBILITY_OPTIONS:
            raise ValueError(f"Invalid visibility_in_phase {self.visibility_in_phase!r}; must be one of {sorted(VISIBILITY_OPTIONS)}")


@dataclass
class PreprocessPolicy:
    """Controls how incoming messages are summarized before colony deliberation."""
    enabled: bool = True
    prompt_template: str = DEFAULT_PROMPT


@dataclass
class PostprocessPolicy:
    """Controls how the winning candidate is rewritten before sending."""
    enabled: bool = True
    prompt_template: str = DEFAULT_PROMPT


@dataclass
class OnMessagePolicy:
    """Policy for the on_message entry point."""
    preprocess: PreprocessPolicy = field(default_factory=PreprocessPolicy)
    postprocess: PostprocessPolicy = field(default_factory=PostprocessPolicy)


@dataclass
class ThinkingPolicy:
    """Policy for the heartbeat thinking/constitution phase."""
    stages: list[StagePolicy] = field(default_factory=list)
    post_spawn: list[StagePolicy] = field(default_factory=list)

    def __post_init__(self) -> None:
        if len(self.stages) > 3:
            raise ValueError(f"At most 3 pre-voting stages allowed, got {len(self.stages)}")
        if len(self.post_spawn) > 2:
            raise ValueError(f"At most 2 post-spawn stages allowed, got {len(self.post_spawn)}")


@dataclass
class ThrivemindPolicies:
    """Top-level policy container for a Thrivemind instance."""
    on_message: OnMessagePolicy = field(default_factory=OnMessagePolicy)
    thinking: ThinkingPolicy = field(default_factory=ThinkingPolicy)

    @classmethod
    def from_dict(cls, raw: dict) -> ThrivemindPolicies:
        """Parse policies from a raw config dict (YAML).

        Missing keys fall back to defaults (today's behavior).
        """
        if not raw or not isinstance(raw, dict):
            return cls()

        # Parse on_message policy
        om_raw = raw.get("on_message", {})
        if not isinstance(om_raw, dict):
            om_raw = {}
        pre_raw = om_raw.get("preprocess", {})
        post_raw = om_raw.get("postprocess", {})
        on_message = OnMessagePolicy(
            preprocess=PreprocessPolicy(
                enabled=bool(pre_raw.get("enabled", True)) if isinstance(pre_raw, dict) else True,
                prompt_template=str(pre_raw.get("prompt_template", DEFAULT_PROMPT)) if isinstance(pre_raw, dict) else DEFAULT_PROMPT,
            ),
            postprocess=PostprocessPolicy(
                enabled=bool(post_raw.get("enabled", True)) if isinstance(post_raw, dict) else True,
                prompt_template=str(post_raw.get("prompt_template", DEFAULT_PROMPT)) if isinstance(post_raw, dict) else DEFAULT_PROMPT,
            ),
        )

        # Parse thinking policy
        th_raw = raw.get("thinking", {})
        if not isinstance(th_raw, dict):
            th_raw = {}
        stages = _parse_stages(th_raw.get("stages", []))
        post_spawn = _parse_stages(th_raw.get("post_spawn", []))
        thinking = ThinkingPolicy(stages=stages, post_spawn=post_spawn)

        return cls(on_message=on_message, thinking=thinking)

    def describe(self) -> str:
        """Return a human-readable description of current policies."""
        lines: list[str] = ["# Active Policies\n"]

        lines.append("## on_message\n")
        pre = self.on_message.preprocess
        lines.append(f"- preprocess: {'enabled' if pre.enabled else 'disabled'}")
        if pre.enabled and pre.prompt_template != DEFAULT_PROMPT:
            lines.append(f"  prompt: {pre.prompt_template[:80]}...")
        post = self.on_message.postprocess
        lines.append(f"- postprocess: {'enabled' if post.enabled else 'disabled'}")
        if post.enabled and post.prompt_template != DEFAULT_PROMPT:
            lines.append(f"  prompt: {post.prompt_template[:80]}...")

        lines.append("\n## thinking\n")
        if self.thinking.stages:
            lines.append(f"- {len(self.thinking.stages)} pre-voting stage(s):")
            for s in self.thinking.stages:
                custom = " (custom prompt)" if s.prompt_template != DEFAULT_PROMPT else ""
                lines.append(f"  - {s.name}: type={s.stage_type}, ordering={s.ordering}, "
                             f"visibility_after={s.visibility_after}{custom}")
        else:
            lines.append("- stages: default (single contribution + synthesis)")

        if self.thinking.post_spawn:
            lines.append(f"- {len(self.thinking.post_spawn)} post-spawn stage(s):")
            for s in self.thinking.post_spawn:
                lines.append(f"  - {s.name}: type={s.stage_type}")
        else:
            lines.append("- post_spawn: none")

        return "\n".join(lines)


def _parse_stages(raw_list: list) -> list[StagePolicy]:
    """Parse a list of stage dicts into StagePolicy objects."""
    if not isinstance(raw_list, list):
        return []
    stages = []
    for item in raw_list:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", f"stage_{len(stages)}"))
        stage_type = str(item.get("type", item.get("stage_type", "individual")))
        prompt_template = str(item.get("prompt_template", DEFAULT_PROMPT))
        ordering = str(item.get("ordering", "random"))
        visibility_after = str(item.get("visibility_after", "revealed"))
        visibility_in_phase = str(item.get("visibility_in_phase", "none"))
        try:
            stages.append(StagePolicy(
                name=name,
                stage_type=stage_type,
                prompt_template=prompt_template,
                ordering=ordering,
                visibility_after=visibility_after,
                visibility_in_phase=visibility_in_phase,
            ))
        except ValueError as e:
            logger.warning("Skipping invalid stage %r: %s", name, e)
    return stages


POLICY_FILE = "policy.yaml"


def load_policies(ctx: InstanceContext, cfg: ThrivemindConfig | None = None) -> ThrivemindPolicies:
    """Load policies from instance config, overlaid with colony-adopted policy.yaml.

    Precedence: colony-adopted policy.yaml > instance config > defaults.
    """
    # 1. Base policies from instance config
    raw = ctx.config("thrivemind") or {}
    if not isinstance(raw, dict):
        raw = {}
    base_raw = raw.get("policies", {})
    if not isinstance(base_raw, dict):
        base_raw = {}

    # 2. Colony-adopted overlay from memory
    overlay_text = ctx.read(POLICY_FILE)
    overlay_raw: dict = {}
    if overlay_text:
        try:
            parsed = yaml.safe_load(overlay_text)
            if isinstance(parsed, dict):
                overlay_raw = parsed.get("policies", parsed)
                if not isinstance(overlay_raw, dict):
                    overlay_raw = {}
        except yaml.YAMLError as e:
            logger.warning("Ignoring malformed %s: %s", POLICY_FILE, e)

    # 3. Merge: overlay wins over base
    merged = _deep_merge(base_raw, overlay_raw)
    return ThrivemindPolicies.from_dict(merged)


def _deep_merge(base: dict, overlay: dict) -> dict:
    """Recursively merge overlay into base, overlay wins on conflicts."""
    result = dict(base)
    for key, value in overlay.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def describe_processes(cfg: ThrivemindConfig, policies: ThrivemindPolicies) -> str:
    """Generate a human-readable description of the colony's active processes.

    Written to processes.md so colony members can inspect their own governance.
    """
    lines: list[str] = ["# Colony Processes\n"]

    # Colony configuration
    lines.append("## Configuration\n")
    lines.append(f"- Colony size: {cfg.min_colony_size}–{cfg.max_colony_size}")
    lines.append(f"- Suggestion fraction: {cfg.suggestion_fraction:.0%}")
    lines.append(f"- Consensus threshold: {cfg.consensus_threshold:.0%}")
    lines.append(f"- Approval threshold: {cfg.approval_threshold}")
    lines.append(f"- Voice space: {cfg.voice_space}")
    if cfg.suggestion_model:
        lines.append(f"- Suggestion model: {cfg.suggestion_model}")
    if cfg.writer_model:
        lines.append(f"- Writer model: {cfg.writer_model}")

    # on_message flow
    lines.append("\n## on_message Flow\n")
    lines.append("```")
    pre = policies.on_message.preprocess
    if pre.enabled:
        if pre.prompt_template != "default":
            lines.append("1. PREPROCESS (custom) → reframe incoming messages")
        else:
            lines.append("1. PREPROCESS (default) → summarize message history")
    else:
        lines.append("1. PREPROCESS: disabled")
    lines.append("2. REFLECT → each individual reflects on colony state")
    lines.append("3. MESSAGING → internal colony messaging phase")
    lines.append("4. DELIBERATE → suggesters propose, all vote (Borda)")
    post = policies.on_message.postprocess
    if post.enabled:
        if post.prompt_template != "default":
            lines.append("5. RECOMPOSE + POSTPROCESS (custom) → refine response")
        else:
            lines.append("5. RECOMPOSE (default) → unified colony voice")
    else:
        lines.append("5. RECOMPOSE: postprocessing disabled")
    lines.append("6. SEND → deliver to voice space")
    lines.append("```")

    # Heartbeat flow
    lines.append("\n## heartbeat Flow\n")
    lines.append("```")
    lines.append("1. REFLECT → per-individual reflections (randomized order)")
    lines.append("2. MESSAGING → internal colony messaging phase")

    thinking_stages = policies.thinking.stages
    if thinking_stages:
        lines.append(f"3. THINKING ({len(thinking_stages)} stage(s)):")
        for i, s in enumerate(thinking_stages, 1):
            vis = f", visibility_after={s.visibility_after}" if s.visibility_after != "revealed" else ""
            vis_in = f", visibility_in_phase={s.visibility_in_phase}" if s.visibility_in_phase != "none" else ""
            custom = " [custom prompt]" if s.prompt_template != "default" else ""
            lines.append(f"   3.{i}. {s.name}: type={s.stage_type}, ordering={s.ordering}{vis}{vis_in}{custom}")
    else:
        lines.append("3. CONTRIBUTE → each individual proposes one principle")

    lines.append("4. REWRITE → colony writer synthesizes constitution draft")
    lines.append("5. VOTE → round 1 (+ round 2 if needed)")
    lines.append("6. PEER APPROVAL → each individual endorses 2 others")
    lines.append("7. SPAWN CYCLE → eligible individuals reproduce")

    post_spawn = policies.thinking.post_spawn
    if post_spawn:
        lines.append(f"8. POST-SPAWN ({len(post_spawn)} stage(s)):")
        for i, s in enumerate(post_spawn, 1):
            lines.append(f"   8.{i}. {s.name}: type={s.stage_type}, ordering={s.ordering}")

    lines.append("9. ORGANIZE → knowledge organization")
    lines.append("10. CREATE → artifact creation")
    lines.append("```")

    # Active policies detail
    lines.append("\n" + policies.describe())

    return "\n".join(lines)


def write_process_description(ctx: InstanceContext, cfg: ThrivemindConfig, policies: ThrivemindPolicies) -> None:
    """Write processes.md to instance memory for colony self-inspection."""
    content = describe_processes(cfg, policies)
    ctx.write("processes.md", content)


# ---------------------------------------------------------------------------
# Policy extraction & adoption from constitution text
# ---------------------------------------------------------------------------

_POLICY_FENCE_RE = re.compile(
    r"```(?:yaml|policies)\s*\n(.*?)```",
    re.DOTALL,
)

_POLICY_FIX_PROMPT = """\
The colony's constitution contained a policies block, but it failed validation:

{error}

Original block:
```yaml
{block}
```

Rewrite the policies block so it is valid YAML matching this schema:
- on_message.preprocess.enabled: bool
- on_message.preprocess.prompt_template: string ("default" or custom)
- on_message.postprocess.enabled: bool
- on_message.postprocess.prompt_template: string ("default" or custom)
- thinking.stages: list of {{name, type, prompt_template, ordering, visibility_after, visibility_in_phase}}
  - type: individual | collective | writer
  - ordering: cohesion_asc | cohesion_desc | approval_asc | approval_desc | combined_asc | combined_desc | random
  - visibility_after / visibility_in_phase: private | revealed | incremental | none
  - max 3 stages
- thinking.post_spawn: list (same schema, max 2)

Return ONLY the corrected YAML inside a ```yaml fence. No explanation.
"""


def extract_policy_block(constitution_text: str) -> str | None:
    """Extract a ```yaml or ```policies fenced block from constitution text.

    Returns the raw YAML string, or None if no block found.
    """
    match = _POLICY_FENCE_RE.search(constitution_text)
    if match:
        return match.group(1).strip()
    return None


def parse_policy_block(raw_yaml: str) -> tuple[ThrivemindPolicies | None, str]:
    """Try to parse a YAML string into ThrivemindPolicies.

    Returns (policies, "") on success, or (None, error_message) on failure.
    """
    try:
        parsed = yaml.safe_load(raw_yaml)
    except yaml.YAMLError as e:
        return None, f"YAML parse error: {e}"

    if not isinstance(parsed, dict):
        return None, f"Expected a YAML mapping, got {type(parsed).__name__}"

    # Allow either top-level keys or wrapped in "policies:"
    policies_dict = parsed.get("policies", parsed)
    if not isinstance(policies_dict, dict):
        return None, "Policies block must be a YAML mapping"

    try:
        policies = ThrivemindPolicies.from_dict(policies_dict)
    except (ValueError, TypeError) as e:
        return None, f"Validation error: {e}"

    return policies, ""


def try_adopt_policies(
    ctx: InstanceContext,
    constitution_text: str,
    cfg: ThrivemindConfig,
    max_retries: int = 3,
) -> ThrivemindPolicies | None:
    """Extract and validate a policy block from adopted constitution text.

    If the block is invalid, ask the writer to fix it up to max_retries times.
    On success, writes policy.yaml to memory and returns the new policies.
    Returns None if no policy block found or all retries exhausted.
    """
    raw_block = extract_policy_block(constitution_text)
    if raw_block is None:
        logger.info("No policy block found in constitution; skipping policy adoption")
        return None

    # First attempt
    policies, error = parse_policy_block(raw_block)
    if policies is not None:
        _write_policy_file(ctx, raw_block)
        logger.info("Policy block adopted from constitution on first attempt")
        return policies

    # Retry loop: ask writer to fix the block
    provider, model = parse_model(cfg.writer_model) if cfg.writer_model else (None, "")
    writer = Identity(name="PolicyFixer", model=model, provider=provider)
    current_block = raw_block
    current_error = error

    for attempt in range(1, max_retries + 1):
        logger.warning(
            "Policy block validation failed (attempt %d/%d): %s",
            attempt, max_retries, current_error,
        )
        fix_prompt = (
            _POLICY_FIX_PROMPT
            .replace("{error}", current_error)
            .replace("{block}", current_block)
        )
        raw_response = generate_with_identity(
            ctx, writer, fix_prompt,
            context="Fix the colony's policy YAML so it passes validation.",
            model=cfg.writer_model,
            max_tokens=2048,
        )

        # Extract the fenced block from the response
        fixed_block = extract_policy_block(f"```yaml\n{raw_response}\n```")
        if fixed_block is None:
            # Try the raw response as YAML
            fixed_block = raw_response.strip()

        policies, current_error = parse_policy_block(fixed_block)
        if policies is not None:
            _write_policy_file(ctx, fixed_block)
            logger.info("Policy block adopted from constitution after %d retry(ies)", attempt)
            return policies
        current_block = fixed_block

    logger.warning(
        "Policy adoption failed after %d retries; keeping existing policies",
        max_retries,
    )
    return None


def _write_policy_file(ctx: InstanceContext, raw_yaml: str) -> None:
    """Write validated policy YAML to the colony's memory."""
    content = f"# Colony-adopted policies (written by constitution process)\n---\npolicies:\n"
    # Re-indent the raw YAML under the policies: key
    for line in raw_yaml.splitlines():
        content += f"  {line}\n"
    ctx.write(POLICY_FILE, content)


def order_colony(colony: list[Identity], ordering: str, rng: random.Random | None = None) -> list[Identity]:
    """Return colony members sorted by the given ordering scheme."""
    rng = rng or random.Random()
    if ordering == "random":
        result = list(colony)
        rng.shuffle(result)
        return result
    if ordering == "cohesion_asc":
        return sorted(colony, key=lambda i: i.cohesion)
    if ordering == "cohesion_desc":
        return sorted(colony, key=lambda i: i.cohesion, reverse=True)
    if ordering == "approval_asc":
        return sorted(colony, key=lambda i: i.approval)
    if ordering == "approval_desc":
        return sorted(colony, key=lambda i: i.approval, reverse=True)
    if ordering == "combined_asc":
        return sorted(colony, key=lambda i: i.cohesion * (i.approval + 1))
    if ordering == "combined_desc":
        return sorted(colony, key=lambda i: i.cohesion * (i.approval + 1), reverse=True)
    # Fallback: random
    result = list(colony)
    rng.shuffle(result)
    return result


def run_thinking_stage(
    ctx: InstanceContext,
    colony: list[Identity],
    cfg: ThrivemindConfig,
    stage: StagePolicy,
    constitution: str,
    reflections: dict[str, str],
    prior_stage_outputs: dict[str, str] | None = None,
) -> dict[str, str]:
    """Execute a single thinking stage. Returns {individual_name: output} dict.

    Stage types:
      - individual: each person writes independently to their own doc
      - collective: ordered access to a shared document
      - writer: colony writer synthesizes from prior stage outputs
    """
    ordered = order_colony(colony, stage.ordering)
    outputs: dict[str, str] = {}

    # Resolve prompt template
    prompt_base = stage.prompt_template
    if prompt_base == DEFAULT_PROMPT:
        prompt_base = (
            "You are {individual_name}, a colony member.\n"
            "Constitution:\n{constitution}\n\n"
            "Propose one principle or refinement (≤18 words, one sentence)."
        )

    if stage.stage_type == "individual":
        for individual in ordered:
            prompt = (
                prompt_base
                .replace("{individual_name}", individual.name)
                .replace("{constitution}", _truncate_for_prompt(constitution, 2400))
                .replace("{personality}", format_persona(individual))
                .replace("{reflections}", _truncate_for_prompt(reflections.get(individual.name, ""), 900))
            )
            # Inject prior stage visibility
            visible_prior = _resolve_visibility(prior_stage_outputs or {}, individual.name, stage.visibility_in_phase, ordered)
            if visible_prior:
                prompt += f"\n\n## Prior contributions\n{visible_prior}"

            raw = generate_with_identity(
                ctx, individual, prompt,
                context="Colony constitution contribution phase.",
                model=cfg.suggestion_model,
                max_tokens=2048,
            )
            outputs[individual.name] = _sanitize_contribution(raw, max_chars=220)

    elif stage.stage_type == "collective":
        shared_doc = ""
        for individual in ordered:
            prompt = (
                prompt_base
                .replace("{individual_name}", individual.name)
                .replace("{constitution}", _truncate_for_prompt(constitution, 2400))
                .replace("{personality}", format_persona(individual))
                .replace("{reflections}", _truncate_for_prompt(reflections.get(individual.name, ""), 900))
                .replace("{shared_doc}", shared_doc)
                .replace("{previous_contributions}", shared_doc)
            )
            # Inject prior stage visibility
            visible_prior = _resolve_visibility(prior_stage_outputs or {}, individual.name, stage.visibility_in_phase, ordered)
            if visible_prior:
                prompt += f"\n\n## Prior stage\n{visible_prior}"

            raw = generate_with_identity(
                ctx, individual, prompt,
                context="Colony collective thinking phase.",
                model=cfg.suggestion_model,
                max_tokens=2048,
            )
            contribution = _sanitize_contribution(raw, max_chars=220)
            outputs[individual.name] = contribution
            if contribution:
                shared_doc += f"\n- {individual.name}: {contribution}"

    elif stage.stage_type == "writer":
        # Colony writer synthesizes from all prior outputs
        all_prior = prior_stage_outputs or outputs
        input_text = "\n".join(f"- {name}: {text}" for name, text in all_prior.items() if text)
        prompt = (
            prompt_base
            .replace("{contributions}", input_text)
            .replace("{constitution}", _truncate_for_prompt(constitution, 2400))
        )
        provider, model = parse_model(cfg.writer_model) if cfg.writer_model else (None, "")
        writer = Identity(name="ColonyWriter", model=model, provider=provider)
        raw = generate_with_identity(
            ctx, writer, prompt,
            context="Synthesize colony contributions into constitution.",
            model=cfg.writer_model,
            max_tokens=4096,
        )
        outputs["_synthesis"] = raw.strip()

    return outputs


def _resolve_visibility(
    outputs: dict[str, str],
    current_name: str,
    visibility: str,
    ordered: list[Identity],
) -> str:
    """Resolve what content from prior outputs is visible to current_name."""
    if not outputs or visibility == "none" or visibility == "private":
        return ""
    if visibility == "revealed":
        return "\n".join(f"- {n}: {t}" for n, t in outputs.items() if t and n != "_synthesis")
    if visibility == "incremental":
        # Show only contributions from individuals who came before in ordering
        visible = []
        for ind in ordered:
            if ind.name == current_name:
                break
            if ind.name in outputs and outputs[ind.name]:
                visible.append(f"- {ind.name}: {outputs[ind.name]}")
        return "\n".join(visible)
    return ""


def apply_preprocess(
    ctx: InstanceContext,
    events_text: str,
    policy: PreprocessPolicy,
    cfg: ThrivemindConfig,
) -> str:
    """Apply preprocessing policy to summarize/reframe incoming messages.

    If policy uses default prompt, returns the standard summarization.
    If disabled, returns the raw events text.
    """
    if not policy.enabled:
        return events_text

    if policy.prompt_template == DEFAULT_PROMPT:
        # Use the existing summarization path unchanged
        return events_text  # Caller uses summarize_message_history as before

    # Custom preprocessing: run the template through LLM
    prompt = policy.prompt_template.replace("{message}", events_text).replace("{events}", events_text)
    provider, model = parse_model(cfg.suggestion_model) if cfg.suggestion_model else (None, "")
    summarizer = Identity(name="Preprocessor", model=model, provider=provider)
    result = generate_with_identity(
        ctx, summarizer, prompt,
        context="Preprocess incoming messages for the colony.",
        model=cfg.suggestion_model,
        max_tokens=1024,
    )
    return result.strip()


def apply_postprocess(
    ctx: InstanceContext,
    candidate_text: str,
    policy: PostprocessPolicy,
    cfg: ThrivemindConfig,
    constitution: str = "",
) -> str:
    """Apply postprocessing policy to refine the winning candidate.

    If policy uses default prompt, the standard recompose path is used (caller handles).
    If disabled, returns the candidate text unchanged.
    """
    if not policy.enabled:
        return candidate_text

    if policy.prompt_template == DEFAULT_PROMPT:
        return candidate_text  # Caller uses standard recompose

    # Custom postprocessing: run the template through LLM
    prompt = (
        policy.prompt_template
        .replace("{candidate}", candidate_text)
        .replace("{constitution}", _truncate_for_prompt(constitution, 2000))
    )
    provider, model = parse_model(cfg.writer_model) if cfg.writer_model else (None, "")
    writer = Identity(name="Postprocessor", model=model, provider=provider)
    result = generate_with_identity(
        ctx, writer, prompt,
        context="Refine colony response.",
        model=cfg.writer_model,
        max_tokens=4096,
    )
    return result.strip()


# ---------------------------------------------------------------------------
# Config / colony persistence
# ---------------------------------------------------------------------------


def _coerce_float(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_int(value, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def load_config(ctx: InstanceContext) -> ThrivemindConfig:
    raw = ctx.config("thrivemind") or {}
    if not isinstance(raw, dict):
        raw = {}
    legacy_size = max(1, _coerce_int(raw.get("colony_size"), 0))
    min_size = max(1, _coerce_int(raw.get("min_colony_size"), legacy_size or 8))
    max_size = max(min_size, _coerce_int(raw.get("max_colony_size"), legacy_size or 16))
    return ThrivemindConfig(
        colony_size=legacy_size,
        min_colony_size=min_size,
        max_colony_size=max_size,
        suggestion_fraction=max(0.1, _coerce_float(raw.get("suggestion_fraction"), 0.5)),
        approval_threshold=_coerce_int(raw.get("approval_threshold"), 5),
        consensus_threshold=max(0.0, _coerce_float(raw.get("consensus_threshold"), 0.6)),
        round_timeout_s=_coerce_int(raw.get("round_timeout_s"), 30),
        suggestion_model=str(raw.get("suggestion_model", "")),
        writer_model=str(raw.get("writer_model", "")),
        voice_space=str(raw.get("voice_space", "main")),
    )


def _identity_from_dict(d: dict) -> Identity:
    parents_raw = d.get("parents", [])
    return Identity(
        name=str(d.get("name") or d.get("id", "")),
        dims={k: float(v) for k, v in d.get("dims", {}).items()},
        approval=int(d.get("approval", 0)),
        cohesion=float(d.get("cohesion", 0.0)),
        created_at=int(d.get("created_at", 0)),
        age=int(d.get("age", 0)),
        parents=list(parents_raw) if isinstance(parents_raw, list) else [],
    )


def _identity_to_dict(ind: Identity) -> dict:
    return {
        "name": ind.name,
        "dims": ind.dims or {},
        "approval": ind.approval,
        "cohesion": ind.cohesion,
        "created_at": ind.created_at,
        "age": ind.age,
        "parents": ind.parents,
    }


def load_colony(ctx: InstanceContext) -> list[Identity]:
    store = ctx.store(COLONY_NAMESPACE)
    result = []
    for _key, value in store.scan():
        if isinstance(value, dict) and ("name" in value or "id" in value):
            result.append(_identity_from_dict(value))
    result.sort(key=lambda i: i.created_at)
    return result


def save_colony(ctx: InstanceContext, colony: list[Identity]) -> None:
    store = ctx.store(COLONY_NAMESPACE)
    for key, _ in store.scan():
        store.delete(key)
    for ind in colony:
        store.put(ind.name, _identity_to_dict(ind))


def _md_escape(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def _spawn_score(ind: Identity) -> float:
    """Combined spawn score: cohesion + approval."""
    return ind.cohesion + ind.approval


def build_colony_snapshot(
    colony: list[Identity],
    events: list[dict] | None = None,
    descriptors: dict[str, str] | None = None,
) -> str:
    """Render a human-readable colony snapshot markdown document."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    lines = [
        "# Colony",
        "",
        f"Generated: {timestamp}",
        f"Individuals: {len(colony)}",
        "",
        "| Individual | Descriptor | Personality | Cohesion | Approval | Age |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]

    if not colony:
        lines.append("| - | - | - | 0 | 0 | 0 |")
    else:
        for ind in sorted(colony, key=lambda i: (-_spawn_score(i), i.created_at, i.name)):
            personality = _md_escape(format_persona(ind))
            descriptor = _md_escape((descriptors or {}).get(ind.name, ""))
            lines.append(
                f"| `{ind.name}` | {descriptor} | {personality} | {ind.cohesion:.1f} | {ind.approval} | {ind.age} |"
            )

    events_section = format_events_for_context(events or [])
    if events_section:
        lines.append("")
        lines.append(events_section)

    return "\n".join(lines) + "\n"


def save_colony_snapshot(ctx: InstanceContext, colony: list[Identity]) -> None:
    """Write colony.md snapshot to instance memory storage."""
    events = load_events(ctx)
    descriptors = {ind.name: load_descriptor(ctx, ind) for ind in colony}
    content = build_colony_snapshot(colony, events=events, descriptors=descriptors)
    logger.info(
        "Writing thrivemind colony.md (individuals=%d, events=%d, chars=%d)",
        len(colony),
        len(events),
        len(content),
    )
    ctx.write("colony.md", content)


def _reflection_path(individual: Identity | str) -> str:
    name = individual.name if isinstance(individual, Identity) else str(individual)
    return f"reflections/{quote(name, safe='-_.,')}.md"


def load_reflection(ctx: InstanceContext, individual: Identity) -> str:
    return ctx.read(_reflection_path(individual))


def save_reflection(ctx: InstanceContext, individual: Identity, text: str) -> None:
    path = _reflection_path(individual)
    logger.info(
        "Writing thrivemind %s (individual=%s, chars=%d)",
        path,
        individual.name,
        len(text or ""),
    )
    ctx.write(path, text)


def load_ancestor_reflections(ctx: InstanceContext, individual: Identity) -> str:
    """Load the last known thoughts of an individual's direct ancestors.

    Checks live reflections first, then archived files for removed ancestors.
    Returns a formatted string for inclusion in the reflection prompt, or empty string.
    """
    if not individual.parents:
        return ""
    parts = []
    for parent_name in individual.parents:
        # Live reflection
        reflection = ctx.read(_reflection_path(parent_name))
        if not reflection:
            # Archived (parent was removed from colony)
            reflection = ctx.read(f"removed/{quote(parent_name, safe='-_.,')}.md")
        if reflection and reflection.strip():
            trimmed = reflection.strip()[:600]
            parts.append(f"### {parent_name}\n{trimmed}")
    if not parts:
        return ""
    return "## Ancestral Thoughts\n\n" + "\n\n".join(parts)


def summarize_message_history(
    ctx: InstanceContext,
    events: list,
    cfg: ThrivemindConfig,
) -> str:
    """Summarize current message history once for a shared colony context."""
    if not events:
        return ""

    provider, model = parse_model(cfg.suggestion_model) if cfg.suggestion_model else (None, "")
    summarizer = Identity(
        name="ColonySummarizer",
        model=model,
        provider=provider,
        personality="Concise colony historian focused on decision-relevant context.",
    )
    prompt = _PROMPT_SUMMARIZE.replace(
        "{events}", format_events(events, self_entity_id=get_entity_id(ctx))
    )
    summary = generate_with_identity(ctx, summarizer, prompt, model=cfg.suggestion_model).strip()
    logger.info("Generated thrivemind message summary (chars=%d)", len(summary))
    return summary


_REFLECTION_CONTEXT_LIMITS = {
    "colony_md": 1000,
    "constitution": 2000,
    "prior_reflection": 800,
    "message_summary": 600,
}


def build_reflection_context(
    colony_md: str,
    constitution: str,
    prior_reflection: str,
    message_summary: str,
    events_text: str = "",
    inbox_text: str = "",
) -> str:
    parts = []
    if colony_md:
        cm = (colony_md or "").strip()
        max_cm = _REFLECTION_CONTEXT_LIMITS["colony_md"]
        if len(cm) > max_cm:
            cm = cm[:max_cm] + "\n...[truncated]"
        parts.append(f"## Colony\n{cm}")
    if constitution:
        cs = (constitution or "").strip()
        max_cs = _REFLECTION_CONTEXT_LIMITS["constitution"]
        if len(cs) > max_cs:
            cs = cs[:max_cs] + "\n...[truncated]"
        parts.append(f"## Constitution\n{cs}")
    if events_text:
        parts.append(events_text)
    if inbox_text:
        parts.append(f"## Messages Received\n{inbox_text}")
    if prior_reflection:
        pr = (prior_reflection or "").strip()
        max_pr = _REFLECTION_CONTEXT_LIMITS["prior_reflection"]
        if len(pr) > max_pr:
            pr = pr[-max_pr:]  # tail: keep most recent content
        parts.append(f"## Prior Reflection\n{pr}")
    if message_summary:
        ms = (message_summary or "").strip()
        max_ms = _REFLECTION_CONTEXT_LIMITS["message_summary"]
        if len(ms) > max_ms:
            ms = ms[:max_ms] + "\n...[truncated]"
        parts.append(f"## Message Summary\n{ms}")
    return "\n\n".join(parts)


def increment_ages(colony: list[Identity]) -> None:
    """Increment age counter for all colony members (call once per thinking session)."""
    for ind in colony:
        ind.age += 1


def _relative_age_label(individual: Identity, colony: list[Identity]) -> str:
    """Return a relative age label for context injection."""
    if not colony:
        return ""
    avg_age = sum(ind.age for ind in colony) / len(colony)
    if avg_age == 0:
        return "newborn"
    ratio = individual.age / avg_age
    if ratio < 0.3:
        return "very young (much younger than average)"
    if ratio < 0.7:
        return "young (younger than average)"
    if ratio < 1.3:
        return "average age"
    if ratio < 2.0:
        return "experienced (older than average)"
    return "elder (much older than average)"


def reflect_on_colony(
    ctx: InstanceContext,
    cfg: ThrivemindConfig,
    individual: Identity,
    colony: list[Identity],
    constitution: str,
    prior_reflection: str,
    message_summary: str,
    events_text: str = "",
) -> str:
    colony_md = ctx.read("colony.md") or build_colony_snapshot(colony)
    inbox_text = read_inbox(ctx, individual)
    context = build_reflection_context(
        colony_md=colony_md,
        constitution=constitution,
        prior_reflection=prior_reflection,
        message_summary=message_summary,
        events_text=events_text,
        inbox_text=inbox_text,
    )
    age_label = _relative_age_label(individual, colony)
    age_block = f"\nYou are {age_label} in the colony (age={individual.age}, colony average={sum(i.age for i in colony)/len(colony):.1f}).\n" if colony else ""
    ancestry_block = ""
    if individual.parents:
        ancestry_text = load_ancestor_reflections(ctx, individual)
        parents_str = " and ".join(individual.parents)
        ancestry_block = f"\nYou are the offspring of {parents_str}.\n"
        if ancestry_text:
            ancestry_block += f"\n{ancestry_text}\n"
    prompt = _PROMPT_REFLECT.replace("{age_block}", age_block).replace("{ancestry_block}", ancestry_block)
    reflection = generate_with_identity(
        ctx,
        individual,
        prompt,
        context=context,
        model=cfg.suggestion_model,
    ).strip()
    logger.info(
        "Generated thrivemind reflection (individual=%s, chars=%d)",
        individual.name,
        len(reflection),
    )
    return reflection


# ---------------------------------------------------------------------------
# Colony events
# ---------------------------------------------------------------------------

_EVENTS_FILE = "colony_events.json"


def load_events(ctx: InstanceContext) -> list[dict]:
    """Load pending colony events."""
    raw = ctx.read(_EVENTS_FILE)
    if not raw:
        return []
    try:
        events = json.loads(raw)
        return events if isinstance(events, list) else []
    except (json.JSONDecodeError, TypeError):
        return []


_EVENTS_MAX = 100


def _append_event(ctx: InstanceContext, event: dict) -> None:
    """Append a single event to the event log, pruning to keep at most _EVENTS_MAX entries."""
    events = load_events(ctx)
    events.append(event)
    if len(events) > _EVENTS_MAX:
        events = events[-_EVENTS_MAX:]
    ctx.write(_EVENTS_FILE, json.dumps(events, ensure_ascii=False))


def clear_events(ctx: InstanceContext) -> None:
    """Clear the event log (call after events have been consumed)."""
    ctx.write(_EVENTS_FILE, "[]")


def record_reply_event(
    ctx: InstanceContext,
    winner_name: str,
    candidate_count: int,
    has_consensus: bool,
    approval_ratio: float,
) -> None:
    _append_event(ctx, {
        "type": "reply",
        "winner": winner_name,
        "candidates": candidate_count,
        "consensus": has_consensus,
        "approval_ratio": round(approval_ratio, 3),
        "time": int(time.time()),
    })


def record_constitution_event(
    ctx: InstanceContext,
    accepted: bool,
    acceptance_ratio: float,
    rounds: int,
) -> None:
    _append_event(ctx, {
        "type": "constitution_vote",
        "accepted": accepted,
        "ratio": round(acceptance_ratio, 3),
        "rounds": rounds,
        "time": int(time.time()),
    })


def record_spawn_event(
    ctx: InstanceContext,
    removed: list[str],
    spawned: int,
    colony_size: int,
) -> None:
    _append_event(ctx, {
        "type": "spawn",
        "removed": removed,
        "spawned": spawned,
        "colony_size": colony_size,
        "time": int(time.time()),
    })


def format_events_for_context(events: list[dict]) -> str:
    """Format colony events as human-readable text for injection into prompts."""
    if not events:
        return ""
    lines = ["## Recent Colony Events"]
    for evt in events:
        etype = evt.get("type", "unknown")
        if etype == "reply":
            consensus = "with consensus" if evt.get("consensus") else "without consensus"
            lines.append(
                f"- Reply sent ({consensus}, winner: {evt.get('winner', '?')}, "
                f"{evt.get('candidates', '?')} candidates, "
                f"{evt.get('approval_ratio', 0)*100:.0f}% approval)"
            )
        elif etype == "constitution_vote":
            result = "accepted" if evt.get("accepted") else "rejected"
            lines.append(
                f"- Constitution {result} "
                f"({evt.get('ratio', 0)*100:.0f}% approval, "
                f"{evt.get('rounds', 1)} round(s))"
            )
        elif etype == "spawn":
            removed = evt.get("removed", [])
            if removed:
                lines.append(
                    f"- Spawn cycle: {len(removed)} removed ({', '.join(removed)}), "
                    f"{evt.get('spawned', 0)} new, colony now {evt.get('colony_size', '?')}"
                )
            else:
                lines.append(
                    f"- Spawn cycle: {evt.get('spawned', 0)} new members added, "
                    f"colony now {evt.get('colony_size', '?')}"
                )
        elif etype == "message":
            lines.append(
                f"- Message from {evt.get('from', '?')} to {evt.get('to', '?')}"
            )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Individual inboxes
# ---------------------------------------------------------------------------


def _inbox_path(individual: Identity | str) -> str:
    name = individual.name if isinstance(individual, Identity) else str(individual)
    return f"inbox/{quote(name, safe='-_.,')}.md"


def read_inbox(ctx: InstanceContext, individual: Identity) -> str:
    """Read and return inbox contents for an individual."""
    return ctx.read(_inbox_path(individual)) or ""


def clear_inbox(ctx: InstanceContext, individual: Identity) -> None:
    """Clear an individual's inbox after messages have been consumed."""
    ctx.write(_inbox_path(individual), "")


def _descriptor_path(individual: "Identity | str") -> str:
    name = individual.name if isinstance(individual, Identity) else str(individual)
    return f"descriptors/{quote(name, safe='-_.,')}.md"


def load_descriptor(ctx: "InstanceContext", individual: "Identity | str") -> str:
    """Load the current self-descriptor for an individual, or empty string."""
    return ctx.read(_descriptor_path(individual)) or ""


def save_descriptor(ctx: "InstanceContext", individual: "Identity | str", text: str) -> None:
    ctx.write(_descriptor_path(individual), text)


def generate_descriptor(
    ctx: "InstanceContext",
    individual: Identity,
    cfg: "ThrivemindConfig",
    reflection: str = "",
    inbox_text: str = "",
) -> str:
    """Generate a one-liner self-descriptor from the individual's current state."""
    prev_descriptor = load_descriptor(ctx, individual)
    persona = format_persona(individual)
    context_parts = [f"Your personality: {persona}"]
    if reflection.strip():
        context_parts.append(f"Your recent thinking:\n{reflection.strip()[:600]}")
    if inbox_text.strip():
        context_parts.append(f"Messages you received:\n{inbox_text.strip()[:400]}")
    if prev_descriptor.strip():
        context_parts.append(f"Your previous descriptor: {prev_descriptor.strip()}")
    context = "\n\n".join(context_parts)
    prompt = _PROMPT_DESCRIPTOR.replace("{individual_name}", individual.name)
    raw = generate_with_identity(
        ctx, individual, prompt,
        context=context,
        model=cfg.suggestion_model,
        max_tokens=128,
    ).strip()
    # Take first line only, cap at 200 chars
    descriptor = raw.splitlines()[0].strip()[:200] if raw else ""
    if descriptor:
        save_descriptor(ctx, individual, descriptor)
    logger.info("Generated descriptor for %s (%d chars)", individual.name, len(descriptor))
    return descriptor


def update_heritage(ctx: "InstanceContext", name: str, descriptor: str, cause: str) -> None:
    """Append a removed individual's record to heritage.md."""
    import time as _time
    timestamp = _time.strftime("%Y-%m-%d %H:%M UTC", _time.gmtime())
    existing = ctx.read("heritage.md") or "# Heritage\n\nThis file records the lives of those who have left the colony.\n"
    entry = f"\n## {name}\n\n**Departed:** {timestamp}  \n**Cause:** {cause}  \n**Descriptor:** {descriptor or '(none)'}\n"
    ctx.write("heritage.md", existing.rstrip() + entry)
    logger.info("Updated heritage.md for removed individual %s (cause=%s)", name, cause)


def archive_removed_individual(ctx: "InstanceContext", name: str, cause: str = "removed") -> None:
    """Archive a removed colony individual's final reflection and clean up their files.

    Copies the individual's last reflection (updated during the heartbeat
    reflection phase) to removed/{name}.md for preservation in the synced data
    repo. Clears the original reflection file. The inbox is already cleared
    during the heartbeat reflection phase before the spawn cycle runs.
    Appends a record (with self-descriptor and cause) to heritage.md.
    """
    reflection = (ctx.read(_reflection_path(name)) or "").strip()
    archived = reflection if reflection else "(no data)"

    ctx.write(f"removed/{quote(name, safe='-_.,')}.md", archived)
    ctx.write(_reflection_path(name), "")

    descriptor = load_descriptor(ctx, name)
    ctx.write(_descriptor_path(name), "")

    update_heritage(ctx, name, descriptor, cause)
    logger.info("Archived removed individual %s to removed/%s.md (cause=%s)", name, name, cause)


def deliver_message(
    ctx: InstanceContext,
    sender: Identity,
    recipient_name: str,
    message: str,
) -> None:
    """Deliver a short message to another individual's inbox."""
    path = _inbox_path(recipient_name)
    existing = ctx.read(path) or ""
    entry = f"**{sender.name}**: {message}\n"
    ctx.write(path, existing + entry)
    logger.info(
        "Delivered colony message from %s to %s (%d chars)",
        sender.name, recipient_name, len(message),
    )


def run_messaging_phase(
    ctx: InstanceContext,
    colony: list[Identity],
    cfg: ThrivemindConfig,
) -> int:
    """Give each individual a chance to send one message to another.

    Returns the number of messages sent.
    """
    colony_names = [ind.name for ind in colony]
    sent = 0

    for individual in colony:
        others = [n for n in colony_names if n != individual.name]
        if not others:
            continue

        persona = format_persona(individual)
        inbox = read_inbox(ctx, individual)
        inbox_block = f"\nMessages in your inbox:\n{inbox}\n" if inbox.strip() else ""
        prompt = (
            _PROMPT_MESSAGING
            .replace("{individual_name}", individual.name)
            .replace("{colony_size}", str(len(colony)))
            .replace("{personality}", persona)
            .replace("{inbox_block}", inbox_block)
            .replace("{others}", ", ".join(others))
        )
        raw = generate_with_identity(
            ctx, individual, prompt,
            context="Colony internal messaging phase. Keep messages brief and meaningful.",
            model=cfg.suggestion_model,
            max_tokens=256,
        )

        try:
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0]
            data = json.loads(cleaned)
            target = data.get("to")
            msg = str(data.get("message", "")).strip()
            if target and msg and target in colony_names and target != individual.name:
                deliver_message(ctx, individual, target, msg[:200])
                sent += 1
        except Exception:
            pass

    logger.info("Colony messaging phase: %d messages sent", sent)
    return sent


_NAME_ADJECTIVES = [
    "amber", "ashen", "bright", "calm", "cold", "crimson", "dark", "deep",
    "dim", "dusk", "fading", "fierce", "frozen", "gentle", "gilded", "grey",
    "hollow", "iron", "keen", "last", "lone", "lucid", "muted", "pale",
    "quiet", "restless", "risen", "sharp", "silent", "silver", "slow",
    "soft", "stark", "still", "swift", "thin", "true", "vast", "warm", "wild",
]
_NAME_NOUNS = [
    "ash", "bell", "bone", "brook", "cairn", "clay", "coal", "crow",
    "dawn", "dust", "elm", "ember", "fern", "flame", "fox", "frost",
    "gate", "glass", "grove", "hawk", "haze", "hedge", "lark", "leaf",
    "marsh", "moss", "moth", "oak", "pearl", "pine", "reed", "root",
    "sage", "salt", "shade", "shard", "smoke", "spark", "spine", "stone",
    "storm", "thorn", "tide", "vale", "vine", "wren",
]


def _generate_name(rng: random.Random, existing_names: set[str]) -> str:
    """Generate a unique adjective-adjective-noun name."""
    for _ in range(50):
        a1 = rng.choice(_NAME_ADJECTIVES)
        a2 = rng.choice(_NAME_ADJECTIVES)
        while a2 == a1:
            a2 = rng.choice(_NAME_ADJECTIVES)
        noun = rng.choice(_NAME_NOUNS)
        name = f"{a1}-{a2}-{noun}"
        if name not in existing_names:
            return name
    # Fallback: append UUID fragment for uniqueness
    return f"{rng.choice(_NAME_ADJECTIVES)}-{rng.choice(_NAME_ADJECTIVES)}-{uuid.uuid4().hex[:6]}"


def spawn_initial_colony(
    cfg: ThrivemindConfig, rng: random.Random | None = None
) -> list[Identity]:
    rng = rng or random.Random()
    colony = []
    now = int(time.time())
    existing_names: set[str] = set()
    target = cfg.min_colony_size
    for _ in range(target):
        dims = {name: round(rng.uniform(-1.0, 1.0), 4) for name in AXIS_NAMES}
        ind_name = _generate_name(rng, existing_names)
        existing_names.add(ind_name)
        colony.append(Identity(name=ind_name, dims=dims, approval=0, cohesion=0.0, created_at=now))
    return colony


# ---------------------------------------------------------------------------
# Colony selection
# ---------------------------------------------------------------------------


def select_suggesters(
    colony: list[Identity],
    n: int,
    rng: random.Random | None = None,
) -> list[Identity]:
    """Select n individuals weighted by approval (approval+1, min 0)."""
    rng = rng or random.Random()
    weights = approval_weights(colony)
    return weighted_sample(colony, weights, n, rng)


# ---------------------------------------------------------------------------
# Constitution management
# ---------------------------------------------------------------------------


def contribute_constitution_line(
    ctx: InstanceContext,
    individual: Identity,
    current_constitution: str,
    cfg: ThrivemindConfig,
    reflections: str = "",
) -> str:
    persona = format_persona(individual)
    constitution_excerpt = _truncate_for_prompt(current_constitution, max_chars=2400)
    reflections_excerpt = _truncate_for_prompt(reflections, max_chars=900)
    reflections_block = f"\n\nReflections:\n{reflections_excerpt}" if reflections_excerpt else ""
    prompt = (
        _PROMPT_CONTRIBUTE
        .replace("{individual_name}", individual.name)
        .replace("{personality}", persona)
        .replace("{constitution}", constitution_excerpt)
        .replace("{reflections_block}", reflections_block)
    )
    raw = generate_with_identity(
        ctx,
        individual,
        prompt,
        context=(
            "You are contributing one principle or value to your colony's shared constitution.\n"
            "Keep it concise and practical.\n"
            "Your first character should be the principle sentence."
        ),
        model=cfg.suggestion_model,
        max_tokens=2048,
    )
    line = _sanitize_contribution(raw, max_chars=220)
    if not line:
        retry_prompt = (
            _PROMPT_CONTRIBUTE_RETRY
            .replace("{individual_name}", individual.name)
            .replace("{personality}", persona)
        )
        retry_raw = generate_with_identity(
            ctx,
            individual,
            retry_prompt,
            context="Final-answer mode.",
            model=cfg.suggestion_model,
            max_tokens=2048,
        )
        retry_line = _sanitize_contribution(retry_raw, max_chars=220)
        if retry_line:
            logger.info(
                "Recovered constitution contribution after empty sanitized output (individual=%s, retry_chars=%d)",
                individual.name,
                len(retry_line),
            )
            line = retry_line
    if line != (raw or "").strip():
        logger.info(
            "Sanitized constitution contribution output (individual=%s, raw_chars=%d, final_chars=%d)",
            individual.name,
            len(raw or ""),
            len(line),
        )
    return line


def rewrite_constitution(
    ctx: InstanceContext,
    lines: list[str],
    current_constitution: str,
    cfg: ThrivemindConfig,
) -> str:
    combined = "\n".join(f"- {line}" for line in lines if line.strip())
    constitution_excerpt = _truncate_for_prompt(current_constitution, max_chars=5000)
    combined_excerpt = _truncate_for_prompt(combined, max_chars=2600)
    input_word_count = _word_count(current_constitution) + _word_count(combined)
    target_words = max(20, min(500, input_word_count))
    min_words = max(12, int(target_words * 0.7))
    max_words = max(min_words + 8, int(target_words * 1.3))
    rewrite_max_tokens = max(2048, min(8192, max_words * 8))

    # Use a writer identity with the writer model
    provider, model = parse_model(cfg.writer_model) if cfg.writer_model else (None, "")
    writer = Identity(
        name="ColonyWriter",
        model=model,
        provider=provider,
        personality="You are a scribe immortalizing your colony's principles into a concise constitution.",
    )
    prompt = (
        _PROMPT_REWRITE
        .replace("{current_constitution}", constitution_excerpt)
        .replace("{proposed_principles}", combined_excerpt)
        .replace("{target_words}", str(target_words))
        .replace("{min_words}", str(min_words))
        .replace("{max_words}", str(max_words))
    )
    raw = generate_with_identity(
        ctx,
        writer,
        prompt,
        model=cfg.writer_model,
        max_tokens=rewrite_max_tokens,
    )
    rewritten = _sanitize_constitution_candidate(raw, max_words=max_words)
    if not rewritten:
        rewritten = combined
    if rewritten != (raw or "").strip():
        logger.info(
            "Sanitized constitution candidate output (raw_chars=%d, final_chars=%d)",
            len(raw or ""),
            len(rewritten),
        )
    if rewritten:
        return rewritten
    fallback = current_constitution.strip() or "# Constitution\n"
    logger.warning("Constitution rewrite became empty after sanitization; falling back to current constitution")
    return fallback


def vote_constitution(
    ctx: InstanceContext,
    individual: Identity,
    current: str,
    proposed: str,
    cfg: ThrivemindConfig,
    reflections: str = "",
    round_context: str = "",
) -> bool:
    persona = format_persona(individual)
    current_excerpt = _truncate_for_prompt(current, max_chars=3200)
    proposed_excerpt = _truncate_for_prompt(proposed, max_chars=3200)
    reflections_excerpt = _truncate_for_prompt(reflections, max_chars=1000)
    reflections_block = f"\n\nReflections:\n{reflections_excerpt}" if reflections_excerpt else ""
    round_context_block = ""
    if round_context.strip():
        round_context_excerpt = _truncate_for_prompt(round_context.strip(), max_chars=1400)
        round_context_block = f"\n\nAdditional context:\n{round_context_excerpt}"
    
    prompt = (
        _PROMPT_VOTE
        .replace("{individual_name}", individual.name)
        .replace("{personality}", persona)
        .replace("{current_constitution}", current_excerpt)
        .replace("{proposed_constitution}", proposed_excerpt)
        .replace("{reflections_block}", reflections_block)
        .replace("{round_context_block}", round_context_block)
    )
    raw = generate_with_identity(
        ctx,
        individual,
        prompt,
        context="You are voting on whether to adopt a new constitution for your colony.",
        model=cfg.suggestion_model,
    )
    try:
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
        data = json.loads(raw)
        return bool(data.get("accept", False))
    except Exception:
        return False


def save_constitution(ctx: InstanceContext, text: str) -> None:
    logger.info(
        "Writing thrivemind constitution.md (chars=%d)",
        len(text or ""),
    )
    ctx.write("constitution.md", text)


def build_contributions_snapshot(lines: list[str]) -> str:
    """Render the raw constitution contributions for visibility."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    out = [
        "# Contributions",
        "",
        f"Generated: {timestamp}",
        f"Count: {len(lines)}",
        "",
    ]
    if not lines:
        out.append("(no contributions)")
    else:
        for idx, line in enumerate(lines, start=1):
            out.append(f"{idx}. {line}")
    out.append("")
    return "\n".join(out)


def save_contributions(ctx: InstanceContext, lines: list[str]) -> None:
    content = build_contributions_snapshot(lines)
    logger.info(
        "Writing thrivemind contributions.md (count=%d, chars=%d)",
        len(lines),
        len(content),
    )
    ctx.write("contributions.md", content)


def save_candidate(ctx: InstanceContext, candidate: str) -> None:
    logger.info("Writing thrivemind candidate.md (chars=%d)", len(candidate or ""))
    ctx.write("candidate.md", candidate)


def load_constitution(ctx: InstanceContext) -> str:
    existing = ctx.read("constitution.md")
    if existing:
        return existing
    logger.info("No thrivemind constitution.md found; using default scaffold")
    return "# Constitution\n"


# ---------------------------------------------------------------------------
# Spawn cycle
# ---------------------------------------------------------------------------


def _make_offspring(
    primary: Identity,
    other: Identity,
    rng: random.Random,
    existing_names: set[str],
    now: int,
) -> Identity:
    """Create a single offspring from primary × other with dim inheritance."""
    offspring_dims: dict[str, float] = {}
    for name in AXIS_NAMES:
        roll = rng.random()
        if roll < 0.15:
            offspring_dims[name] = round(rng.uniform(-1.0, 1.0), 4)
        elif roll < 0.40:
            offspring_dims[name] = other.dims.get(name, 0.0) if other.dims else 0.0
        else:
            offspring_dims[name] = primary.dims.get(name, 0.0) if primary.dims else 0.0
    child_name = _generate_name(rng, existing_names)
    existing_names.add(child_name)
    return Identity(name=child_name, dims=offspring_dims, approval=0, cohesion=0.0, created_at=now,
                    parents=[primary.name, other.name])


def run_spawn_cycle(
    colony: list[Identity],
    cfg: ThrivemindConfig,
    rng: random.Random | None = None,
) -> tuple[list[Identity], dict[str, str]]:
    """Eligible individuals spawn offspring; colony stays within size bounds.

    - If colony is below max, eligible members spawn offspring (parents stay).
    - If colony is at max, eligible members are replaced by offspring.
    - Colony is trimmed to max_colony_size if over.

    Returns (new_colony, removal_causes) where removal_causes maps removed
    individual name → cause string ("spawned offspring" or "eliminated").
    """
    rng = rng or random.Random()
    if not colony:
        return colony, {}

    eligible = [ind for ind in colony if _spawn_score(ind) >= cfg.approval_threshold]
    if not eligible:
        return colony, {}

    total_score = sum(max(0.0, _spawn_score(ind)) for ind in eligible)
    if total_score == 0:
        return colony, {}

    existing_names = {ind.name for ind in colony}
    new_individuals: list[Identity] = []
    now = int(time.time())
    room = cfg.max_colony_size - len(colony)
    spawned_parents: set[str] = set()

    if room > 0:
        # Under max: parents survive, just add offspring
        for primary in eligible:
            if len(new_individuals) >= room:
                break
            weights = [max(0.0, _spawn_score(ind)) / total_score for ind in eligible]
            (other,) = rng.choices(eligible, weights=weights, k=1)
            new_individuals.append(_make_offspring(primary, other, rng, existing_names, now))

        new_colony = colony + new_individuals
    else:
        # At or over max: replace eligible parents with offspring
        to_remove_names: set[str] = set()
        for primary in eligible:
            weights = [max(0.0, _spawn_score(ind)) / total_score for ind in eligible]
            (other,) = rng.choices(eligible, weights=weights, k=1)
            new_individuals.append(_make_offspring(primary, other, rng, existing_names, now))
            to_remove_names.add(primary.name)
            spawned_parents.add(primary.name)

        remaining = [ind for ind in colony if ind.name not in to_remove_names]
        new_colony = remaining + new_individuals

    original_names = {ind.name for ind in colony}
    # Trim to max (lowest combined score removed first)
    eliminated: set[str] = set()
    if len(new_colony) > cfg.max_colony_size:
        new_colony.sort(key=_spawn_score)
        trimmed = new_colony[:len(new_colony) - cfg.max_colony_size]
        eliminated = {ind.name for ind in trimmed if ind.name in original_names}
        new_colony = new_colony[len(new_colony) - cfg.max_colony_size:]

    # Pad to min
    while len(new_colony) < cfg.min_colony_size:
        primary = rng.choice(eligible)
        other = rng.choice(eligible)
        new_colony.append(_make_offspring(primary, other, rng, existing_names, now))

    # Apply decay to cohesion and approval
    for ind in new_colony:
        ind.cohesion *= 0.9
        ind.approval *= 0.9

    # Build removal causes map for individuals that were in original colony but not in new
    new_names = {ind.name for ind in new_colony}
    removal_causes: dict[str, str] = {}
    for name in original_names:
        if name not in new_names:
            if name in spawned_parents:
                removal_causes[name] = "spawned offspring"
            elif name in eliminated:
                removal_causes[name] = "eliminated"
            else:
                removal_causes[name] = "removed"

    return new_colony, removal_causes


# ---------------------------------------------------------------------------
# Approval updates
# ---------------------------------------------------------------------------


def update_cohesion(
    colony: list[Identity],
    votes: dict[str, list[str]],
    winner_ids: str | list[str],
    cfg: ThrivemindConfig,  # noqa: ARG001 kept for API consistency
) -> list[Identity]:
    """Update cohesion scores based on message deliberation vote results.

    Cohesion measures how closely aligned an individual is with the group
    consensus. Winners receive cohesion bonuses:
      - Single winner: +10.0 cohesion
      - Combined consensus (two winners): +5.0 each
    Voters who backed a winner: +1.0 cohesion.
    Voters who did not back any winner: -1.0 cohesion.
    """
    id_map = {ind.name: ind for ind in colony}
    
    # Normalize winner_ids to a list
    if isinstance(winner_ids, str):
        winner_list = [winner_ids] if winner_ids else []
    else:
        winner_list = list(winner_ids)
    winner_set = set(winner_list)
    
    logger.info(
        "update_cohesion: winners=%s, voters=%d, colony_size=%d",
        winner_list, len(votes), len(colony),
    )

    # Apply winner bonuses: +10.0 for single winner, +5.0 each for combined
    bonus = 10.0 if len(winner_list) == 1 else 5.0
    for winner_id in winner_list:
        if winner_id in id_map:
            id_map[winner_id].cohesion += bonus
            logger.debug("  winner %s → +%.1f cohesion", winner_id, bonus)

    # Apply voter bonuses/penalties
    cohesion_changes = len(winner_list)  # Count winner bonuses
    for voter_name, ranking in votes.items():
        if not ranking:
            continue
        top_two = set(ranking[:2])
        if winner_set & top_two:
            # Voter backed a winner → +1.0
            if voter_name in id_map:
                id_map[voter_name].cohesion += 1.0
                cohesion_changes += 1
                logger.debug("  %s backed winner → voter +1.0", voter_name)
        else:
            # Voter did not back any winner → -1.0
            if voter_name in id_map:
                id_map[voter_name].cohesion -= 1.0
                cohesion_changes += 1
                logger.debug("  %s did not back winners %s → voter -1.0", voter_name, winner_list)

    logger.info(
        "update_cohesion: applied %d cohesion changes",
        cohesion_changes,
    )

    return list(id_map.values())


def vote_peer_approval(
    ctx: "InstanceContext",
    colony: list[Identity],
    cfg: ThrivemindConfig,  # noqa: ARG001 kept for API consistency
    constitution_result: str,
    colony_overview: str,
) -> list[Identity]:
    """Peer approval voting after constitution vote.

    Each individual endorses exactly 2 others. Approval changes by:
        new_approval = prior_approval + votes_received - 1

    The -1 is the cost of voting (you always spend 1 regardless of outcome).
    """
    id_map = {ind.name: ind for ind in colony}
    votes_received: dict[str, int] = {ind.name: 0 for ind in colony}
    valid_names = set(id_map)

    for individual in colony:
        prompt = (
            _PROMPT_VOTE_PEER
            .replace("{name}", individual.name)
            .replace("{constitution_result}", constitution_result)
            .replace("{colony_overview}", colony_overview)
        )
        response = ctx.llm(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=256,
            caller=f"peer_vote_{individual.name}",
        )
        raw = (response.message or "").strip()

        # Parse JSON endorsements
        endorsements: list[str] = []
        try:
            data = json.loads(raw)
            endorsements = [str(e) for e in data.get("endorsements", [])]
        except (ValueError, AttributeError):
            # Try to extract names from raw text
            for name in valid_names:
                if name in raw and name != individual.name:
                    endorsements.append(name)

        # Apply endorsements (max 2, excluding self)
        for endorsed in endorsements[:2]:
            if endorsed in votes_received and endorsed != individual.name:
                votes_received[endorsed] += 1

    # Apply approval delta: votes_received - 1 (cost of voting)
    for ind in colony:
        ind.approval = ind.approval + votes_received[ind.name] - 1

    logger.info(
        "Peer approval vote: %s",
        ", ".join(f"{n}={v}" for n, v in sorted(votes_received.items())),
    )
    return colony


# ---------------------------------------------------------------------------
# Deliberation / consensus helpers
# ---------------------------------------------------------------------------


def pick_fallback_candidate_ids(
    scores: dict[str, int], threshold: float
) -> tuple[list[str], float]:
    """Pick top candidates until cumulative score crosses threshold ratio."""
    if not scores:
        return [], 0.0

    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    total = sum(max(0, score) for _, score in ranked)
    if total <= 0:
        return [ranked[0][0]], 1.0

    chosen: list[str] = []
    running = 0
    coverage = 0.0
    for candidate_id, score in ranked:
        chosen.append(candidate_id)
        running += max(0, score)
        coverage = running / total
        if coverage > threshold:
            break
    return chosen, coverage


def join_fallback_messages(candidates: dict[str, str], candidate_ids: list[str]) -> str:
    """Join selected candidate messages in ranked order, dropping empties/duplicates."""
    parts: list[str] = []
    seen: set[str] = set()
    for candidate_id in candidate_ids:
        text = (candidates.get(candidate_id) or "").strip()
        if not text:
            continue
        normalized = " ".join(text.split())
        if normalized in seen:
            continue
        seen.add(normalized)
        parts.append(text)
    return "\n\n".join(parts).strip()


def winner_approval_ratio(result: dict) -> float:
    """Calculate the approval ratio of the winning candidate."""
    scores = result.get("scores", {})
    if not isinstance(scores, dict) or not scores:
        return 0.0

    total_score = sum(max(0, int(score)) for score in scores.values())
    if total_score <= 0:
        return 1.0 if int(result.get("candidate_count", 0) or 0) == 1 else 0.0

    winner_id = str(result.get("winner_member", ""))
    winner_score = max(0, int(scores.get(winner_id, 0)))
    return winner_score / total_score


def format_consensus_status(
    result: dict,
    fallback_coverage: float | None = None,
    drafters: list[str] | None = None,
) -> str:
    """Format a human-readable consensus status line with drafter attribution."""
    # Use new approval voting consensus field if available
    winner_consensus = result.get("consensus", 0.0)
    if winner_consensus == 0.0:
        # Fallback to old Borda ratio calculation for backwards compatibility
        winner_consensus = winner_approval_ratio(result)
    
    winner_pct = max(0, min(100, int(round(winner_consensus * 100))))
    has_consensus = (
        bool(result.get("has_consensus"))
        or int(result.get("candidate_count", 0) or 0) == 1
    )
    has_combined = bool(result.get("has_combined_consensus", False))

    # Build drafter attribution
    if drafters and len(drafters) >= 2:
        drafter_text = f"{drafters[0]} and {drafters[1]}"
    elif drafters and len(drafters) == 1:
        drafter_text = drafters[0]
    else:
        drafter_text = ""

    if has_consensus:
        pct = winner_pct
        if drafter_text:
            return f"Drafted by {drafter_text} ({pct}% consensus)"
        return f"Consensus: {pct}% approval."

    if has_combined:
        combined_pct = max(0, min(100, int(round(result.get("combined_consensus", 0.0) * 100))))
        if drafter_text:
            return f"Drafted by {drafter_text} ({combined_pct}% consensus)"
        return f"Consensus: Combined {combined_pct}% approval (top 2 candidates)."

    if fallback_coverage is not None and fallback_coverage > 0:
        fallback_pct = max(0, min(100, int(round(fallback_coverage * 100))))
        return (
            "Consensus: No consensus "
            f"(top candidate {winner_pct}% approval; fallback coalition {fallback_pct}% approval)."
        )

    return f"Consensus: No consensus (top candidate {winner_pct}% approval)."


def with_consensus_status(status: str, message: str) -> str:
    """Prepend consensus status to a message body."""
    body = (message or "").strip()
    return f"{body}\n\n{status}".strip() if body else status
