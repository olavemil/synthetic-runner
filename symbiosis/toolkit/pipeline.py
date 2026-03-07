"""YAML pipeline runner — parse and execute declarative pipeline definitions."""

from __future__ import annotations

import logging
from typing import Any, Callable, TYPE_CHECKING

import yaml

from pathlib import Path

from symbiosis.toolkit import patterns

if TYPE_CHECKING:
    from symbiosis.harness.context import InstanceContext

logger = logging.getLogger(__name__)


def _lazy_deliberate(name: str) -> Callable:
    """Return a lazy-imported deliberate function to avoid circular imports."""
    def _fn(*args, **kwargs):
        import importlib
        mod = importlib.import_module("symbiosis.toolkit.deliberate")
        return getattr(mod, name)(*args, **kwargs)
    _fn.__name__ = name
    return _fn


# Map stage names to pattern functions
STAGE_REGISTRY: dict[str, Callable] = {
    "gut_response": patterns.gut_response,
    "plan_response": patterns.plan_response,
    "compose_response": patterns.compose_response,
    "run_subconscious": patterns.run_subconscious,
    "run_react": patterns.run_react,
    "update_relationships": patterns.update_relationships,
    "distill_memory": patterns.distill_memory,
    "distill_messages": patterns.distill_messages,
    "run_session": patterns.run_session,
    "llm_generate": patterns.llm_generate,
    "thinking_session": patterns.thinking_session,
    "format_context": patterns.format_context,
    "generate_with_identity": _lazy_deliberate("generate_with_identity"),
    "multi_generate": _lazy_deliberate("multi_generate"),
    "multi_vote": _lazy_deliberate("multi_vote"),
    "deliberate": _lazy_deliberate("deliberate"),
    "recompose": _lazy_deliberate("recompose"),
    "think_with_context": _lazy_deliberate("think_with_context"),
}


class PipelineError(Exception):
    pass


def resolve_input(ctx: InstanceContext, source: str, pipeline_state: dict) -> Any:
    """Resolve an input source reference to its value."""
    if source == "events.all":
        return pipeline_state.get("_events", [])

    if source == "events.formatted":
        from symbiosis.toolkit.prompts import format_events
        return format_events(pipeline_state.get("_events", []))

    if source == "item":
        return pipeline_state.get("_foreach_item")

    if source.startswith("memory."):
        path = source[len("memory."):]
        if path == "*":
            from symbiosis.toolkit.prompts import read_memory
            return read_memory(ctx)
        return ctx.read(f"{path}.md") or ctx.read(path)

    if source.startswith("pipeline."):
        key = source[len("pipeline."):]
        return pipeline_state.get(key)

    if source.startswith("inbox."):
        return ctx.read_inbox()

    if source.startswith("config."):
        key = source[len("config."):]
        return ctx.config(key)

    if source.startswith("file:"):
        rel_path = source[len("file:"):]
        species_dir = pipeline_state.get("_species_dir")
        if species_dir:
            file_path = Path(species_dir) / rel_path
            if file_path.exists():
                text = file_path.read_text()
                text = text.replace("{instance_id}", ctx.instance_id)
                text = text.replace("{species_id}", ctx.species_id)
                return text
        return f"(file not found: {rel_path})"

    if source.startswith("store."):
        parts = source[len("store."):].split(".", 1)
        if len(parts) == 2 and parts[0] == "shared":
            ns_and_key = parts[1].split("/", 1)
            if len(ns_and_key) == 2:
                ns, key = ns_and_key
                store = ctx.shared_store(ns)
                if key.endswith("*"):
                    return store.scan(key[:-1])
                return store.get(key)
        else:
            ns_and_key = parts[0].split("/", 1) if len(parts) == 1 else parts
            if len(ns_and_key) == 2:
                ns, key = ns_and_key
                store = ctx.store(ns)
                if key.endswith("*"):
                    return store.scan(key[:-1])
                return store.get(key)

    return source  # treat as literal


def write_output(ctx: InstanceContext, destination: str, value: Any, pipeline_state: dict) -> None:
    """Write a value to an output destination."""
    if destination.startswith("memory."):
        path = destination[len("memory."):]
        content = value if isinstance(value, str) else str(value)
        ctx.write(f"{path}.md" if "." not in path else path, content)

    elif destination.startswith("pipeline."):
        key = destination[len("pipeline."):]
        pipeline_state[key] = value

    elif destination.startswith("store."):
        parts = destination[len("store."):].split(".", 1)
        if len(parts) == 2 and parts[0] == "shared":
            ns_and_key = parts[1].split("/", 1)
            if len(ns_and_key) == 2:
                ns, key = ns_and_key
                key = key.replace("{instance_id}", ctx.instance_id)
                ctx.shared_store(ns).put(key, value)
        else:
            ns_and_key = parts[0].split("/", 1) if len(parts) == 1 else parts
            if len(ns_and_key) == 2:
                ns, key = ns_and_key
                key = key.replace("{instance_id}", ctx.instance_id)
                ctx.store(ns).put(key, value)


def consume_input(ctx: InstanceContext, source: str) -> None:
    """Clear an input source after successful processing."""
    if source.startswith("inbox."):
        pass  # inbox is consumed on read_inbox()
    elif source.startswith("memory."):
        path = source[len("memory."):]
        ctx.write(f"{path}.md" if "." not in path else path, "")
    elif source.startswith("store."):
        # Clear store key
        parts = source[len("store."):].split(".", 1)
        if len(parts) == 2 and parts[0] == "shared":
            ns_and_key = parts[1].split("/", 1)
            if len(ns_and_key) == 2:
                ns, key = ns_and_key
                ctx.shared_store(ns).delete(key)


def apply_preprocessor(
    ctx: InstanceContext,
    value: Any,
    preprocessor: dict,
) -> Any:
    """Apply a preprocessor to an input value."""
    pp_type = preprocessor.get("type")

    if pp_type == "truncate":
        max_chars = preprocessor.get("max_chars", 2000)
        if isinstance(value, str):
            return value[:max_chars]
        return value

    if pp_type == "distill":
        if isinstance(value, list):
            keep_recent = preprocessor.get("keep_recent", 4)
            if len(value) > keep_recent:
                from symbiosis.harness.adapters import Event
                older = value[:-keep_recent]
                recent = value[-keep_recent:]
                if older and all(isinstance(e, Event) for e in older):
                    summary = patterns.distill_messages(ctx, older)
                    return [{"summary": summary}] + recent
            return value
        if isinstance(value, str):
            return value  # already text
        return value

    if pp_type == "map":
        # Apply sub-pipeline to each item in a list
        if not isinstance(value, list):
            return value
        return value  # placeholder for sub-pipeline expansion

    if pp_type == "reduce":
        if isinstance(value, list):
            combined = "\n".join(str(v) for v in value)
            return combined
        return value

    return value


def _run_single_stage(
    ctx: InstanceContext,
    stage_def: dict,
    pipeline_state: dict,
) -> Any:
    """Execute the LLM/pattern call for a stage (no foreach handling)."""
    stage_name = stage_def["stage"]

    if stage_name not in STAGE_REGISTRY:
        raise PipelineError(f"Unknown stage: {stage_name}")

    inputs = {}
    for slot, source in stage_def.get("inputs", {}).items():
        value = resolve_input(ctx, source, pipeline_state)
        preprocessors = stage_def.get("preprocessors", {})
        if slot in preprocessors:
            value = apply_preprocessor(ctx, value, preprocessors[slot])
        inputs[slot] = value

    fn = STAGE_REGISTRY[stage_name]
    try:
        result = fn(ctx, **inputs)
    except TypeError:
        result = fn(ctx)

    for slot, destination in stage_def.get("outputs", {}).items():
        output_value = result if not isinstance(result, dict) else result.get(slot, result)
        write_output(ctx, destination, output_value, pipeline_state)

    for slot, should_consume in stage_def.get("consume_inputs", {}).items():
        if should_consume:
            source = stage_def.get("inputs", {}).get(slot)
            if source:
                consume_input(ctx, source)

    return result


def run_stage(
    ctx: InstanceContext,
    stage_def: dict,
    pipeline_state: dict,
) -> Any:
    """Execute a single pipeline stage, with optional foreach/collect iteration."""
    foreach_source = stage_def.get("foreach")
    collect_to = stage_def.get("collect")
    collect_key = stage_def.get("collect_key")

    if foreach_source:
        items = resolve_input(ctx, foreach_source, pipeline_state)
        container: Any = {} if collect_key else []
        for item in (items if isinstance(items, list) else [items]):
            local_state = {**pipeline_state, "_foreach_item": item}
            r = _run_single_stage(ctx, stage_def, local_state)
            if collect_key:
                container[getattr(item, collect_key, str(item))] = r
            else:
                container.append(r)
        if collect_to:
            pipeline_state[collect_to] = container
        return container

    return _run_single_stage(ctx, stage_def, pipeline_state)


def run_pipeline(
    ctx: InstanceContext,
    steps: list[dict],
    events: list | None = None,
    initial_state: dict | None = None,
) -> dict:
    """Execute a pipeline — a linear sequence of stages."""
    pipeline_state: dict = {"_events": events or []}
    if initial_state:
        pipeline_state.update(initial_state)

    for step in steps:
        logger.info("Running stage: %s", step.get("stage"))
        run_stage(ctx, step, pipeline_state)

    return pipeline_state


def load_pipeline(yaml_text: str) -> dict:
    """Parse a YAML pipeline definition."""
    data = yaml.safe_load(yaml_text)
    if not isinstance(data, dict):
        raise PipelineError("Pipeline YAML must be a mapping")
    return data


def register_stage(name: str, fn: Callable) -> None:
    """Register a custom stage function."""
    STAGE_REGISTRY[name] = fn
