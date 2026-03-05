"""Interactive setup wizard for bootstrapping Symbiosis config files."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

import yaml
from dotenv import dotenv_values


PromptInput = Callable[[str], str]
PromptOutput = Callable[[str], None]

_ENV_LINE_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=")

_INTELLIGENCE_TYPES: list[tuple[str, str]] = [
    ("Persistent", "persistent"),
    ("Generational", "generational"),
    ("Hivemind", "hivemind"),
]

_OPERATING_MODES: list[tuple[str, str]] = [
    ("Reactive", "reactive"),
    ("Hybrid (reactive + scheduled)", "hybrid"),
    ("Scheduled", "scheduled"),
]

_INSTANCE_TEMPLATE_STEMS = {"example", "sample", "template"}


def _ask_text(
    prompt: str,
    input_fn: PromptInput,
    output_fn: PromptOutput,
    *,
    default: str | None = None,
    allow_empty: bool = True,
) -> str:
    while True:
        suffix = f" [{default}]" if default is not None and default != "" else ""
        raw = input_fn(f"{prompt}{suffix}: ").strip()
        if raw:
            return raw
        if default is not None:
            return default
        if allow_empty:
            return ""
        output_fn("A value is required.")


def _ask_choice(
    prompt: str,
    choices: list[tuple[str, str]],
    input_fn: PromptInput,
    output_fn: PromptOutput,
    *,
    default_index: int = 0,
) -> str:
    output_fn(prompt)
    for idx, (label, _) in enumerate(choices, start=1):
        output_fn(f"  {idx}. {label}")

    default_pick = str(default_index + 1)
    while True:
        raw = input_fn(f"Choose [default {default_pick}]: ").strip()
        if not raw:
            return choices[default_index][1]
        if raw.isdigit():
            choice_index = int(raw) - 1
            if 0 <= choice_index < len(choices):
                return choices[choice_index][1]
        output_fn("Please enter a valid number.")


def _upsert_item(items: list[dict], item: dict) -> None:
    item_id = item.get("id")
    if not item_id:
        items.append(item)
        return

    for idx, existing in enumerate(items):
        if existing.get("id") == item_id:
            merged = dict(existing)
            merged.update(item)
            items[idx] = merged
            return

    items.append(item)


def _load_harness_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _write_harness_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def _upsert_env_values(path: Path, updates: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = path.read_text().splitlines() if path.exists() else []

    key_to_index: dict[str, int] = {}
    for idx, line in enumerate(lines):
        match = _ENV_LINE_RE.match(line)
        if match:
            key_to_index[match.group(1)] = idx

    for key, value in updates.items():
        entry = f"{key}={value}"
        if key in key_to_index:
            lines[key_to_index[key]] = entry
        else:
            lines.append(entry)

    path.write_text("\n".join(lines).rstrip() + "\n")


def _slugify_instance_name(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", name.strip().lower())
    slug = re.sub(r"-{2,}", "-", slug).strip("-")
    return slug or "agent-1"


def _default_model_for_provider(provider_id: str) -> str:
    if "anthropic" in provider_id:
        return "claude-sonnet-4-6"
    return "local-model"


def _default_heartbeat(intelligence_type: str) -> str:
    if intelligence_type == "generational":
        return "0 */6 * * *"
    if intelligence_type == "hivemind":
        return "*/15 * * * *"
    return "0 * * * *"


def _is_template_instance_file(path: Path) -> bool:
    stem = path.stem.lower()
    return (
        stem in _INSTANCE_TEMPLATE_STEMS
        or stem.endswith(".example")
        or stem.endswith(".sample")
        or stem.endswith(".template")
    )


def _build_pipeline_steps(
    intelligence_type: str,
    operating_mode: str,
) -> dict:
    reactive_steps: list[dict] = []
    scheduled_steps: list[dict] = []

    if intelligence_type == "persistent":
        reactive_steps = [
            {
                "stage": "gut_response",
                "inputs": {"events": "inbox.messages"},
                "outputs": {"guidance": "pipeline.gut"},
            },
            {
                "stage": "plan_response",
                "inputs": {"gut_brief": "pipeline.gut"},
                "outputs": {"plan": "pipeline.plan"},
            },
            {
                "stage": "compose_response",
                "inputs": {"guidance": "pipeline.plan"},
                "outputs": {"message": "pipeline.message"},
            },
            {"stage": "run_subconscious", "inputs": {"session_type": "reactive"}},
            {"stage": "run_react", "inputs": {"session_type": "reactive"}},
            {
                "stage": "update_relationships",
                "inputs": {"session_type": "reactive", "events": "inbox.messages"},
            },
        ]
        scheduled_steps = [
            {"stage": "distill_memory", "outputs": {"digest": "memory.digest"}},
            {"stage": "run_subconscious", "inputs": {"session_type": "heartbeat"}},
            {"stage": "run_react", "inputs": {"session_type": "heartbeat"}},
        ]
    elif intelligence_type == "generational":
        reactive_steps = [
            {
                "stage": "gut_response",
                "inputs": {"events": "inbox.messages"},
                "outputs": {"guidance": "pipeline.gut"},
            },
            {"stage": "run_subconscious", "inputs": {"session_type": "reactive"}},
        ]
        scheduled_steps = [
            {"stage": "distill_memory", "outputs": {"digest": "memory.digest"}},
            {"stage": "run_react", "inputs": {"session_type": "generational"}},
        ]
    else:  # hivemind
        reactive_steps = [
            {
                "stage": "distill_messages",
                "inputs": {"messages": "inbox.messages"},
                "outputs": {"summary": "memory.inbox_summary"},
            },
            {"stage": "run_subconscious", "inputs": {"session_type": "hivemind"}},
        ]
        scheduled_steps = [
            {"stage": "distill_memory", "outputs": {"digest": "memory.collective_digest"}},
            {"stage": "run_react", "inputs": {"session_type": "hivemind"}},
        ]

    pipeline: dict = {}
    if operating_mode in {"reactive", "hybrid"}:
        pipeline["on_inbox"] = {"steps": reactive_steps}
    if operating_mode in {"scheduled", "hybrid"}:
        pipeline["on_schedule"] = {
            "cron": _default_heartbeat(intelligence_type),
            "steps": scheduled_steps,
        }
    return pipeline


def _build_pipeline_yaml(
    instance_id: str,
    intelligence_type: str,
    operating_mode: str,
) -> dict:
    return {
        "species_id": instance_id,
        "intelligence_type": intelligence_type,
        "operating_mode": operating_mode,
        "pipeline": _build_pipeline_steps(intelligence_type, operating_mode),
    }


def _write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def run_setup(
    base_dir: str | Path = ".",
    *,
    input_fn: PromptInput = input,
    output_fn: PromptOutput = print,
) -> None:
    """Run interactive bootstrap setup for harness + first instance config."""
    base = Path(base_dir)
    env_path = base / ".env"
    harness_path = base / "config" / "harness.yaml"
    instances_dir = base / "config" / "instances"
    pipelines_dir = base / "config" / "pipelines"

    instances_dir.mkdir(parents=True, exist_ok=True)
    pipelines_dir.mkdir(parents=True, exist_ok=True)

    output_fn("Symbiosis setup")
    output_fn("General config (leave blank where optional).")

    existing_env = {k: v for k, v in dotenv_values(env_path).items() if v is not None}
    env_updates: dict[str, str] = {}

    anthropic_api_key = _ask_text(
        "ANTHROPIC_API_KEY (optional)",
        input_fn,
        output_fn,
        default=existing_env.get("ANTHROPIC_API_KEY", ""),
        allow_empty=True,
    )
    lmstudio_base_url = _ask_text(
        "LMSTUDIO_BASE_URL",
        input_fn,
        output_fn,
        default=existing_env.get("LMSTUDIO_BASE_URL", "http://localhost:1234/v1"),
        allow_empty=True,
    )
    matrix_homeserver = _ask_text(
        "MATRIX_HOMESERVER (optional)",
        input_fn,
        output_fn,
        default=existing_env.get("MATRIX_HOMESERVER", ""),
        allow_empty=True,
    )

    env_updates["ANTHROPIC_API_KEY"] = anthropic_api_key
    env_updates["LMSTUDIO_BASE_URL"] = lmstudio_base_url
    env_updates["MATRIX_HOMESERVER"] = matrix_homeserver

    harness = _load_harness_yaml(harness_path)
    providers = harness.get("providers", [])
    adapters = harness.get("adapters", [])

    provider_choice = _ask_choice(
        "Provider setup",
        [
            ("Both (LMStudio + Anthropic)", "both"),
            ("LMStudio only", "lmstudio"),
            ("Anthropic only", "anthropic"),
            ("Keep existing providers only", "existing"),
        ],
        input_fn,
        output_fn,
        default_index=0 if not providers else 3,
    )

    if provider_choice in {"both", "lmstudio"}:
        _upsert_item(
            providers,
            {
                "id": "lmstudio",
                "type": "openai_compat",
                "base_url": "${LMSTUDIO_BASE_URL}",
                "api_key": "lm-studio",
            },
        )

    if provider_choice in {"both", "anthropic"}:
        _upsert_item(
            providers,
            {
                "id": "anthropic",
                "type": "anthropic",
                "api_key": "${ANTHROPIC_API_KEY}",
            },
        )

    add_matrix_adapter = _ask_choice(
        "Configure matrix-main adapter in harness config?",
        [("No", "no"), ("Yes", "yes")],
        input_fn,
        output_fn,
        default_index=0,
    )
    if add_matrix_adapter == "yes":
        _upsert_item(
            adapters,
            {
                "id": "matrix-main",
                "type": "matrix",
                "homeserver": "${MATRIX_HOMESERVER}",
            },
        )

    harness["providers"] = providers
    harness["adapters"] = adapters
    harness["storage_dir"] = harness.get("storage_dir", "instances")
    harness["store_path"] = harness.get("store_path", "harness.db")
    harness["poll_interval"] = harness.get("poll_interval", 30)

    existing_instances = sorted(
        p for p in instances_dir.glob("*.yaml")
        if not _is_template_instance_file(p)
    )
    if not existing_instances:
        create_first = _ask_choice(
            "No species instances found. Create the first instance now?",
            [("Yes", "yes"), ("No", "no")],
            input_fn,
            output_fn,
            default_index=0,
        )

        if create_first == "yes":
            raw_name = _ask_text(
                "Instance name",
                input_fn,
                output_fn,
                default="agent-1",
                allow_empty=False,
            )
            instance_id = _slugify_instance_name(raw_name)

            intelligence_type = _ask_choice(
                "Intelligence type",
                _INTELLIGENCE_TYPES,
                input_fn,
                output_fn,
                default_index=0,
            )
            operating_mode = _ask_choice(
                "Operating mode",
                _OPERATING_MODES,
                input_fn,
                output_fn,
                default_index=1,
            )

            if not providers:
                _upsert_item(
                    providers,
                    {
                        "id": "lmstudio",
                        "type": "openai_compat",
                        "base_url": "${LMSTUDIO_BASE_URL}",
                        "api_key": "lm-studio",
                    },
                )
                harness["providers"] = providers

            provider_ids = [p["id"] for p in providers if p.get("id")]
            if not provider_ids:
                provider_ids = ["lmstudio"]
            selected_provider = _ask_choice(
                "Default provider for this instance",
                [(pid, pid) for pid in provider_ids],
                input_fn,
                output_fn,
                default_index=0,
            )
            model = _ask_text(
                "Model",
                input_fn,
                output_fn,
                default=_default_model_for_provider(selected_provider),
                allow_empty=False,
            )

            configure_messaging = _ask_choice(
                "Configure Matrix messaging for this instance now?",
                [("No", "no"), ("Yes", "yes")],
                input_fn,
                output_fn,
                default_index=0,
            )

            messaging: dict | None = None
            if configure_messaging == "yes":
                has_matrix = any(a.get("id") == "matrix-main" for a in adapters)
                if not has_matrix:
                    _upsert_item(
                        adapters,
                        {
                            "id": "matrix-main",
                            "type": "matrix",
                            "homeserver": "${MATRIX_HOMESERVER}",
                        },
                    )
                    harness["adapters"] = adapters

                entity_id = _ask_text(
                    "Matrix entity_id (optional, e.g. @bot:matrix.org)",
                    input_fn,
                    output_fn,
                    default="",
                    allow_empty=True,
                )
                room_handle = _ask_text(
                    "Main room handle (optional, e.g. !room:matrix.org)",
                    input_fn,
                    output_fn,
                    default="",
                    allow_empty=True,
                )
                token_env_var = f"{instance_id.upper().replace('-', '_')}_MATRIX_TOKEN"
                token_value = _ask_text(
                    f"{token_env_var} (optional access token)",
                    input_fn,
                    output_fn,
                    default=existing_env.get(token_env_var, ""),
                    allow_empty=True,
                )
                env_updates[token_env_var] = token_value

                spaces = []
                if room_handle:
                    spaces.append({"name": "main", "handle": room_handle})

                messaging = {
                    "adapter": "matrix-main",
                    "entity_id": entity_id,
                    "access_token": f"${{{token_env_var}}}",
                    "spaces": spaces,
                }

            instance_path = instances_dir / f"{instance_id}.yaml"
            if instance_path.exists():
                overwrite = _ask_choice(
                    f"{instance_path.name} already exists. Overwrite?",
                    [("No", "no"), ("Yes", "yes")],
                    input_fn,
                    output_fn,
                    default_index=0,
                )
                if overwrite == "no":
                    output_fn("Skipped writing instance config.")
                else:
                    instance_data = {
                        "species": "draum",
                        "provider": selected_provider,
                        "model": model,
                        "intelligence": {
                            "type": intelligence_type,
                            "operating_mode": operating_mode,
                        },
                        "pipeline": {
                            "file": f"config/pipelines/{instance_id}.yaml",
                        },
                    }
                    if operating_mode in {"scheduled", "hybrid"}:
                        instance_data["schedule"] = {
                            "heartbeat": _default_heartbeat(intelligence_type)
                        }
                    if messaging is not None:
                        instance_data["messaging"] = messaging
                    _write_yaml(instance_path, instance_data)
                    output_fn(f"Wrote {instance_path}")
            else:
                instance_data = {
                    "species": "draum",
                    "provider": selected_provider,
                    "model": model,
                    "intelligence": {
                        "type": intelligence_type,
                        "operating_mode": operating_mode,
                    },
                    "pipeline": {
                        "file": f"config/pipelines/{instance_id}.yaml",
                    },
                }
                if operating_mode in {"scheduled", "hybrid"}:
                    instance_data["schedule"] = {
                        "heartbeat": _default_heartbeat(intelligence_type)
                    }
                if messaging is not None:
                    instance_data["messaging"] = messaging
                _write_yaml(instance_path, instance_data)
                output_fn(f"Wrote {instance_path}")

            pipeline_path = pipelines_dir / f"{instance_id}.yaml"
            pipeline_data = _build_pipeline_yaml(
                instance_id=instance_id,
                intelligence_type=intelligence_type,
                operating_mode=operating_mode,
            )
            _write_yaml(pipeline_path, pipeline_data)
            output_fn(f"Wrote {pipeline_path}")
    else:
        output_fn("Existing instance configs found; skipped first-species bootstrap.")

    _write_harness_yaml(harness_path, harness)
    _upsert_env_values(env_path, env_updates)

    output_fn(f"Wrote {harness_path}")
    output_fn(f"Updated {env_path}")
    output_fn("Setup complete. Run `symbiosis` or `symbiosis run` to start the scheduler.")
