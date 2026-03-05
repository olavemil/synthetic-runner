"""Config system — YAML loading with ${VAR} env resolution."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import dotenv_values


ENV_VAR_PATTERN = re.compile(r"\$\{(\w+)\}")


def _resolve_env_vars(value: str, env: dict[str, str | None]) -> str:
    """Replace ${VAR} references with values from env dict."""

    def replacer(match: re.Match) -> str:
        var_name = match.group(1)
        resolved = env.get(var_name)
        if resolved is None:
            raise ValueError(f"Environment variable ${{{var_name}}} is not set")
        return resolved

    return ENV_VAR_PATTERN.sub(replacer, value)


def _resolve_recursive(obj, env: dict[str, str | None]):
    """Walk a nested structure and resolve all ${VAR} strings."""
    if isinstance(obj, str):
        return _resolve_env_vars(obj, env)
    if isinstance(obj, dict):
        return {k: _resolve_recursive(v, env) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_recursive(item, env) for item in obj]
    return obj


def _load_env(env_path: Path | None = None) -> dict[str, str | None]:
    """Load .env file and merge with os.environ (os.environ takes precedence)."""
    env = {}
    if env_path and env_path.exists():
        env.update(dotenv_values(env_path))
    env.update(os.environ)
    return env


@dataclass
class ProviderConfig:
    id: str
    type: str
    base_url: str | None = None
    api_key: str | None = None


@dataclass
class AdapterConfig:
    id: str
    type: str
    homeserver: str | None = None
    access_token: str | None = None


@dataclass
class HarnessConfig:
    providers: list[ProviderConfig] = field(default_factory=list)
    adapters: list[AdapterConfig] = field(default_factory=list)
    storage_dir: str = "instances"
    store_path: str = "harness.db"
    poll_interval: int = 30

    def get_provider(self, provider_id: str) -> ProviderConfig:
        for p in self.providers:
            if p.id == provider_id:
                return p
        raise KeyError(f"Provider '{provider_id}' not found in config")

    def get_adapter(self, adapter_id: str) -> AdapterConfig:
        for a in self.adapters:
            if a.id == adapter_id:
                return a
        raise KeyError(f"Adapter '{adapter_id}' not found in config")


@dataclass
class SpaceMapping:
    name: str
    handle: str


@dataclass
class MessagingConfig:
    adapter: str
    entity_id: str = ""
    spaces: list[SpaceMapping] = field(default_factory=list)


@dataclass
class InstanceConfig:
    instance_id: str
    species: str
    provider: str
    model: str
    messaging: MessagingConfig | None = None
    schedule: dict[str, str] = field(default_factory=dict)
    extra: dict = field(default_factory=dict)


def load_harness_config(
    path: str | Path,
    env_path: str | Path | None = None,
) -> HarnessConfig:
    """Load and resolve harness.yaml."""
    path = Path(path)
    env = _load_env(Path(env_path) if env_path else path.parent.parent / ".env")

    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    resolved = _resolve_recursive(raw, env)

    providers = [
        ProviderConfig(**p) for p in resolved.get("providers", [])
    ]
    adapters = [
        AdapterConfig(**a) for a in resolved.get("adapters", [])
    ]

    return HarnessConfig(
        providers=providers,
        adapters=adapters,
        storage_dir=resolved.get("storage_dir", "instances"),
        store_path=resolved.get("store_path", "harness.db"),
        poll_interval=resolved.get("poll_interval", 30),
    )


def load_instance_config(
    path: str | Path,
    env_path: str | Path | None = None,
) -> InstanceConfig:
    """Load and resolve an instance YAML config."""
    path = Path(path)
    env = _load_env(Path(env_path) if env_path else path.parent.parent.parent / ".env")

    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    resolved = _resolve_recursive(raw, env)

    instance_id = path.stem

    messaging = None
    if "messaging" in resolved:
        m = resolved["messaging"]
        spaces = [SpaceMapping(**s) for s in m.get("spaces", [])]
        messaging = MessagingConfig(
            adapter=m["adapter"],
            entity_id=m.get("entity_id", ""),
            spaces=spaces,
        )

    return InstanceConfig(
        instance_id=instance_id,
        species=resolved["species"],
        provider=resolved["provider"],
        model=resolved["model"],
        messaging=messaging,
        schedule=resolved.get("schedule", {}),
        extra={
            k: v
            for k, v in resolved.items()
            if k not in ("species", "provider", "model", "messaging", "schedule")
        },
    )
