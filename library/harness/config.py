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
class SchedulerConfig:
    check_interval: int = 300   # seconds between check runs
    work_interval: int = 60     # seconds between work runs
    log_file: str | None = None


@dataclass
class ProviderConfig:
    id: str
    type: str
    base_url: str | None = None
    api_key: str | None = None
    max_concurrency: int | None = None


@dataclass
class AdapterConfig:
    id: str
    type: str
    homeserver: str | None = None
    base_dir: str | None = None


@dataclass
class SyncConfig:
    repo: str | None = None       # path to data repo (or subdir within it)
    prefix: str | None = None     # subdirectory within repo for this project
    branch: str = "main"
    remote: str | None = None     # git remote URL for initial clone


@dataclass
class AnalyticsConfig:
    base_url: str = "http://localhost:4000"
    api_key: str | None = None


@dataclass
class CompactConfig:
    provider: str
    model: str
    threshold_chars: int = 6000


@dataclass
class HarnessConfig:
    providers: list[ProviderConfig] = field(default_factory=list)
    adapters: list[AdapterConfig] = field(default_factory=list)
    storage_dir: str = "instances"
    store_path: str = "harness.db"
    poll_interval: int = 30
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    sync: SyncConfig = field(default_factory=SyncConfig)
    analytics: AnalyticsConfig | None = None
    compact: CompactConfig | None = None

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
class SchedulingConstraints:
    """Per-instance scheduling budget constraints."""

    # Guaranteed thinking windows
    guaranteed_thinking_interval: int = 14400  # seconds (4 hours)

    # Reply budget between guaranteed thinking windows
    max_replies_per_window: int = 3

    # Reactive thinking after messages
    reactive_thinking_max_sessions: int = 2  # max thinking sessions per message event
    reactive_thinking_cooldown: int = 900    # seconds (15 min) between reactive sessions

    # Phase restrictions for on_message (e.g., {"THINKING"} to exclude COMPOSING/REVIEWING)
    on_message_thinking_phases: set[str] | None = None


@dataclass
class MessagingConfig:
    adapter: str
    entity_id: str = ""
    access_token: str | None = None
    spaces: list[SpaceMapping] = field(default_factory=list)


@dataclass
class InstanceConfig:
    instance_id: str
    species: str
    provider: str
    model: str
    messaging: MessagingConfig | None = None
    schedule: dict[str, object] = field(default_factory=dict)
    scheduling_constraints: SchedulingConstraints | None = None
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

    scheduler_raw = resolved.get("scheduler", {})
    scheduler = SchedulerConfig(
        check_interval=int(scheduler_raw.get("check_interval", 300)),
        work_interval=int(scheduler_raw.get("work_interval", 60)),
        log_file=scheduler_raw.get("log_file"),
    )

    sync_raw = resolved.get("sync", {})
    sync = SyncConfig(
        repo=sync_raw.get("repo"),
        prefix=sync_raw.get("prefix"),
        branch=sync_raw.get("branch", "main"),
        remote=sync_raw.get("remote"),
    )

    analytics = None
    if "analytics" in resolved:
        analytics_raw = resolved["analytics"]
        analytics = AnalyticsConfig(
            base_url=analytics_raw.get("base_url", "http://localhost:4000"),
            api_key=analytics_raw.get("api_key"),
        )

    compact = None
    if "compact" in resolved:
        compact_raw = resolved["compact"]
        compact = CompactConfig(
            provider=compact_raw["provider"],
            model=compact_raw["model"],
            threshold_chars=int(compact_raw.get("threshold_chars", 6000)),
        )

    return HarnessConfig(
        providers=providers,
        adapters=adapters,
        storage_dir=resolved.get("storage_dir", "instances"),
        store_path=resolved.get("store_path", "harness.db"),
        poll_interval=resolved.get("poll_interval", 30),
        scheduler=scheduler,
        sync=sync,
        analytics=analytics,
        compact=compact,
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
            access_token=m.get("access_token"),
            spaces=spaces,
        )

    scheduling_constraints = None
    if "scheduling_constraints" in resolved:
        sc = resolved["scheduling_constraints"]
        phases = sc.get("on_message_thinking_phases")
        scheduling_constraints = SchedulingConstraints(
            guaranteed_thinking_interval=int(
                sc.get("guaranteed_thinking_interval", 14400)
            ),
            max_replies_per_window=int(sc.get("max_replies_per_window", 3)),
            reactive_thinking_max_sessions=int(
                sc.get("reactive_thinking_max_sessions", 2)
            ),
            reactive_thinking_cooldown=int(sc.get("reactive_thinking_cooldown", 900)),
            on_message_thinking_phases=set(phases) if phases else None,
        )

    return InstanceConfig(
        instance_id=instance_id,
        species=resolved["species"],
        provider=resolved["provider"],
        model=resolved["model"],
        messaging=messaging,
        schedule=resolved.get("schedule", {}),
        scheduling_constraints=scheduling_constraints,
        extra={
            k: v
            for k, v in resolved.items()
            if k
            not in (
                "species",
                "provider",
                "model",
                "messaging",
                "schedule",
                "scheduling_constraints",
            )
        },
    )
