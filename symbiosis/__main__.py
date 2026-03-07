"""CLI entry point — `python -m symbiosis` or `symbiosis run`."""

from __future__ import annotations

import argparse
import importlib
import inspect
import logging
import logging.handlers
import pkgutil
import sys
from pathlib import Path

from symbiosis.harness.config import load_harness_config, load_instance_config
from symbiosis.harness.registry import Registry
from symbiosis.harness.scheduler import Scheduler, build_providers, build_adapters
from symbiosis.species import Species
from symbiosis.setup_wizard import run_setup

_INSTANCE_TEMPLATE_STEMS = {"example", "sample", "template"}

_KNOWN_COMMANDS = {"run", "setup", "check", "work", "tick", "schedule", "-h", "--help"}


class _MaxLevelFilter(logging.Filter):
    def __init__(self, max_level: int):
        super().__init__()
        self._max_level = max_level

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno <= self._max_level


def load_species(species_id: str):
    """Load a species by ID. Returns a Species instance."""
    import symbiosis.species as species_pkg

    for mod_info in pkgutil.iter_modules(species_pkg.__path__):
        if mod_info.name.startswith("_"):
            continue

        module_name = f"{species_pkg.__name__}.{mod_info.name}"
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue

        for _, cls in inspect.getmembers(module, inspect.isclass):
            if cls.__module__ != module_name:
                continue
            if not issubclass(cls, Species) or cls is Species:
                continue
            instance = cls()
            if instance.manifest().species_id == species_id:
                return instance

    raise ValueError(f"Unknown species: {species_id}")


def _is_template_instance_file(path: Path) -> bool:
    stem = path.stem.lower()
    return (
        stem in _INSTANCE_TEMPLATE_STEMS
        or stem.endswith(".example")
        or stem.endswith(".sample")
        or stem.endswith(".template")
    )


def _configure_logging(level_name: str, log_file: str | None) -> None:
    level = getattr(logging, level_name)
    fmt = logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s")

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    stdout_handler.setLevel(level)
    stdout_handler.addFilter(_MaxLevelFilter(logging.WARNING))
    stdout_handler.setFormatter(fmt)
    root.addHandler(stdout_handler)

    stderr_handler = logging.StreamHandler(stream=sys.stderr)
    stderr_handler.setLevel(logging.ERROR)
    stderr_handler.setFormatter(fmt)
    root.addHandler(stderr_handler)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
            encoding="utf-8",
        )
        fh.setFormatter(fmt)
        root.addHandler(fh)


def _load_registry(base: Path) -> tuple[Registry, object]:
    """Load harness config, build registry, return (registry, harness_config)."""
    harness_config = load_harness_config(base / "config" / "harness.yaml")
    registry = Registry()
    instances_dir = base / "config" / "instances"

    if instances_dir.exists():
        for config_file in sorted(instances_dir.glob("*.yaml")):
            if _is_template_instance_file(config_file):
                continue
            instance_config = load_instance_config(config_file)
            species_id = instance_config.species
            if species_id not in registry.list_species():
                species = load_species(species_id)
                registry.register_species(species)
            registry.register_instance(instance_config)

    return registry, harness_config


def _run_scheduler(args: argparse.Namespace) -> None:
    """Run the legacy continuous scheduler loop."""
    _configure_logging(args.log_level, getattr(args, "log_file", None))

    base = Path(args.base_dir)
    harness_config = load_harness_config(base / args.config)

    providers = build_providers(harness_config)
    adapters = build_adapters(harness_config)

    registry = Registry()
    instances_dir = base / "config" / "instances"

    if instances_dir.exists():
        for config_file in sorted(instances_dir.glob("*.yaml")):
            if _is_template_instance_file(config_file):
                continue
            instance_config = load_instance_config(config_file)
            species_id = instance_config.species
            if species_id not in registry.list_species():
                species = load_species(species_id)
                registry.register_species(species)
            registry.register_instance(instance_config)

    scheduler = Scheduler(
        harness_config=harness_config,
        registry=registry,
        providers=providers,
        adapters=adapters,
        base_dir=base,
    )
    scheduler.run_forever()


def _run_check(args: argparse.Namespace) -> None:
    """Poll adapters and enqueue jobs (no pipeline execution)."""
    _configure_logging(args.log_level, getattr(args, "log_file", None))

    from symbiosis.harness.checker import Checker
    from symbiosis.harness.store import open_store

    base = Path(args.base_dir)
    harness_config = load_harness_config(base / args.config)
    store_db = open_store(base / harness_config.store_path)

    registry, _ = _load_registry(base)

    checker = Checker(
        harness_config=harness_config,
        registry=registry,
        store_db=store_db,
        base_dir=base,
    )
    checker.run()
    store_db.close()


def _run_work(args: argparse.Namespace) -> None:
    """Drain the job queue run-to-empty."""
    _configure_logging(args.log_level, getattr(args, "log_file", None))

    from symbiosis.harness.worker import Worker
    from symbiosis.harness.store import open_store

    base = Path(args.base_dir)
    harness_config = load_harness_config(base / args.config)
    store_db = open_store(base / harness_config.store_path)

    registry, _ = _load_registry(base)
    providers = build_providers(harness_config)
    adapters = build_adapters(harness_config)

    worker = Worker(
        harness_config=harness_config,
        registry=registry,
        providers=providers,
        adapters=adapters,
        store_db=store_db,
        base_dir=base,
    )
    worker.run()
    store_db.close()


def _run_tick(args: argparse.Namespace) -> None:
    """Run one check + one work cycle (convenience for testing)."""
    _run_check(args)
    _run_work(args)


def _run_schedule(args: argparse.Namespace) -> None:
    """Generate OS scheduler config files."""
    from symbiosis.scheduling import (
        generate_schedule_files,
        print_install_instructions,
        detect_scheduler,
    )

    base = Path(args.base_dir)
    harness_config = load_harness_config(base / args.config)
    sched_config = harness_config.scheduler

    scheduler_type = getattr(args, "scheduler_type", "auto") or "auto"
    actual_type = scheduler_type if scheduler_type != "auto" else detect_scheduler()

    files = generate_schedule_files(
        base_dir=base,
        working_dir=Path.cwd(),
        check_interval=sched_config.check_interval,
        work_interval=sched_config.work_interval,
        log_dir="logs",
        config=args.config,
        scheduler_type=scheduler_type,
    )

    print_install_instructions(files, actual_type)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Symbiosis agent framework")
    subparsers = parser.add_subparsers(dest="command")

    # Shared arguments factory
    def _add_common(p):
        p.add_argument(
            "--config", "-c",
            default="config/harness.yaml",
            help="Path to harness config file",
        )
        p.add_argument(
            "--base-dir", "-d",
            default=".",
            help="Base directory for storage and data",
        )
        p.add_argument(
            "--log-level",
            default="INFO",
            choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        )
        p.add_argument(
            "--log-file",
            default=None,
            metavar="PATH",
            help="Write logs to this file in addition to stdout (rotating, 10 MB × 5)",
        )

    run_parser = subparsers.add_parser("run", help="Run continuous scheduler loop (legacy)")
    _add_common(run_parser)

    check_parser = subparsers.add_parser("check", help="Poll adapters and enqueue jobs")
    _add_common(check_parser)

    work_parser = subparsers.add_parser("work", help="Drain job queue run-to-empty")
    _add_common(work_parser)

    tick_parser = subparsers.add_parser("tick", help="Run one check + work cycle")
    _add_common(tick_parser)

    schedule_parser = subparsers.add_parser("schedule", help="Generate OS scheduler config files")
    schedule_parser.add_argument(
        "--base-dir", "-d",
        default=".",
        help="Base directory for config files",
    )
    schedule_parser.add_argument(
        "--config", "-c",
        default="config/harness.yaml",
        help="Path to harness config file",
    )
    schedule_parser.add_argument(
        "--type",
        dest="scheduler_type",
        default="auto",
        choices=["auto", "launchd", "systemd", "crontab"],
        help="OS scheduler type (default: auto-detect)",
    )

    setup_parser = subparsers.add_parser("setup", help="Run interactive setup wizard")
    setup_parser.add_argument(
        "--base-dir", "-d",
        default=".",
        help="Base directory for config files",
    )

    return parser


def main(argv: list[str] | None = None):
    parser = _build_parser()
    raw_args = list(sys.argv[1:] if argv is None else argv)

    # Backward compatibility:
    # `symbiosis -c config/harness.yaml` should still mean `symbiosis run -c ...`
    if not raw_args:
        raw_args = ["run"]
    elif raw_args[0] not in _KNOWN_COMMANDS:
        raw_args = ["run", *raw_args]

    args = parser.parse_args(raw_args)

    if args.command == "setup":
        run_setup(base_dir=args.base_dir)
        return

    if args.command == "check":
        _run_check(args)
        return

    if args.command == "work":
        _run_work(args)
        return

    if args.command == "tick":
        _run_tick(args)
        return

    if args.command == "schedule":
        _run_schedule(args)
        return

    _run_scheduler(args)


if __name__ == "__main__":
    main()
