"""CLI entry point — `python -m symbiosis` or `symbiosis run`."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from symbiosis.harness.config import load_harness_config, load_instance_config
from symbiosis.harness.registry import Registry
from symbiosis.harness.scheduler import Scheduler, build_providers, build_adapters
from symbiosis.setup_wizard import run_setup

_INSTANCE_TEMPLATE_STEMS = {"example", "sample", "template"}


def load_species(species_id: str):
    """Load a species by ID. Returns a Species instance."""
    if species_id == "draum":
        from symbiosis.species.draum import DraumSpecies
        return DraumSpecies()
    raise ValueError(f"Unknown species: {species_id}")


def _is_template_instance_file(path: Path) -> bool:
    stem = path.stem.lower()
    return (
        stem in _INSTANCE_TEMPLATE_STEMS
        or stem.endswith(".example")
        or stem.endswith(".sample")
        or stem.endswith(".template")
    )


def _run_scheduler(args: argparse.Namespace) -> None:
    """Run scheduler loop."""

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    base = Path(args.base_dir)
    harness_config = load_harness_config(base / args.config)

    # Build providers and adapters
    providers = build_providers(harness_config)
    adapters = build_adapters(harness_config)

    # Load instances and register
    registry = Registry()
    instances_dir = base / "config" / "instances"

    if instances_dir.exists():
        for config_file in sorted(instances_dir.glob("*.yaml")):
            if _is_template_instance_file(config_file):
                continue
            instance_config = load_instance_config(config_file)

            # Load and register species if not already done
            species_id = instance_config.species
            if species_id not in registry.list_species():
                species = load_species(species_id)
                registry.register_species(species)

            registry.register_instance(instance_config)

    # Start scheduler
    scheduler = Scheduler(
        harness_config=harness_config,
        registry=registry,
        providers=providers,
        adapters=adapters,
        base_dir=base,
    )
    scheduler.run_forever()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Symbiosis agent framework")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run scheduler")
    run_parser.add_argument(
        "--config", "-c",
        default="config/harness.yaml",
        help="Path to harness config file",
    )
    run_parser.add_argument(
        "--base-dir", "-d",
        default=".",
        help="Base directory for storage and data",
    )
    run_parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
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
    elif raw_args[0] not in {"run", "setup", "-h", "--help"}:
        raw_args = ["run", *raw_args]

    args = parser.parse_args(raw_args)

    if args.command == "setup":
        run_setup(base_dir=args.base_dir)
        return

    _run_scheduler(args)


if __name__ == "__main__":
    main()
