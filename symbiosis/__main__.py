"""CLI entry point — `python -m symbiosis` or `symbiosis run`."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from symbiosis.harness.config import load_harness_config, load_instance_config
from symbiosis.harness.registry import Registry
from symbiosis.harness.scheduler import Scheduler, build_providers, build_adapters


def load_species(species_id: str):
    """Load a species by ID. Returns a Species instance."""
    if species_id == "draum":
        from symbiosis.species.draum import DraumSpecies
        return DraumSpecies()
    raise ValueError(f"Unknown species: {species_id}")


def main():
    parser = argparse.ArgumentParser(description="Symbiosis agent framework")
    parser.add_argument(
        "--config", "-c",
        default="config/harness.yaml",
        help="Path to harness config file",
    )
    parser.add_argument(
        "--base-dir", "-d",
        default=".",
        help="Base directory for storage and data",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

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


if __name__ == "__main__":
    main()
