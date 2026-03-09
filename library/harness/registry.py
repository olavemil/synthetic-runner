"""Registry — species and instance registration, handler lookup."""

from __future__ import annotations

from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from library.harness.config import InstanceConfig
    from library.species import Species, SpeciesManifest


class Registry:
    def __init__(self):
        self._species: dict[str, SpeciesManifest] = {}
        self._instances: dict[str, InstanceConfig] = {}

    def register_species(self, species: Species) -> None:
        manifest = species.manifest()
        self._species[manifest.species_id] = manifest

    def register_species_manifest(self, manifest: SpeciesManifest) -> None:
        self._species[manifest.species_id] = manifest

    def register_instance(self, instance_config: InstanceConfig) -> None:
        self._instances[instance_config.instance_id] = instance_config

    def get_manifest(self, species_id: str) -> SpeciesManifest:
        if species_id not in self._species:
            raise KeyError(f"Species '{species_id}' not registered")
        return self._species[species_id]

    def get_instance_config(self, instance_id: str) -> InstanceConfig:
        if instance_id not in self._instances:
            raise KeyError(f"Instance '{instance_id}' not registered")
        return self._instances[instance_id]

    def get_handler(self, instance_id: str, entry_point_name: str) -> Callable:
        config = self.get_instance_config(instance_id)
        manifest = self.get_manifest(config.species)
        for ep in manifest.entry_points:
            if ep.name == entry_point_name:
                return ep.handler
        raise KeyError(
            f"Entry point '{entry_point_name}' not found for species '{config.species}'"
        )

    def list_instances(self) -> list[InstanceConfig]:
        return list(self._instances.values())

    def list_species(self) -> list[str]:
        return list(self._species.keys())
