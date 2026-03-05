"""Species (Layer 2) — stateless behavior code with manifest-based registration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from symbiosis.harness.context import InstanceContext


@dataclass
class EntryPoint:
    name: str
    handler: Callable
    schedule: str | None = None
    trigger: str | None = None


@dataclass
class ToolDef:
    name: str
    schema: dict
    handler: Callable


@dataclass
class SpeciesManifest:
    species_id: str
    entry_points: list[EntryPoint] = field(default_factory=list)
    tools: list[ToolDef] = field(default_factory=list)
    default_files: dict[str, str] = field(default_factory=dict)
    spawn: Callable | None = None


class Species:
    def manifest(self) -> SpeciesManifest:
        raise NotImplementedError
