"""Thrivemind species — hivemind profile built on Draum pipeline handlers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from symbiosis.species import Species, SpeciesManifest, EntryPoint
from symbiosis.species.draum import (
    on_message as draum_on_message,
    heartbeat as draum_heartbeat,
    DEFAULT_FILES as DRAUM_DEFAULT_FILES,
)

if TYPE_CHECKING:
    from symbiosis.harness.context import InstanceContext


class ThrivemindSpecies(Species):
    def manifest(self) -> SpeciesManifest:
        return SpeciesManifest(
            species_id="thrivemind",
            entry_points=[
                EntryPoint(
                    name="on_message",
                    handler=draum_on_message,
                    trigger="message",
                ),
                EntryPoint(
                    name="heartbeat",
                    handler=draum_heartbeat,
                    schedule="*/15 * * * *",
                ),
            ],
            tools=[],
            default_files=DRAUM_DEFAULT_FILES,
            spawn=self._spawn,
        )

    def _spawn(self, ctx: InstanceContext) -> None:
        for path, content in DRAUM_DEFAULT_FILES.items():
            if not ctx.exists(path):
                ctx.write(path, content)
