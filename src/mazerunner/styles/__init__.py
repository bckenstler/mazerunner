"""Style archetype registry.

Archetypes are decoupled from topology families: any archetype can render any
world it `supports`. `CLASSIC_FOR_FAMILY` preserves the original smoke-set
look (one canonical archetype per family) for `mazes/` regeneration.
"""

from .classic import CLASSIC
from .expanded import EXPANDED

ARCHETYPES = {a.name: a for a in [*CLASSIC, *EXPANDED]}

CLASSIC_FOR_FAMILY = {
    "rectilinear": "notebook",
    "braided": "dungeon-pebble",
    "rooms": "blueprint-rooms",
    "organic": "forest-path",
    "cave": "glow-cavern",
    "radial": "parchment-chart",
    "island": "watercolor-archipelago",
    "pipes": "neon-pipes",
}


def register(archetypes) -> None:
    """Add archetypes to the registry, replacing any of the same name. New
    styles must pass certification on every family they claim to support —
    see tests/test_certify.py."""
    for archetype in archetypes:
        ARCHETYPES[archetype.name] = archetype
