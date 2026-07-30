"""Topology family generators. FAMILIES preserves the doc's presentation order."""

from . import braided, cave, island, organic, pipes, radial, rectilinear, rooms

FAMILIES = {
    "rectilinear": rectilinear,
    "braided": braided,
    "rooms": rooms,
    "organic": organic,
    "cave": cave,
    "radial": radial,
    "island": island,
    "pipes": pipes,
}
