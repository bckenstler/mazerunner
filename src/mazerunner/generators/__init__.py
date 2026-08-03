"""Topology family generators.

Every family exposes the same entry point:

    build(seed: int, overrides: dict | None = None) -> World

`seed` alone determines the maze — a family is a pure function of it, which is
what lets any task be rebuilt from provenance and checked byte-identical.
`overrides` merges over the family's defaults (`o = {defaults, **(overrides or
{})}` in every build) and is how the dataset sampler reaches difficulty knobs
like grid size, corridor width, and loop count without the families knowing
anything about tiers.

Each build carves a candidate adjacency, picks far-apart endpoints, retains a
route between them (see common.py), and returns a World whose graph and mask
agree. The generator never certifies its own output: validate_world and the
render certification are separate and fail closed.

FAMILIES is ordered for presentation, so figures and contact sheets list the
families the same way everywhere.
"""

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
