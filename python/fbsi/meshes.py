"""Small test meshes and a neighbour search, used by the examples and tests.

In your own solver you would use the mesh and neighbour lists you already have;
this module only exists so the examples can run without external mesh files.
"""

from __future__ import annotations

from collections import Counter
from itertools import permutations
from typing import List, Sequence, Tuple

import numpy as np

from .cell import Cell, exterior_cell


class Mesh:
    """A list of convex cells built from vertices and cell -> face -> vertex lists."""

    def __init__(self, vertices: np.ndarray, cell_faces: Sequence[Sequence[Sequence[int]]]):
        self.vertices = np.asarray(vertices, dtype=float)
        self.cells: List[Cell] = [Cell.from_polyhedron(self.vertices, faces) for faces in cell_faces]
        self.bbox_min = np.array([cell.bbox_min for cell in self.cells])
        self.bbox_max = np.array([cell.bbox_max for cell in self.cells])
        self.volumes = np.array([cell.volume for cell in self.cells])

        # A face that belongs to only one cell lies on the domain boundary.
        count = Counter(tuple(sorted(face)) for faces in cell_faces for face in faces)
        self.boundary_faces: List[Tuple[List[int], int]] = [
            (list(face), owner)
            for owner, faces in enumerate(cell_faces)
            for face in faces
            if count[tuple(sorted(face))] == 1
        ]

    def __len__(self) -> int:
        return len(self.cells)

    def candidates(self, center: np.ndarray, radius: float) -> np.ndarray:
        """Indices of the cells whose bounding box overlaps the particle's."""
        center = np.asarray(center, dtype=float)
        overlap = np.all(self.bbox_max >= center - radius, axis=1) & np.all(self.bbox_min <= center + radius, axis=1)
        return np.flatnonzero(overlap)

    def locate(self, x: np.ndarray) -> int:
        """Index of the cell that contains point ``x`` (the PCM host cell)."""
        for i in self.candidates(x, 0.0):
            if self.cells[i].contains(x):
                return int(i)
        raise ValueError(f"point {x} is outside the mesh")

    def exterior_cells(self, center: np.ndarray, radius: float) -> List[Cell]:
        """Virtual exterior cells for every boundary face the particle crosses."""
        center = np.asarray(center, dtype=float)
        result = []
        for face, owner in self.boundary_faces:
            polygon = self.vertices[face]
            if np.any(polygon.max(axis=0) < center - radius) or np.any(polygon.min(axis=0) > center + radius):
                continue
            ghost = exterior_cell(polygon, self.cells[owner].centroid)
            # The boundary face is the first face of the virtual cell.
            if abs(ghost.signed_distance(center)[0]) < radius:
                result.append(ghost)
        return result

    def mean_equivalent_radius(self) -> float:
        """Mean radius of the volume-equivalent sphere of the cells."""
        return float(np.mean((3.0 * self.volumes / (4.0 * np.pi)) ** (1.0 / 3.0)))


def _grid_vertices(n: int, lengths: Sequence[float]) -> np.ndarray:
    axis = np.linspace(0.0, 1.0, n + 1)
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    return np.column_stack([x.ravel(), y.ravel(), z.ravel()]) * np.asarray(lengths, dtype=float)


def _vertex_id(n: int, i: int, j: int, k: int) -> int:
    return (i * (n + 1) + j) * (n + 1) + k


def hex_mesh(n: int = 10, lengths: Sequence[float] = (1.0, 1.0, 1.0)) -> Mesh:
    """Structured hexahedral mesh of a box with ``n`` cells per direction."""
    cells = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                v = lambda a, b, c: _vertex_id(n, i + a, j + b, k + c)  # noqa: E731
                cells.append([
                    [v(0, 0, 0), v(0, 1, 0), v(0, 1, 1), v(0, 0, 1)],  # x-
                    [v(1, 0, 0), v(1, 1, 0), v(1, 1, 1), v(1, 0, 1)],  # x+
                    [v(0, 0, 0), v(1, 0, 0), v(1, 0, 1), v(0, 0, 1)],  # y-
                    [v(0, 1, 0), v(1, 1, 0), v(1, 1, 1), v(0, 1, 1)],  # y+
                    [v(0, 0, 0), v(1, 0, 0), v(1, 1, 0), v(0, 1, 0)],  # z-
                    [v(0, 0, 1), v(1, 0, 1), v(1, 1, 1), v(0, 1, 1)],  # z+
                ])
    return Mesh(_grid_vertices(n, lengths), cells)


def tet_mesh(n: int = 10, lengths: Sequence[float] = (1.0, 1.0, 1.0)) -> Mesh:
    """Structured tetrahedral mesh: each hexahedron split into 6 tetrahedra.

    Uses the Kuhn (Freudenthal) split along the main diagonal, which gives
    congruent tetrahedra and a conforming mesh.
    """
    unit = np.eye(3, dtype=int)
    cells = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for order in permutations(range(3)):
                    p = np.array([i, j, k])
                    corners = [p.copy()]
                    for axis in order:
                        p = p + unit[axis]
                        corners.append(p.copy())
                    ids = [_vertex_id(n, *c) for c in corners]
                    cells.append([[ids[0], ids[1], ids[2]], [ids[0], ids[1], ids[3]],
                                  [ids[0], ids[2], ids[3]], [ids[1], ids[2], ids[3]]])
    return Mesh(_grid_vertices(n, lengths), cells)
