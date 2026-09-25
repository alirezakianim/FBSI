"""Direct-numerical (DN) reference used to validate FBSI.

The particle is voxelised into ``n_per_dim**3`` sub-volumes and every voxel
centre that lies inside the sphere is assigned to the cell that contains it.
The method is accurate but costs O(n_per_dim**3) per particle, so it is meant
for testing, not for use inside a solver.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Sequence

import numpy as np

from .cell import Cell


@lru_cache(maxsize=8)
def _unit_sphere_samples(n_per_dim: int) -> np.ndarray:
    """Voxel centres of a uniform grid on [-1, 1]^3 that lie inside the unit sphere."""
    step = 2.0 / n_per_dim
    axis = -1.0 + step * (np.arange(n_per_dim) + 0.5)
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    points = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    return points[np.einsum("ij,ij->i", points, points) <= 1.0]


def direct_numerical_fractions(center: Sequence[float], radius: float, cells: Sequence[Cell],
                               n_per_dim: int = 64) -> np.ndarray:
    """Fraction of the particle volume that lies inside each cell.

    Each voxel is counted at most once (the first cell that contains it). The
    fractions sum to 1 when the cells cover the whole particle and to less than
    1 when part of the particle is outside all of them.
    """
    center = np.asarray(center, dtype=float)
    samples = center + radius * _unit_sphere_samples(n_per_dim)
    n_total = len(samples)
    fractions = np.zeros(len(cells))
    for i, cell in enumerate(cells):
        if len(samples) == 0:
            break
        if cell.bbox_min is not None and (np.any(center + radius < cell.bbox_min)
                                          or np.any(center - radius > cell.bbox_max)):
            continue  # the particle cannot reach this cell
        inside = np.all(cell.signed_distance(samples) <= 0.0, axis=1)
        fractions[i] = np.count_nonzero(inside) / n_total
        samples = samples[~inside]  # assigned voxels are not tested again
    return fractions
