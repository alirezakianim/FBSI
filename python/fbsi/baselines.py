"""Common methods FBSI is compared against in the paper (for comparison only).

* PCM: the whole particle goes to the cell containing its centre.
* Isotropic polynomial kernel: W = (1 - r/h)^3 with h = (3 V_c / 4 pi)^(1/3) + 2 R_p.
* Gaussian kernel: W = exp(-r^2 / 2 sigma^2) with sigma = 2 R_p / (2 sqrt(2 ln 2)).

Here r is the distance between the particle centre and the cell centroid, so the
isotropic kernels see every cell as a sphere, whatever its real shape. The
kernel weights are normalised over the candidate cells, as in FBSI.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .cell import Cell


def _normalise(weights: np.ndarray, center: np.ndarray, cells: Sequence[Cell]) -> np.ndarray:
    total = weights.sum()
    if total > 0.0:
        return weights / total
    return pcm_fractions(center, 0.0, cells)


def _centroid_distances(center: np.ndarray, cells: Sequence[Cell]) -> np.ndarray:
    centroids = np.array([cell.centroid for cell in cells])
    return np.linalg.norm(centroids - center, axis=1)


def pcm_fractions(center: Sequence[float], radius: float, cells: Sequence[Cell]) -> np.ndarray:
    """Point-Centroid Method: phi = 1 for the cell containing the centre, 0 elsewhere."""
    center = np.asarray(center, dtype=float)
    fractions = np.zeros(len(cells))
    for i, cell in enumerate(cells):
        if not cell.exterior and cell.contains(center):
            fractions[i] = 1.0
            return fractions
    raise ValueError("no candidate cell contains the particle centre")


def isotropic_fractions(center: Sequence[float], radius: float, cells: Sequence[Cell]) -> np.ndarray:
    """Isotropic cubic polynomial kernel (SPH / porous-cube type)."""
    center = np.asarray(center, dtype=float)
    r = _centroid_distances(center, cells)
    volumes = np.array([cell.volume for cell in cells])
    h = (3.0 * volumes / (4.0 * np.pi)) ** (1.0 / 3.0) + 2.0 * radius
    weights = np.where(r < h, (1.0 - r / h) ** 3, 0.0)
    return _normalise(weights, center, cells)


def gaussian_fractions(center: Sequence[float], radius: float, cells: Sequence[Cell]) -> np.ndarray:
    """Gaussian kernel with full-width at half-maximum equal to the particle diameter."""
    center = np.asarray(center, dtype=float)
    r = _centroid_distances(center, cells)
    sigma = 2.0 * radius / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    weights = np.exp(-r * r / (2.0 * sigma * sigma))
    return _normalise(weights, center, cells)
