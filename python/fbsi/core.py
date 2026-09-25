"""Face-Based Soft-Intersection (FBSI): the method itself.

Notation follows Kianimoqadam & Schrader, CMAME 461 (2026) 119161:

    d_pf   = n_f . (x_p - x_f)              signed particle-face distance (< 0 inside)
    xi_pf  = clamp(-d_pf / R_p, -1, 1)      transition coordinate
    t_pf   = (xi_pf + 1) / 2
    W_pf   = t_pf^2 (3 - 2 t_pf)            soft half-space (cubic smoothstep)
    S_pc   = prod_f W_pf                    anisotropic shape factor of cell c
    phi_pc = S_pc / sum_k S_pk              fraction of particle p assigned to cell c

The method has no tuning parameter: it uses only the face geometry and R_p.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .cell import Cell


def face_weights(d: np.ndarray, radius: float, smooth: bool = True) -> np.ndarray:
    """Soft half-space weight W_pf for signed distance(s) ``d``.

    Returns 1 when the particle is fully inside the face (d <= -R), 0 when it is
    fully outside (d >= R), and a smooth value in between. ``smooth=False``
    gives the linear ramp W = t (only C0-continuous; shown for comparison).
    """
    xi = np.clip(-np.asarray(d, dtype=float) / radius, -1.0, 1.0)
    t = 0.5 * (xi + 1.0)
    if smooth:
        return t * t * (3.0 - 2.0 * t)
    return t


def shape_factor(cell: Cell, center: np.ndarray, radius: float, smooth: bool = True) -> float:
    """Anisotropic shape factor S_pc: the product of the face weights of ``cell``."""
    d = cell.signed_distance(center)
    if np.any(d >= radius):
        return 0.0  # the particle is completely outside one face plane (early exit)
    return float(np.prod(face_weights(d, radius, smooth)))


def _bbox_overlaps(cell: Cell, center: np.ndarray, radius: float) -> bool:
    """Cheap rejection: can the particle's bounding box touch the cell's?"""
    if cell.bbox_min is None or cell.bbox_max is None:
        return True
    return bool(np.all(center + radius >= cell.bbox_min) and np.all(center - radius <= cell.bbox_max))


def fbsi_fractions(center: Sequence[float], radius: float, cells: Sequence[Cell],
                   smooth: bool = True) -> np.ndarray:
    """Fraction phi_pc of the particle volume assigned to each candidate cell.

    Parameters
    ----------
    center, radius
        Particle centre x_p and radius R_p.
    cells
        Candidate cells: the cell containing the particle centre plus its
        neighbours (any cell the particle may overlap). Virtual exterior cells
        from :func:`fbsi.exterior_cell` may be included to handle boundaries.
    smooth
        Use the cubic smoothstep (default, C1-continuous) or a linear ramp.

    Returns
    -------
    numpy.ndarray
        ``phi[i]`` for ``cells[i]``. The values are non-negative and sum to 1.
    """
    center = np.asarray(center, dtype=float)
    weights = np.array([
        shape_factor(cell, center, radius, smooth) if _bbox_overlaps(cell, center, radius) else 0.0
        for cell in cells
    ])
    total = weights.sum()
    if total <= 0.0:
        raise ValueError(
            "The particle does not overlap any candidate cell. The candidate list "
            "must include the cell that contains the particle centre."
        )
    return weights / total


def solid_volume_fraction(fractions_per_particle, particle_volumes, cell_volumes) -> np.ndarray:
    """Accumulate the cell solid volume fraction Phi_c = sum_p phi_pc V_p / V_c.

    Parameters
    ----------
    fractions_per_particle
        Iterable of ``(cell_indices, phi)`` pairs, one per particle, as produced by
        calling :func:`fbsi_fractions` on each particle's candidate cells.
    particle_volumes
        V_p for each particle, in the same order.
    cell_volumes
        V_c for every cell of the mesh.
    """
    cell_volumes = np.asarray(cell_volumes, dtype=float)
    solid = np.zeros_like(cell_volumes)
    for (indices, phi), v_p in zip(fractions_per_particle, particle_volumes):
        np.add.at(solid, np.asarray(indices), np.asarray(phi) * v_p)
    return solid / cell_volumes
