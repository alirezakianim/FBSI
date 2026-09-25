"""Geometric description of a convex computational cell.

FBSI needs only two things for every face of a cell:

* the outward unit normal ``n_f``, and
* one point ``x_f`` on the face plane (the face centroid is the natural choice).

Everything else stored here (volume, centroid, bounding box) is optional. The
bounding box is used for a cheap rejection test, and the volume and centroid
are used only by the comparison methods in :mod:`fbsi.baselines`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


def _newell_normal(polygon: np.ndarray) -> np.ndarray:
    """Area-weighted normal of a planar polygon (Newell's method).

    The direction follows the vertex order (right-hand rule); callers orient it.
    """
    center = polygon.mean(axis=0)
    normal = np.zeros(3)
    for i in range(len(polygon)):
        a = polygon[i] - center
        b = polygon[(i + 1) % len(polygon)] - center
        normal += np.cross(a, b)
    return normal


@dataclass
class Cell:
    """A convex cell described as the intersection of its face half-spaces.

    Parameters
    ----------
    normals : (N_f, 3) array
        Outward face normals. They are normalised to unit length on creation.
    points : (N_f, 3) array
        One point on each face plane (usually the face centroid).
    centroid, volume : optional
        Cell centroid and volume (needed only by the isotropic comparison kernels).
    bbox_min, bbox_max : optional
        Axis-aligned bounding box, used to skip cells the particle cannot touch.
    exterior : bool
        True for a virtual cell outside the fluid domain (see :func:`exterior_cell`).
    """

    normals: np.ndarray
    points: np.ndarray
    centroid: Optional[np.ndarray] = None
    volume: Optional[float] = None
    bbox_min: Optional[np.ndarray] = None
    bbox_max: Optional[np.ndarray] = None
    exterior: bool = False

    def __post_init__(self) -> None:
        self.normals = np.asarray(self.normals, dtype=float).reshape(-1, 3)
        self.points = np.asarray(self.points, dtype=float).reshape(-1, 3)
        if self.normals.shape != self.points.shape:
            raise ValueError("normals and points must have the same shape (N_f, 3)")
        lengths = np.linalg.norm(self.normals, axis=1)
        if np.any(lengths == 0.0):
            raise ValueError("face normals must be non-zero")
        self.normals = self.normals / lengths[:, None]
        # n_f . x_f for every face, so that d = n_f . x - offset_f
        self.offsets = np.einsum("ij,ij->i", self.normals, self.points)

    @property
    def n_faces(self) -> int:
        return len(self.normals)

    def signed_distance(self, x: np.ndarray) -> np.ndarray:
        """Signed distance ``d_f = n_f . (x - x_f)`` from point(s) ``x`` to every face.

        ``d_f < 0`` means ``x`` is on the inside of face ``f``. For ``x`` of shape
        (3,) the result has shape (N_f,); for (M, 3) it has shape (M, N_f).
        """
        return np.asarray(x, dtype=float) @ self.normals.T - self.offsets

    def contains(self, x: np.ndarray) -> bool:
        """Boolean indicator I_c(x): True if ``x`` lies inside (or on) every face."""
        return bool(np.all(self.signed_distance(x) <= 0.0))

    # ------------------------------------------------------------------ builders
    @classmethod
    def from_polyhedron(cls, vertices: np.ndarray, faces: Sequence[Sequence[int]]) -> "Cell":
        """Build a cell from vertex coordinates and face -> vertex index lists.

        This is the connectivity every finite-volume mesh already stores. The
        vertices of each face must be ordered around the face, in either
        direction; normals are oriented outward automatically.
        """
        vertices = np.asarray(vertices, dtype=float)
        used = sorted({i for face in faces for i in face})
        cell_vertices = vertices[used]
        inside = cell_vertices.mean(axis=0)  # interior point of a convex cell

        normals, points = [], []
        volume, moment = 0.0, np.zeros(3)
        for face in faces:
            polygon = vertices[list(face)]
            center = polygon.mean(axis=0)
            normal = _newell_normal(polygon)
            if np.dot(normal, center - inside) < 0.0:
                normal = -normal
            normals.append(normal)
            points.append(center)
            # Volume and centroid from a fan of tetrahedra (inside, a, b, center).
            for i in range(len(polygon)):
                a, b = polygon[i], polygon[(i + 1) % len(polygon)]
                tet_volume = abs(np.dot(a - inside, np.cross(b - inside, center - inside))) / 6.0
                volume += tet_volume
                moment += tet_volume * (inside + a + b + center) / 4.0

        return cls(
            normals=np.array(normals),
            points=np.array(points),
            centroid=moment / volume,
            volume=volume,
            bbox_min=cell_vertices.min(axis=0),
            bbox_max=cell_vertices.max(axis=0),
        )

    @classmethod
    def box(cls, lo: Sequence[float], hi: Sequence[float]) -> "Cell":
        """Axis-aligned hexahedron with opposite corners ``lo`` and ``hi``."""
        (x0, y0, z0), (x1, y1, z1) = lo, hi
        vertices = np.array([
            [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
            [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
        ])
        faces = [[0, 3, 7, 4], [1, 2, 6, 5], [0, 1, 5, 4],
                 [3, 2, 6, 7], [0, 1, 2, 3], [4, 5, 6, 7]]
        return cls.from_polyhedron(vertices, faces)

    @classmethod
    def tetrahedron(cls, v0, v1, v2, v3) -> "Cell":
        """Tetrahedron from its four vertices (any order)."""
        vertices = np.array([v0, v1, v2, v3], dtype=float)
        faces = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
        return cls.from_polyhedron(vertices, faces)


def exterior_cell(face_vertices: np.ndarray, interior_point: np.ndarray) -> Cell:
    """Virtual, semi-infinite control volume on the outside of a boundary face.

    Used to treat particles that overlap a wall, inlet or outlet. The virtual
    cell is bounded by the boundary face itself plus one side face per edge,
    with side normal ``n_side ~ n_b x e_m``. Include it in the same list of
    candidate cells as the interior cells: its FBSI fraction is the part of the
    particle that lies outside the domain.

    Parameters
    ----------
    face_vertices : (m, 3) array
        Vertices of the boundary face, ordered around the face.
    interior_point : (3,) array
        Any point inside the domain, e.g. the centroid of the cell that owns the
        face. It is used only to decide which side is "outside".
    """
    polygon = np.asarray(face_vertices, dtype=float)
    center = polygon.mean(axis=0)
    n_b = _newell_normal(polygon)
    n_b /= np.linalg.norm(n_b)
    if np.dot(n_b, center - np.asarray(interior_point, dtype=float)) < 0.0:
        n_b = -n_b  # n_b now points out of the domain

    # The boundary face bounds the virtual cell; seen from the virtual cell its
    # outward normal points back into the domain.
    normals, points = [-n_b], [center]
    for i in range(len(polygon)):
        a, b = polygon[i], polygon[(i + 1) % len(polygon)]
        n_side = np.cross(n_b, b - a)
        midpoint = 0.5 * (a + b)
        if np.dot(n_side, midpoint - center) < 0.0:
            n_side = -n_side
        normals.append(n_side)
        points.append(midpoint)

    return Cell(normals=np.array(normals), points=np.array(points), exterior=True)
