"""Face-Based Soft-Intersection (FBSI) particle-to-cell volume mapping.

Reference implementation accompanying

    A. Kianimoqadam, A. J. Schrader, "An anisotropic, Face-Based Soft-Intersection
    method for robust Lagrangian-Eulerian coupling", Computer Methods in Applied
    Mechanics and Engineering 461 (2026) 119161. doi:10.1016/j.cma.2026.119161

If you use this code, please cite this paper.
"""

from .baselines import gaussian_fractions, isotropic_fractions, pcm_fractions
from .cell import Cell, exterior_cell
from .core import face_weights, fbsi_fractions, shape_factor, solid_volume_fraction
from .meshes import Mesh, hex_mesh, tet_mesh
from .reference import direct_numerical_fractions

__version__ = "1.0.1"

__all__ = [
    "Cell",
    "exterior_cell",
    "face_weights",
    "shape_factor",
    "fbsi_fractions",
    "solid_volume_fraction",
    "direct_numerical_fractions",
    "pcm_fractions",
    "isotropic_fractions",
    "gaussian_fractions",
    "Mesh",
    "hex_mesh",
    "tet_mesh",
]
