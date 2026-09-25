"""FBSI quick start: the method on the smallest possible examples.

    python examples/01_quickstart.py
"""

import numpy as np

from fbsi import Cell, fbsi_fractions, gaussian_fractions, isotropic_fractions


def cap_fraction(d, radius):
    """Exact volume fraction of a sphere lying on the inner side (d < 0) of a plane."""
    h = np.clip(radius - np.asarray(d, dtype=float), 0.0, 2.0 * radius)
    return h * h * (3.0 * radius - h) / (4.0 * radius**3)


# ---------------------------------------------------------------------------
# 1. A particle moving across the face shared by two cubic cells
# ---------------------------------------------------------------------------
left = Cell.box(lo=[0, 0, 0], hi=[1, 1, 1])
right = Cell.box(lo=[1, 0, 0], hi=[2, 1, 1])
radius = 0.3

print("1) Particle (R = 0.3) crossing the face x = 1 between two cells")
print(f"{'x_p':>6} | {'FBSI left':>9} {'FBSI right':>10} | {'exact left':>10} {'exact right':>11}")
for x in [0.6, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4]:
    phi = fbsi_fractions([x, 0.5, 0.5], radius, [left, right])
    exact_left = cap_fraction(x - 1.0, radius)
    print(f"{x:6.2f} | {phi[0]:9.4f} {phi[1]:10.4f} | {exact_left:10.4f} {1 - exact_left:11.4f}")
print("   For a single face the smoothstep t^2 (3 - 2t) is exactly the spherical-cap")
print("   volume fraction, so FBSI is exact when a particle crosses one face.")

# ---------------------------------------------------------------------------
# 2. A particle centred on a vertex shared by eight cells
# ---------------------------------------------------------------------------
cells = [Cell.box([i, j, k], [i + 1, j + 1, k + 1]) for i in (-1, 0) for j in (-1, 0) for k in (-1, 0)]
phi = fbsi_fractions([0.0, 0.0, 0.0], 0.3, cells)
print("\n2) Particle centred on a vertex shared by 8 cubes -> 1/8 each:")
print("  ", np.round(phi, 4), " sum =", round(phi.sum(), 12))

# ---------------------------------------------------------------------------
# 3. Anisotropy: a thin, stretched cell
# ---------------------------------------------------------------------------
# Isotropic kernels only see the distance to the cell centroid, so they treat
# every cell as a sphere. FBSI uses the faces and sees the real cell shape.
thin = Cell.box([0, 0, 0], [4, 4, 0.25])
large = Cell.box([0, 0, 0.25], [4, 4, 4.25])
center, r = [2.0, 2.0, 0.2], 0.2
exact = cap_fraction(0.2 - 0.25, r)
print("\n3) Particle (R = 0.2) in a thin, 0.25-thick cell below a large cell")
print(f"   {'method':<10} {'thin cell':>9} {'large cell':>10}")
print(f"   {'exact':<10} {exact:9.4f} {1 - exact:10.4f}")
for name, method in [("FBSI", fbsi_fractions), ("Isotropic", isotropic_fractions), ("Gaussian", gaussian_fractions)]:
    phi = method(center, r, [thin, large])
    print(f"   {name:<10} {phi[0]:9.4f} {phi[1]:10.4f}")
