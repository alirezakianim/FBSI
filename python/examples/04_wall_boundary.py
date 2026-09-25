"""Particles touching a wall: virtual exterior cells.

A particle that overlaps a boundary face (wall, inlet, outlet) has part of its
volume outside the fluid domain. FBSI handles this by adding a virtual,
semi-infinite "exterior" cell behind each boundary face the particle crosses.
The exterior cells take part in the normalisation like any other cell, and the
fraction they receive is the volume outside the domain.

    python examples/04_wall_boundary.py
"""

import numpy as np

import fbsi

mesh = fbsi.hex_mesh(n=5)  # unit box; the wall of interest is x = 0
radius = 0.08

print(f"Particle (R = {radius}) moving towards the wall x = 0 of a hexahedral mesh")
print("Fraction of the particle volume INSIDE the domain:\n")
print(f"{'x_p':>6} | {'exact':>7} | {'FBSI + exterior':>15} | {'FBSI, no exterior':>17}")
for x in [0.12, 0.08, 0.06, 0.04, 0.02, 0.0]:
    center = np.array([x, 0.33, 0.47])
    interior = [mesh.cells[i] for i in mesh.candidates(center, radius)]
    exterior = mesh.exterior_cells(center, radius)

    phi = fbsi.fbsi_fractions(center, radius, interior + exterior)
    inside_with = phi[: len(interior)].sum()
    inside_without = fbsi.fbsi_fractions(center, radius, interior).sum()

    h = np.clip(radius + x, 0.0, 2 * radius)  # height of the part inside x > 0
    exact = h * h * (3 * radius - h) / (4 * radius**3)
    print(f"{x:6.2f} | {exact:7.4f} | {inside_with:15.4f} | {inside_without:17.4f}")

print("\nWithout exterior cells the normalisation forces the whole particle into")
print("the domain; with them, the volume outside the wall is correctly excluded.")
