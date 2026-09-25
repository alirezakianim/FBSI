"""Smoothness test: move one particle through a tetrahedral mesh.

Tracks the fraction of the particle assigned to one cell while the particle
travels along a straight line. PCM jumps between 0 and 1; FBSI changes smoothly
and follows the direct-numerical reference.

    python examples/03_particle_translation.py          # prints a table
    python examples/03_particle_translation.py --plot   # also saves translation.png
"""

import argparse

import numpy as np

import fbsi

METHODS = {
    "FBSI": fbsi.fbsi_fractions,
    "Isotropic": fbsi.isotropic_fractions,
    "Gaussian": fbsi.gaussian_fractions,
    "PCM": fbsi.pcm_fractions,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--beta", type=float, default=0.8)
    parser.add_argument("--steps", type=int, default=121)
    parser.add_argument("--plot", action="store_true", help="save translation.png (needs matplotlib)")
    args = parser.parse_args()

    mesh = fbsi.tet_mesh(n=4)
    radius = args.beta * mesh.mean_equivalent_radius()
    # Follow one tetrahedron; the path passes through its centroid along x.
    target = mesh.locate([0.5, 0.4, 0.45])
    centroid = mesh.cells[target].centroid
    start, end = centroid - [0.15, 0.0, 0.0], centroid + [0.15, 0.0, 0.0]
    reach = 2.0 * radius + 2.0 * mesh.mean_equivalent_radius()  # covers the kernels' support

    s = np.linspace(0.0, 1.0, args.steps)
    curves = {name: [] for name in ["Reference", *METHODS]}
    for t in s:
        center = start + t * (end - start)
        index = mesh.candidates(center, reach)
        candidates = [mesh.cells[i] for i in index]
        slot = np.flatnonzero(index == target)
        phi = {"Reference": fbsi.direct_numerical_fractions(center, radius, candidates, n_per_dim=96)}
        phi.update({name: method(center, radius, candidates) for name, method in METHODS.items()})
        for name, values in phi.items():
            curves[name].append(values[slot[0]] if len(slot) else 0.0)

    print(f"Fraction of the particle in cell {target} along the path (R_p = {radius:.4f})\n")
    print(f"{'s':>5} " + " ".join(f"{name:>9}" for name in curves))
    for k in range(0, args.steps, max(1, args.steps // 20)):
        print(f"{s[k]:5.2f} " + " ".join(f"{curves[name][k]:9.4f}" for name in curves))

    reference = np.array(curves["Reference"])
    print("\nmean |error| along the path:")
    for name in METHODS:
        print(f"  {name:<10} {np.mean(np.abs(np.array(curves[name]) - reference)):.2e}")

    if args.plot:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot(s, reference, "k-", lw=2.5, label="Direct numerical (reference)")
        for name, style in zip(METHODS, ["-", "--", "-.", ":"]):
            ax.plot(s, curves[name], style, lw=1.5, label=name)
        ax.set_xlabel("position along the path")
        ax.set_ylabel(r"$\phi_{p,c}$ of the tracked cell")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig("translation.png", dpi=150)
        print("\nSaved translation.png")


if __name__ == "__main__":
    main()
