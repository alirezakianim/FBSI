"""Compare FBSI with common methods against the direct-numerical reference.

A small version of the benchmark in the paper: random particles in a box
meshed with hexahedra or tetrahedra, with particle radius
R_p = beta * (mean equivalent-sphere radius of the cells).

    python examples/02_mesh_benchmark.py                    # hexahedral mesh
    python examples/02_mesh_benchmark.py --mesh tet         # tetrahedral mesh
    python examples/02_mesh_benchmark.py --aspect 2.0       # stretched cells
    python examples/02_mesh_benchmark.py --plot             # save a parity plot
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


def r_squared(truth, pred):
    return 1.0 - np.sum((truth - pred) ** 2) / np.sum((truth - truth.mean()) ** 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mesh", choices=["hex", "tet"], default="hex")
    parser.add_argument("--n", type=int, default=10, help="cells per direction (default 10)")
    parser.add_argument("--aspect", type=float, default=1.0, help="z-length of the box; stretches the cells")
    parser.add_argument("--beta", type=float, default=0.8, help="particle-to-cell size ratio (default 0.8)")
    parser.add_argument("--particles", type=int, default=400)
    parser.add_argument("--nv", type=int, default=64, help="voxels per direction for the reference")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--plot", action="store_true", help="save benchmark_parity.png (needs matplotlib)")
    args = parser.parse_args()

    lengths = np.array([1.0, 1.0, args.aspect])
    build = fbsi.hex_mesh if args.mesh == "hex" else fbsi.tet_mesh
    mesh = build(args.n, lengths)
    radius = args.beta * mesh.mean_equivalent_radius()
    rng = np.random.default_rng(args.seed)
    # Particles stay fully inside the box (see 04_wall_boundary.py for walls).
    centers = rng.uniform(radius, lengths - radius, size=(args.particles, 3))

    # The isotropic kernels give weight to every cell whose centroid lies within
    # r_eq + 2 R_p, so all methods get this wider neighbourhood. FBSI and the
    # reference simply return 0 for cells the particle does not touch.
    reach = 2.0 * radius + 2.0 * mesh.mean_equivalent_radius()

    print(f"{args.mesh} mesh, {len(mesh)} cells, box {lengths.tolist()}, beta = {args.beta}, "
          f"R_p = {radius:.4f}, {args.particles} particles")

    truth, pred = [], {name: [] for name in METHODS}
    for center in centers:
        candidates = [mesh.cells[i] for i in mesh.candidates(center, reach)]
        reference = fbsi.direct_numerical_fractions(center, radius, candidates, args.nv)
        results = {name: method(center, radius, candidates) for name, method in METHODS.items()}
        # Same pair selection as the paper: keep the cells that receive volume
        # from the reference or from FBSI.
        keep = (reference > 1e-5) | (results["FBSI"] > 1e-5)
        truth.append(reference[keep])
        for name in METHODS:
            pred[name].append(results[name][keep])

    truth = np.concatenate(truth)
    print(f"\n{len(truth)} particle-cell pairs compared with the direct-numerical reference (N_V = {args.nv})\n")
    print(f"{'method':<10} {'R^2':>8} {'Pearson':>8} {'mean |err|':>11} {'max |err|':>10}")
    for name in METHODS:
        p = np.concatenate(pred[name])
        err = np.abs(p - truth)
        print(f"{name:<10} {r_squared(truth, p):8.4f} {np.corrcoef(truth, p)[0, 1]:8.4f} "
              f"{err.mean():11.2e} {err.max():10.3f}")

    if args.plot:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, len(METHODS), figsize=(3.2 * len(METHODS), 3.2), sharey=True)
        for ax, name in zip(axes, METHODS):
            ax.scatter(truth, np.concatenate(pred[name]), s=4, alpha=0.4)
            ax.plot([0, 1], [0, 1], "k--", lw=1)
            ax.set_title(name)
            ax.set_xlabel(r"$\phi_{p,c}$ reference")
            ax.set_aspect("equal")
        axes[0].set_ylabel(r"$\phi_{p,c}$ method")
        fig.tight_layout()
        fig.savefig("benchmark_parity.png", dpi=150)
        print("\nSaved benchmark_parity.png")


if __name__ == "__main__":
    main()
