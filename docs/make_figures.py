"""Regenerate the figures in docs/images from the Python reference implementation.

    pip install -e python[examples]
    python docs/make_figures.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, Polygon

import fbsi

OUT = Path(__file__).parent / "images"
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
COLORS = {"FBSI": "#2a78d6", "Isotropic": "#eb6834", "Gaussian": "#1baf7a", "PCM": "#eda100"}
STYLES = {"FBSI": "-", "Isotropic": "--", "Gaussian": "-.", "PCM": ":"}
METHODS = {"FBSI": fbsi.fbsi_fractions, "Isotropic": fbsi.isotropic_fractions,
           "Gaussian": fbsi.gaussian_fractions, "PCM": fbsi.pcm_fractions}

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": INK_2, "axes.labelcolor": INK, "xtick.color": INK_2,
    "ytick.color": INK_2, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "savefig.dpi": 200, "savefig.bbox": "tight",
})


def concept():
    """(a) Soft half-spaces of a 2-D cell; (b) the face weight is the spherical-cap fraction."""
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.5, 3.9), gridspec_kw={"width_ratios": [1.05, 1]})

    # (a) a skewed quadrilateral cell, one face highlighted with its +-R band
    cell = np.array([[0.0, 0.0], [2.2, 0.35], [2.6, 2.1], [0.3, 1.8]])
    ax.add_patch(Polygon(cell, closed=True, facecolor="#cde2fb", edgecolor=INK, lw=1.5))
    a, b = cell[1], cell[2]
    tangent = (b - a) / np.linalg.norm(b - a)
    normal = np.array([tangent[1], -tangent[0]])  # outward for this counter-clockwise cell
    radius = 0.45
    label_box = dict(facecolor="white", edgecolor="none", pad=1.5)
    for offset, label in [(-radius, r"$d=-R_p$"), (radius, r"$d=+R_p$")]:
        p, q = a + offset * normal - 0.35 * tangent, b + offset * normal + 0.35 * tangent
        ax.plot([p[0], q[0]], [p[1], q[1]], "--", color=INK_2, lw=1)
        ax.text(q[0], q[1] + 0.08, label, ha="center", va="bottom", color=INK_2, fontsize=9, bbox=label_box)
    top = a + 0.82 * (b - a)
    ax.add_patch(FancyArrowPatch(top, top + 0.6 * normal, arrowstyle="-|>", mutation_scale=12, color=INK, lw=1.3))
    ax.text(*(top + 0.75 * normal), r"$\vec n_f$", ha="center", va="center", color=INK)

    on_face = a + 0.3 * (b - a)
    for center, label in [((1.0, 0.9), r"$W=1$"), (tuple(on_face), r"$0<W<1$"), ((3.75, 0.85), r"$W=0$")]:
        ax.add_patch(Circle(center, 0.3, facecolor="#eb6834", alpha=0.85, edgecolor=INK, lw=1))
        ax.text(center[0], center[1] - 0.42, label, ha="center", va="top", color=INK, fontsize=9, bbox=label_box)
    ax.text(0.45, 1.45, "cell $c$", color=INK)
    ax.set_xlim(-0.3, 4.3)
    ax.set_ylim(-0.6, 2.9)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("(a) each face is a soft half-space", loc="left", color=INK)

    # (b) smoothstep face weight vs exact cap fraction
    d = np.linspace(-1.4, 1.4, 400)
    bx.plot(d, fbsi.face_weights(d, 1.0), color=COLORS["FBSI"], lw=2, label=r"FBSI $W_{p,f}=t^2(3-2t)$")
    bx.plot(d, fbsi.face_weights(d, 1.0, smooth=False), "--", color=INK_2, lw=1.2, label="linear ramp $W=t$")
    dots = np.linspace(-1.0, 1.0, 11)
    h = 1.0 - dots
    bx.plot(dots, h * h * (3 - h) / 4, "o", ms=6, mfc="white", mec=INK, label="exact sphere-cap fraction")
    bx.axvspan(-1, 1, color=GRID, alpha=0.35, lw=0)
    bx.set_xlabel(r"signed distance $d_{p,f}/R_p$  (negative = inside)")
    bx.set_ylabel(r"face weight $W_{p,f}$")
    bx.grid(True, color=GRID, lw=0.6)
    bx.legend(loc="upper right", fontsize=8.5)
    bx.set_title("(b) the weight equals the cap volume fraction", loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "concept.png")
    plt.close(fig)


def parity(mesh_type="tet", n_particles=400, seed=1):
    mesh = fbsi.tet_mesh(10) if mesh_type == "tet" else fbsi.hex_mesh(10)
    radius = 0.8 * mesh.mean_equivalent_radius()
    reach = 2.0 * radius + 2.0 * mesh.mean_equivalent_radius()  # covers the kernels' support
    rng = np.random.default_rng(seed)
    truth, pred = [], {m: [] for m in METHODS}
    for center in rng.uniform(radius, 1 - radius, size=(n_particles, 3)):
        cells = [mesh.cells[i] for i in mesh.candidates(center, reach)]
        ref = fbsi.direct_numerical_fractions(center, radius, cells, 64)
        res = {m: f(center, radius, cells) for m, f in METHODS.items()}
        keep = (ref > 1e-5) | (res["FBSI"] > 1e-5)
        truth.append(ref[keep])
        for m in METHODS:
            pred[m].append(res[m][keep])
    truth = np.concatenate(truth)

    fig, axes = plt.subplots(1, 4, figsize=(11, 3.1), sharey=True)
    for ax, m in zip(axes, METHODS):
        p = np.concatenate(pred[m])
        r2 = 1 - np.sum((truth - p) ** 2) / np.sum((truth - truth.mean()) ** 2)
        ax.plot([0, 1], [0, 1], color=INK_2, lw=1, ls="--")
        ax.scatter(truth, p, s=8, color=COLORS[m], alpha=0.35, edgecolors="none")
        ax.set_title(f"{m}   $R^2$ = {r2:.3f}", color=INK, fontsize=10)
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.03, 1.03)
        ax.set_aspect("equal")
        ax.set_xlabel(r"reference $\phi_{p,c}$")
        ax.grid(True, color=GRID, lw=0.6)
    axes[0].set_ylabel(r"predicted $\phi_{p,c}$")
    fig.tight_layout()
    fig.savefig(OUT / f"parity_{mesh_type}.png")
    plt.close(fig)


def translation(steps=161):
    mesh = fbsi.tet_mesh(4)
    radius = 0.8 * mesh.mean_equivalent_radius()
    target = mesh.locate([0.5, 0.4, 0.45])
    centroid = mesh.cells[target].centroid
    start, end = centroid - [0.15, 0, 0], centroid + [0.15, 0, 0]
    reach = 2.0 * radius + 2.0 * mesh.mean_equivalent_radius()
    s = np.linspace(0, 1, steps)
    curves = {k: [] for k in ["Reference", *METHODS]}
    for t in s:
        center = start + t * (end - start)
        index = mesh.candidates(center, reach)
        cells = [mesh.cells[i] for i in index]
        slot = np.flatnonzero(index == target)
        phi = {"Reference": fbsi.direct_numerical_fractions(center, radius, cells, 96)}
        phi.update({m: f(center, radius, cells) for m, f in METHODS.items()})
        for k, v in phi.items():
            curves[k].append(v[slot[0]] if len(slot) else 0.0)

    fig, ax = plt.subplots(figsize=(6.4, 3.5))
    ax.plot(s, curves["Reference"], color=INK, lw=3.5, alpha=0.25, label="direct-numerical reference")
    for m in METHODS:
        ax.plot(s, curves[m], STYLES[m], color=COLORS[m], lw=2, label=m)
    ax.set_xlabel("particle position along the path")
    ax.set_ylabel(r"$\phi_{p,c}$ of the tracked cell")
    ax.grid(True, color=GRID, lw=0.6)
    ax.legend(fontsize=8.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "translation.png")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    concept()
    parity("tet")
    translation()
    print("figures written to", OUT)
