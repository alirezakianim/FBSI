# The FBSI method

This page explains the Face-Based Soft-Intersection (FBSI) method step by step.
The full derivation, benchmarks and discussion are in the paper:

> A. Kianimoqadam, A. J. Schrader, *An anisotropic, Face-Based Soft-Intersection method for robust
> Lagrangian–Eulerian coupling*, Computer Methods in Applied Mechanics and Engineering **461** (2026) 119161.
> [doi:10.1016/j.cma.2026.119161](https://doi.org/10.1016/j.cma.2026.119161)

## 1. The problem

In Lagrangian–Eulerian simulations (for example CFD–DEM), particles move through a fixed fluid mesh.
The fluid solver needs the **solid volume fraction** of every cell $c$:

```math
\Phi_c = \frac{1}{V_c} \sum_{p} \phi_{p,c}\, V_p , \qquad 0 \le \phi_{p,c} \le 1
```

where $V_p$ is the particle volume, $V_c$ the cell volume, and $\phi_{p,c}$ the **fraction of particle
$p$ that lies in cell $c$**. Everything comes down to computing $\phi_{p,c}$ well.

| Approach | How $\phi_{p,c}$ is obtained | Trade-off |
|---|---|---|
| Direct numerical (DN) | Voxelise the particle and test each voxel against the cell | Accurate, but cost grows as $N_V^3$ per particle |
| Point-Centroid Method (PCM) | Whole particle goes to the cell containing its centre | Very cheap, but $\phi$ jumps between 0 and 1 |
| Isotropic kernels (Gaussian, polynomial, GIM) | Weight depends only on the distance between the particle and the cell **centroid** | Smooth, but every cell is treated as a sphere; errors on stretched, skewed or unstructured cells, and a smoothing length must be tuned |
| **FBSI** | Weight is built from the **faces** of the cell | Smooth, follows the real cell shape, nothing to tune, cost similar to an isotropic kernel |

## 2. A cell is an intersection of half-spaces

A convex cell with faces $f = 1 \dots N_f$ is the set of points that lie on the inner side of every
face plane. Each face is described by its **outward unit normal** $\vec n_f$ and a point on the face
$\vec x_f$ (the face centroid). The signed distance from the particle centre $\vec x_p$ to face $f$ is

```math
d_{p,f} = \vec n_f \cdot (\vec x_p - \vec x_f)
```

which is negative when the centre is inside the face. A hard (Boolean) inside test would require
$d_{p,f} \le 0$ for all faces. FBSI replaces each hard test with a **soft** one.

## 3. Soft half-space: one weight per face

For a particle of radius $R_p$, define the transition coordinate and the face weight

```math
\xi_{p,f} = \mathrm{clamp}\left(-\frac{d_{p,f}}{R_p},\,-1,\,1\right), \qquad
t_{p,f} = \frac{1}{2}\,(\xi_{p,f} + 1), \qquad
W_{p,f} = t_{p,f}^2\,(3 - 2\,t_{p,f})
```

* $W_{p,f} = 1$ when the particle is completely inside the face ($d_{p,f} \le -R_p$),
* $W_{p,f} = 0$ when it is completely outside ($d_{p,f} \ge R_p$),
* in between, $W_{p,f}$ changes smoothly (cubic smoothstep, $C^1$-continuous).

**Why this curve?** The smoothstep is not just a convenient smoothing function. It is *exactly* the
volume fraction of a sphere on one side of a plane (a spherical cap). With cap height
$h = R_p - d_{p,f} = 2R_p\,t_{p,f}$:

```math
\frac{V_\text{cap}}{V_p} = \frac{h^2 (3R_p - h)}{4R_p^3} = t_{p,f}^2\,(3 - 2t_{p,f}) = W_{p,f}
```

So when a particle crosses **one** face, FBSI returns the exact answer. The tests in this repository
check this to machine precision.

![Soft half-space and face weight](images/concept.png)

## 4. Shape factor: combine the faces

The weight of the whole cell is the product of its face weights (a soft logical AND):

```math
S_{p,c} = \prod_{f=1}^{N_f} W_{p,f}
```

Because the faces themselves are used, $S_{p,c}$ stretches with elongated cells and tilts with skewed
cells. This is the **anisotropic shape factor**: it adapts to the actual cell geometry, not to an
equivalent sphere.

*Example.* A particle centred on the vertex shared by 8 cubes is at $d = 0$ from three faces of each
cube, so $S = 0.5^3 = 0.125$ for every cube.

## 5. Normalise over the neighbouring cells

The shape factors are normalised over all candidate cells $k$ (the host cell and its neighbours) so
that the particle volume is conserved exactly:

```math
\phi_{p,c} = \frac{S_{p,c}}{\sum_k S_{p,k}}
```

For the 8-cube example this gives $\phi = 1/8$ for each cube, which is exact.

When a particle crosses several faces at once (near edges and vertices), the product is an
approximation. It is exact for a single face and most accurate for orthogonal faces (on the
hexahedral benchmark here, $R^2 \approx 0.999$ against the direct-numerical reference); the largest
deviations appear at vertices with strongly acute or obtuse face angles (highly skewed cells). The
normalisation step removes most of this error and always conserves the particle volume.

## 6. Particles at domain boundaries

A particle can overlap a wall, inlet or outlet. For every boundary face the particle crosses, FBSI
adds a **virtual exterior cell**: a semi-infinite prism outside the domain, bounded by

* the boundary face itself (normal pointing back into the domain), and
* one side face per edge $\vec e_{b,m}$ of the boundary face, with normal
  $\vec n_{\text{side},m} \propto \vec n_b \times \vec e_{b,m}$, pointing away from the face.

Exterior cells are normalised together with the interior cells. The fraction they receive is the part
of the particle outside the domain, so no wall correction is needed. See
[`python/examples/04_wall_boundary.py`](../python/examples/04_wall_boundary.py).

## 7. Properties

| Property | Why |
|---|---|
| **No tuning** | Uses only face normals, face centroids and $R_p$ |
| **Conservative** | $\sum_c \phi_{p,c} = 1$ by construction |
| **Smooth** | $W_{p,f}$ is $C^1$, so $\phi_{p,c}$ changes smoothly as particles move (no jumps, no "ghost forces") |
| **Local** | A face only matters inside a band of width $2R_p$; cells the particle does not reach get zero |
| **Anisotropic** | Follows elongated, skewed, tetrahedral and polyhedral cells |
| **Cheap** | $O(N_\text{candidates} \times N_f)$ multiply–adds per particle with early exits; about 4× PCM, similar to an isotropic kernel |

## 8. Limitations

* **Convex cells.** The half-space description assumes convex cells, which covers standard
  hexahedral, tetrahedral, prismatic and most polyhedral finite-volume cells. Non-convex cells can be
  split into convex pieces (for example tetrahedra).
* **Multi-face interactions.** Accuracy decreases for highly skewed cells, where particles often
  interact with several non-orthogonal faces at once (see the skewness study in the paper). FBSI still
  had the lowest error of the methods tested.
* **Spherical particles.** The method is formulated for spheres of radius $R_p$.

## Sign convention note

The paper and this repository use **outward** normals with $\xi = \mathrm{clamp}(-d/R_p, -1, 1)$.
The original research scripts used **inward** normals with $\xi = \mathrm{clamp}(+d/R_p, -1, 1)$.
The two are identical.
