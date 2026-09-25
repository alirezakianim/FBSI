# Implementing FBSI in your own solver

FBSI is small: the core fits in about 20 lines. This guide lists what your solver has to provide,
where the method goes in the coupling loop, and how to check your implementation.

The fastest way to start in C++ is to copy [`cpp/include/fbsi/fbsi.hpp`](../cpp/include/fbsi/fbsi.hpp)
(header-only, C++17, no dependencies) into your code. In other languages, port the functions
`face_weight`, `shape_factor` and `fbsi_fractions`.

## 1. Data you need

| Per | Quantity | Notes |
|---|---|---|
| cell face | outward unit normal $\vec n_f$ | Every finite-volume solver stores face area vectors; normalise them and orient them outward (see §4) |
| cell face | a point on the face $\vec x_f$ | The face centroid |
| cell | bounding box (optional) | For a cheap rejection test |
| cell | list of neighbour cells | Used to build the candidate list (see §3) |
| boundary face | its vertices and owner cell | Only if particles can touch walls/inlets/outlets (see §5) |
| particle | centre $\vec x_p$, radius $R_p$, host cell | The host cell is known from particle tracking |

Face data do not change for a static mesh: compute them once at start-up (or after each mesh change).

## 2. Where FBSI goes in the coupling loop

```text
every coupling step:
    solid_volume[c] = 0 for all cells
    for each particle p:                                   # independent: parallelise over particles
        candidates = host(p) + neighbours(host(p))         # §3
        (+ exterior cells for boundary faces within R_p)   # §5
        for each candidate k:
            S[k] = product over faces f of k of W(d_pf)    # with early exit
        phi[k] = S[k] / sum(S)
        for each interior candidate k:
            solid_volume[k] += phi[k] * V_p
    Phi[c] = solid_volume[c] / V_c                         # solid volume fraction
    epsilon[c] = 1 - Phi[c]                                # fluid volume fraction (void fraction)
```

The same weights $\phi_{p,c}$ can also be used to spread each particle's fluid–particle force back onto
the cells, so the momentum exchange is distributed the same way as the volume. (The paper validates
the volume mapping; full CFD–DEM validation is future work.)

### Core in C++

```cpp
#include <fbsi/fbsi.hpp>

// For each particle: gather candidate cells, compute fractions, accumulate.
std::vector<const fbsi::Cell*> candidates = /* host cell + neighbours */;
std::vector<double> phi;
if (fbsi::fbsi_fractions(x_p, R_p, candidates, phi) > 0.0) {
    for (std::size_t k = 0; k < candidates.size(); ++k)
        solid_volume[cell_index_of(candidates[k])] += phi[k] * V_p;
} else {
    solid_volume[host] += V_p;   // candidate list did not cover the particle: PCM fallback
}
```

If your cells are not stored as `fbsi::Cell`, call the array overload directly:

```cpp
double S = fbsi::shape_factor(face_ptr, n_faces, x_p, R_p);   // faces: {normal, point}
```

## 3. Choosing candidate cells

The candidate list must contain **every cell the particle can overlap**.

* The host cell plus the cells sharing a vertex with it (point neighbours) are enough when $R_p$ is
  smaller than the neighbouring cells.
* For particles comparable to or larger than the cells, or strongly graded meshes, add more
  neighbour layers or use a bounding-box search.
* Extra candidates are harmless: FBSI returns $S = 0$ for cells the particle cannot reach (bounding-box
  test and early exit when $d_{p,f} \ge R_p$). A **missing** candidate is the error to avoid: its
  share would be redistributed to the other cells.

## 4. Face orientation

FBSI needs normals pointing **out of** the cell being evaluated. An interior face is shared by two cells
and has one stored area vector, so it is outward for one cell and must be flipped for the other.

| Solver | Face data | Orientation |
|---|---|---|
| OpenFOAM | `mesh.Sf()`, `mesh.Cf()`, `mesh.cells()[c]` | `Sf` points from owner to neighbour: outward for the owner, flip for the neighbour |
| ANSYS Fluent (UDF) | `F_AREA`, `F_CENTROID`, `c_face_loop` | The area vector points from `c0` to `c1`: outward for `c0`, flip for `c1` |
| Any solver | face area vector $\vec A_f$, face centroid $\vec x_f$, cell centroid $\vec x_c$ | Flip $\vec A_f$ if $\vec A_f \cdot (\vec x_f - \vec x_c) < 0$ (valid for convex cells) |

The last rule is the safest because it does not depend on solver conventions. `make_polyhedron` in
C++ and `Cell.from_polyhedron` in Python apply it automatically.

## 5. Boundaries (walls, inlets, outlets)

If a particle overlaps a boundary face, part of its volume lies outside the domain. Handle it with
virtual exterior cells:

1. At start-up, build one exterior cell per boundary face with `make_exterior_cell(face_vertices,
   owner_centroid)` (C++) or `exterior_cell(...)` (Python). They are static, like the interior face data.
2. For each particle, add the exterior cells of the boundary faces within $R_p$ of the particle to
   its candidate list.
3. Normalise as usual, then accumulate only the **interior** fractions. The exterior fraction is the
   volume outside the domain and is discarded.

Without this step the normalisation pushes the whole particle into the domain, which overestimates
$\Phi_c$ next to walls.

## 6. Performance notes

* **Early exits.** Stop evaluating a cell as soon as one face gives $d_{p,f} \ge R_p$. Faces with
  $d_{p,f} \le -R_p$ contribute a factor of 1 and can be skipped.
* **Bounding boxes.** Rejecting cells with a box-overlap test first removes most candidates cheaply.
* **Particle fully inside its host.** If $d_{p,f} \le -R_p$ for every face of the host cell, then
  $\phi_{p,\text{host}} = 1$ and no other cell needs to be evaluated.
* **Memory layout.** Store face normals and points contiguously per cell (or as structure-of-arrays).
* **Parallelism.** Particles are independent. With OpenMP use atomic adds or thread-local
  accumulation buffers for `solid_volume`. With MPI, include halo (ghost) cells in the candidate list
  of particles near partition boundaries, as for any kernel method.

In the paper the average cost was 3.7–4.3× that of PCM, similar to an isotropic polynomial kernel and
below Gaussian-type kernels. `cpp/examples/mesh_benchmark.cpp` measures the ratio on your machine.

## 7. Checking your implementation

Before running a full simulation, check these cases. They are all in the test suites
(`python/tests/test_fbsi.py`, `cpp/tests/test_fbsi.cpp`):

| Test | Expected result |
|---|---|
| Particle well inside one cell | $\phi = 1$ for that cell, 0 elsewhere |
| Particle centred on a face between two cells | $\phi = 0.5 / 0.5$ |
| Particle crossing one face at distance $d$ | exact spherical-cap fraction $h^2(3R_p-h)/(4R_p^3)$ with $h = R_p - d$ |
| Particle centred on a vertex shared by 8 hexahedra | $\phi = 1/8$ each |
| Any particle, no boundary | $\sum_c \phi_{p,c} = 1$ |
| Particle crossing a flat wall at distance $d$ (with exterior cells) | the interior fraction equals the exact cap fraction |
| Random particles on a small mesh | close agreement with a voxel (direct-numerical) reference |

A frequent bug is an inward-pointing normal. It shows up immediately as $\phi \approx 0$ for a
particle sitting in the middle of its host cell.
