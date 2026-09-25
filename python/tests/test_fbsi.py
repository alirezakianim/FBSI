"""Tests for the FBSI reference implementation.  Run with:  pytest python/tests"""

import numpy as np
import pytest

import fbsi
from fbsi import Cell, exterior_cell, face_weights, fbsi_fractions


def cap_fraction(d, radius):
    """Exact volume fraction of a sphere on the inner side (d < 0) of a plane."""
    h = np.clip(radius - np.asarray(d, dtype=float), 0.0, 2.0 * radius)
    return h * h * (3.0 * radius - h) / (4.0 * radius**3)


# ----------------------------------------------------------------- geometry
def test_box_and_tetrahedron_geometry():
    box = Cell.box([0, 0, 0], [2, 1, 1])
    assert box.volume == pytest.approx(2.0)
    assert np.allclose(box.centroid, [1.0, 0.5, 0.5])
    tet = Cell.tetrahedron([0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1])
    assert tet.volume == pytest.approx(1.0 / 6.0)
    assert np.allclose(tet.centroid, [0.25, 0.25, 0.25])


def test_normals_point_outward_regardless_of_vertex_order():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    faces = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
    a = Cell.from_polyhedron(vertices, faces)
    b = Cell.from_polyhedron(vertices, [face[::-1] for face in faces])
    assert np.allclose(a.normals, b.normals)
    assert np.all(a.signed_distance(a.centroid) < 0.0)


def test_meshes_fill_the_box():
    for mesh, n_boundary in [(fbsi.hex_mesh(3), 6 * 9), (fbsi.tet_mesh(3), 6 * 9 * 2)]:
        assert mesh.volumes.sum() == pytest.approx(1.0)
        assert len(mesh.boundary_faces) == n_boundary


# ------------------------------------------------------------- face weight
def test_face_weight_limits_and_midpoint():
    w = face_weights(np.array([-2.0, -1.0, 0.0, 1.0, 2.0]), 1.0)
    assert np.allclose(w, [1.0, 1.0, 0.5, 0.0, 0.0])


def test_smoothstep_equals_spherical_cap_fraction():
    d = np.linspace(-1.2, 1.2, 101)
    assert np.allclose(face_weights(d, 1.0), cap_fraction(d, 1.0))


def test_face_weight_is_monotone_and_c1():
    d = np.linspace(-1.0, 1.0, 2001)
    w = face_weights(d, 1.0)
    assert np.all(np.diff(w) <= 0.0)
    slope = np.gradient(w, d)
    assert abs(slope[0]) < 1e-2 and abs(slope[-1]) < 1e-2  # zero slope at +-R


# --------------------------------------------------------------- fractions
def test_particle_inside_one_cell_goes_to_that_cell():
    cells = [Cell.box([0, 0, 0], [1, 1, 1]), Cell.box([1, 0, 0], [2, 1, 1])]
    assert np.allclose(fbsi_fractions([0.5, 0.5, 0.5], 0.3, cells), [1.0, 0.0])


@pytest.mark.parametrize("x", [0.75, 0.9, 1.0, 1.05, 1.2])
def test_single_face_crossing_is_exact(x):
    cells = [Cell.box([0, 0, 0], [1, 1, 1]), Cell.box([1, 0, 0], [2, 1, 1])]
    phi = fbsi_fractions([x, 0.5, 0.5], 0.3, cells)
    assert phi[0] == pytest.approx(cap_fraction(x - 1.0, 0.3), abs=1e-12)


def test_vertex_shared_by_eight_cells():
    cells = [Cell.box([i, j, k], [i + 1, j + 1, k + 1]) for i in (-1, 0) for j in (-1, 0) for k in (-1, 0)]
    assert np.allclose(fbsi_fractions([0, 0, 0], 0.3, cells), 1.0 / 8.0)


def test_partition_of_unity_on_tet_mesh():
    mesh = fbsi.tet_mesh(4)
    radius = 0.8 * mesh.mean_equivalent_radius()
    rng = np.random.default_rng(0)
    for center in rng.uniform(radius, 1 - radius, size=(50, 3)):
        phi = fbsi_fractions(center, radius, [mesh.cells[i] for i in mesh.candidates(center, radius)])
        assert np.all(phi >= 0.0)
        assert phi.sum() == pytest.approx(1.0)


def test_fractions_change_continuously():
    mesh = fbsi.tet_mesh(3)
    radius = 0.8 * mesh.mean_equivalent_radius()
    index = np.arange(len(mesh))
    previous = None
    for x in np.linspace(0.3, 0.7, 400):
        center = np.array([x, 0.41, 0.52])
        phi = np.zeros(len(mesh))
        candidates = mesh.candidates(center, radius)
        phi[index[candidates]] = fbsi_fractions(center, radius, [mesh.cells[i] for i in candidates])
        if previous is not None:
            assert np.max(np.abs(phi - previous)) < 0.05  # step is 1e-3, no jumps
        previous = phi


def test_error_when_no_candidate_overlaps():
    with pytest.raises(ValueError):
        fbsi_fractions([5.0, 5.0, 5.0], 0.1, [Cell.box([0, 0, 0], [1, 1, 1])])


# -------------------------------------------------------------- boundaries
@pytest.mark.parametrize("x", [0.06, 0.03, 0.0, -0.03])
def test_exterior_cell_captures_volume_outside_a_wall(x):
    mesh = fbsi.hex_mesh(5)
    radius, center = 0.08, np.array([x, 0.31, 0.49])
    interior = [mesh.cells[i] for i in mesh.candidates(center, radius)]
    phi = fbsi_fractions(center, radius, interior + mesh.exterior_cells(center, radius))
    assert phi[: len(interior)].sum() == pytest.approx(cap_fraction(-x, radius), abs=1e-9)


def test_exterior_cell_faces():
    ghost = exterior_cell([[0, 0, 0], [0, 1, 0], [0, 1, 1], [0, 0, 1]], interior_point=[0.5, 0.5, 0.5])
    assert ghost.exterior and ghost.n_faces == 5
    assert ghost.contains([-3.0, 0.5, 0.5])      # behind the wall
    assert not ghost.contains([0.5, 0.5, 0.5])   # inside the domain


# -------------------------------------------------------------- accuracy
def test_accuracy_against_direct_numerical_reference():
    mesh = fbsi.tet_mesh(5)
    radius = 0.8 * mesh.mean_equivalent_radius()
    rng = np.random.default_rng(3)
    err_fbsi, err_iso = [], []
    for center in rng.uniform(radius, 1 - radius, size=(40, 3)):
        cells = [mesh.cells[i] for i in mesh.candidates(center, radius)]
        reference = fbsi.direct_numerical_fractions(center, radius, cells, n_per_dim=48)
        err_fbsi.append(np.abs(fbsi_fractions(center, radius, cells) - reference).sum())
        err_iso.append(np.abs(fbsi.isotropic_fractions(center, radius, cells) - reference).sum())
    assert np.mean(err_fbsi) < 0.5 * np.mean(err_iso)
    assert np.mean(err_fbsi) < 0.15


def test_solid_volume_fraction_conserves_particle_volume():
    mesh = fbsi.hex_mesh(4)
    radius = 0.05
    rng = np.random.default_rng(1)
    centers = rng.uniform(radius, 1 - radius, size=(30, 3))
    v_p = 4.0 / 3.0 * np.pi * radius**3
    per_particle = []
    for center in centers:
        index = mesh.candidates(center, radius)
        per_particle.append((index, fbsi_fractions(center, radius, [mesh.cells[i] for i in index])))
    phi_cells = fbsi.solid_volume_fraction(per_particle, [v_p] * len(centers), mesh.volumes)
    assert np.sum(phi_cells * mesh.volumes) == pytest.approx(len(centers) * v_p)
