// fbsi.hpp - Face-Based Soft-Intersection (FBSI) particle-to-cell volume mapping.
//
// Header-only, C++17, no dependencies. Copy this file into your solver.
//
// Reference:
//   A. Kianimoqadam, A. J. Schrader, "An anisotropic, Face-Based Soft-Intersection
//   method for robust Lagrangian-Eulerian coupling", Computer Methods in Applied
//   Mechanics and Engineering 461 (2026) 119161.
//   https://doi.org/10.1016/j.cma.2026.119161
//
// The method in five lines (x_p: particle centre, R_p: particle radius):
//   d_pf   = n_f . (x_p - x_f)            signed distance to face f (< 0 inside)
//   t_pf   = (clamp(-d_pf / R_p, -1, 1) + 1) / 2
//   W_pf   = t_pf^2 (3 - 2 t_pf)          soft half-space (cubic smoothstep)
//   S_pc   = prod_f W_pf                  anisotropic shape factor of cell c
//   phi_pc = S_pc / sum_k S_pk            fraction of the particle in cell c
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Alireza Kianimoqadam

#ifndef FBSI_FBSI_HPP
#define FBSI_FBSI_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace fbsi {

// ============================================================================
// Minimal 3-vector
// ============================================================================
struct Vec3 {
    double x = 0.0, y = 0.0, z = 0.0;
};

inline Vec3 operator+(const Vec3& a, const Vec3& b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline Vec3 operator-(const Vec3& a, const Vec3& b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline Vec3 operator-(const Vec3& a) { return {-a.x, -a.y, -a.z}; }
inline Vec3 operator*(double s, const Vec3& a) { return {s * a.x, s * a.y, s * a.z}; }
inline double dot(const Vec3& a, const Vec3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline Vec3 cross(const Vec3& a, const Vec3& b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
inline double norm(const Vec3& a) { return std::sqrt(dot(a, a)); }

// ============================================================================
// Cell description
// ============================================================================

// One face of a convex cell: the OUTWARD unit normal and any point on the face
// plane (the face centroid is the natural choice).
struct Face {
    Vec3 normal;
    Vec3 point;
};

// A convex cell as the intersection of its face half-spaces. Only `faces` is
// needed by FBSI; the bounding box enables a cheap rejection test (leave it
// infinite to disable), and volume/centroid are informational.
struct Cell {
    static constexpr double inf = std::numeric_limits<double>::infinity();

    std::vector<Face> faces;
    Vec3 bbox_min{-inf, -inf, -inf};
    Vec3 bbox_max{inf, inf, inf};
    Vec3 centroid{};
    double volume = 0.0;
    bool exterior = false;  // virtual cell outside the domain (boundary treatment)
};

// ============================================================================
// The method
// ============================================================================

// Signed distance from x to the plane of face f; negative on the inside.
inline double signed_distance(const Face& f, const Vec3& x) { return dot(f.normal, x - f.point); }

// Soft half-space weight W_pf. Equals 1 for d <= -R (particle fully inside the
// face), 0 for d >= R (fully outside) and the cubic smoothstep in between,
// which is exactly the volume fraction of the sphere on the inner side of the
// plane. smooth = false gives the linear ramp W = t (for comparison only).
inline double face_weight(double d, double radius, bool smooth = true) {
    const double xi = std::clamp(-d / radius, -1.0, 1.0);
    const double t = 0.5 * (xi + 1.0);
    return smooth ? t * t * (3.0 - 2.0 * t) : t;
}

// Shape factor S_pc = prod_f W_pf for a cell given as a plain array of faces.
// This overload is the one to call from solver data structures.
inline double shape_factor(const Face* faces, std::size_t n_faces, const Vec3& center, double radius,
                           bool smooth = true) {
    double s = 1.0;
    for (std::size_t f = 0; f < n_faces; ++f) {
        const double d = signed_distance(faces[f], center);
        if (d >= radius) return 0.0;                          // completely outside this face
        if (d > -radius) s *= face_weight(d, radius, smooth);  // faces with d <= -R have W = 1
    }
    return s;
}

// Cheap rejection: can the particle's bounding box touch the cell's?
inline bool bbox_overlaps(const Cell& cell, const Vec3& center, double radius) {
    return center.x + radius >= cell.bbox_min.x && center.x - radius <= cell.bbox_max.x &&
           center.y + radius >= cell.bbox_min.y && center.y - radius <= cell.bbox_max.y &&
           center.z + radius >= cell.bbox_min.z && center.z - radius <= cell.bbox_max.z;
}

inline double shape_factor(const Cell& cell, const Vec3& center, double radius, bool smooth = true) {
    if (!bbox_overlaps(cell, center, radius)) return 0.0;
    return shape_factor(cell.faces.data(), cell.faces.size(), center, radius, smooth);
}

// Fractions phi_pc of the particle volume assigned to each candidate cell.
//
// `cells` must contain the cell holding the particle centre and every cell the
// particle may overlap (typically the host cell and its neighbours). Virtual
// exterior cells (make_exterior_cell) may be added to handle boundaries.
//
// On return phi[i] belongs to cells[i] and the values sum to 1. The function
// returns sum_k S_pk; a return value of 0 means no candidate overlaps the
// particle (phi is then all zero; fall back to the host cell).
inline double fbsi_fractions(const Vec3& center, double radius, const std::vector<const Cell*>& cells,
                             std::vector<double>& phi, bool smooth = true) {
    phi.assign(cells.size(), 0.0);
    double total = 0.0;
    for (std::size_t i = 0; i < cells.size(); ++i) {
        phi[i] = shape_factor(*cells[i], center, radius, smooth);
        total += phi[i];
    }
    if (total > 0.0) {
        for (double& value : phi) value /= total;
    }
    return total;
}

// ============================================================================
// Geometry helpers
// ============================================================================

// Boolean indicator I_c(x): true if x is inside (or on) every face of the cell.
inline bool contains(const Cell& cell, const Vec3& x) {
    for (const Face& f : cell.faces) {
        if (signed_distance(f, x) > 0.0) return false;
    }
    return true;
}

// Area-weighted normal of a planar polygon (Newell's method); direction follows
// the vertex order.
inline Vec3 newell_normal(const std::vector<Vec3>& polygon) {
    Vec3 center{};
    for (const Vec3& v : polygon) center = center + v;
    center = (1.0 / polygon.size()) * center;
    Vec3 n{};
    for (std::size_t i = 0; i < polygon.size(); ++i) {
        n = n + cross(polygon[i] - center, polygon[(i + 1) % polygon.size()] - center);
    }
    return n;
}

// Build a convex cell from vertex coordinates and face -> vertex index lists
// (the connectivity every finite-volume mesh stores). Vertices of a face must be
// ordered around the face, in either direction; normals are oriented outward.
inline Cell make_polyhedron(const std::vector<Vec3>& vertices, const std::vector<std::vector<int>>& faces) {
    Cell cell;
    cell.bbox_min = {Cell::inf, Cell::inf, Cell::inf};
    cell.bbox_max = {-Cell::inf, -Cell::inf, -Cell::inf};

    std::vector<int> used;
    for (const auto& face : faces) used.insert(used.end(), face.begin(), face.end());
    std::sort(used.begin(), used.end());
    used.erase(std::unique(used.begin(), used.end()), used.end());

    Vec3 inside{};  // vertex average: an interior point of a convex cell
    for (int i : used) {
        const Vec3& v = vertices[static_cast<std::size_t>(i)];
        inside = inside + v;
        cell.bbox_min = {std::min(cell.bbox_min.x, v.x), std::min(cell.bbox_min.y, v.y), std::min(cell.bbox_min.z, v.z)};
        cell.bbox_max = {std::max(cell.bbox_max.x, v.x), std::max(cell.bbox_max.y, v.y), std::max(cell.bbox_max.z, v.z)};
    }
    inside = (1.0 / used.size()) * inside;

    Vec3 moment{};
    for (const auto& face : faces) {
        std::vector<Vec3> polygon;
        Vec3 center{};
        for (int i : face) {
            polygon.push_back(vertices[static_cast<std::size_t>(i)]);
            center = center + polygon.back();
        }
        center = (1.0 / polygon.size()) * center;
        Vec3 n = newell_normal(polygon);
        if (dot(n, center - inside) < 0.0) n = -n;
        cell.faces.push_back({(1.0 / norm(n)) * n, center});

        // Volume and centroid from a fan of tetrahedra (inside, a, b, center).
        for (std::size_t i = 0; i < polygon.size(); ++i) {
            const Vec3& a = polygon[i];
            const Vec3& b = polygon[(i + 1) % polygon.size()];
            const double v = std::abs(dot(a - inside, cross(b - inside, center - inside))) / 6.0;
            cell.volume += v;
            moment = moment + (v / 4.0) * (inside + a + b + center);
        }
    }
    cell.centroid = (1.0 / cell.volume) * moment;
    return cell;
}

// Axis-aligned hexahedron with opposite corners lo and hi.
inline Cell make_box(const Vec3& lo, const Vec3& hi) {
    const std::vector<Vec3> v = {{lo.x, lo.y, lo.z}, {hi.x, lo.y, lo.z}, {hi.x, hi.y, lo.z}, {lo.x, hi.y, lo.z},
                                 {lo.x, lo.y, hi.z}, {hi.x, lo.y, hi.z}, {hi.x, hi.y, hi.z}, {lo.x, hi.y, hi.z}};
    return make_polyhedron(v, {{0, 3, 7, 4}, {1, 2, 6, 5}, {0, 1, 5, 4}, {3, 2, 6, 7}, {0, 1, 2, 3}, {4, 5, 6, 7}});
}

// Tetrahedron from its four vertices (any order).
inline Cell make_tetrahedron(const Vec3& a, const Vec3& b, const Vec3& c, const Vec3& d) {
    return make_polyhedron({a, b, c, d}, {{0, 1, 2}, {0, 1, 3}, {0, 2, 3}, {1, 2, 3}});
}

// Virtual, semi-infinite control volume on the outside of a boundary face
// (wall, inlet, outlet). It is bounded by the boundary face plus one side face
// per edge with normal n_side ~ n_b x e_m. Add it to the candidate list of a
// particle that crosses the boundary face: the fraction it receives is the part
// of the particle outside the domain.
//
// face_vertices : boundary-face vertices, ordered around the face
// interior_point: any point inside the domain (e.g. the owner-cell centroid)
inline Cell make_exterior_cell(const std::vector<Vec3>& face_vertices, const Vec3& interior_point) {
    Vec3 center{};
    for (const Vec3& v : face_vertices) center = center + v;
    center = (1.0 / face_vertices.size()) * center;

    Vec3 n_b = newell_normal(face_vertices);
    n_b = (1.0 / norm(n_b)) * n_b;
    if (dot(n_b, center - interior_point) < 0.0) n_b = -n_b;  // n_b points out of the domain

    Cell cell;  // unbounded: bounding box stays infinite
    cell.exterior = true;
    cell.faces.push_back({-n_b, center});  // the boundary face, seen from outside
    for (std::size_t i = 0; i < face_vertices.size(); ++i) {
        const Vec3& a = face_vertices[i];
        const Vec3& b = face_vertices[(i + 1) % face_vertices.size()];
        const Vec3 mid = 0.5 * (a + b);
        Vec3 n_side = cross(n_b, b - a);
        if (dot(n_side, mid - center) < 0.0) n_side = -n_side;
        cell.faces.push_back({(1.0 / norm(n_side)) * n_side, mid});
    }
    return cell;
}

// ============================================================================
// Direct-numerical reference (for testing only: cost ~ n_per_dim^3 per particle)
// ============================================================================

// Voxelise the particle with n_per_dim^3 voxels and assign every voxel centre
// inside the sphere to the first cell that contains it. Returns the fraction of
// the particle in each cell (sums to < 1 if part of the particle is outside all
// cells).
inline std::vector<double> direct_numerical_fractions(const Vec3& center, double radius,
                                                      const std::vector<const Cell*>& cells,
                                                      int n_per_dim = 128) {
    std::vector<double> fractions(cells.size(), 0.0);
    const double step = 2.0 * radius / n_per_dim;
    const double r2 = radius * radius;
    long long n_inside = 0;
    for (int i = 0; i < n_per_dim; ++i) {
        for (int j = 0; j < n_per_dim; ++j) {
            for (int k = 0; k < n_per_dim; ++k) {
                const Vec3 offset{-radius + (i + 0.5) * step, -radius + (j + 0.5) * step, -radius + (k + 0.5) * step};
                if (dot(offset, offset) > r2) continue;
                ++n_inside;
                const Vec3 p = center + offset;
                for (std::size_t c = 0; c < cells.size(); ++c) {
                    if (bbox_overlaps(*cells[c], p, 0.0) && contains(*cells[c], p)) {
                        fractions[c] += 1.0;
                        break;
                    }
                }
            }
        }
    }
    for (double& f : fractions) f /= static_cast<double>(n_inside);
    return fractions;
}

}  // namespace fbsi

#endif  // FBSI_FBSI_HPP
