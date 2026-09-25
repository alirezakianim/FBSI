// simple_mesh.hpp - small structured test meshes used by the examples and tests.
//
// In your own solver you would use your existing mesh and neighbour lists; this
// file only exists so the examples run without external mesh files.

#ifndef FBSI_EXAMPLES_SIMPLE_MESH_HPP
#define FBSI_EXAMPLES_SIMPLE_MESH_HPP

#include <fbsi/fbsi.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <stdexcept>
#include <vector>

namespace simple_mesh {

using fbsi::Cell;
using fbsi::Vec3;

struct Mesh {
    std::vector<Vec3> vertices;
    std::vector<Cell> cells;
    std::vector<std::vector<int>> boundary_faces;  // vertex ids of faces on the domain boundary
    std::vector<int> boundary_owner;               // cell that owns each boundary face

    // Cells whose bounding box overlaps the particle's (brute force; fine for small meshes).
    std::vector<const Cell*> candidates(const Vec3& center, double radius) const {
        std::vector<const Cell*> result;
        for (const Cell& cell : cells) {
            if (fbsi::bbox_overlaps(cell, center, radius)) result.push_back(&cell);
        }
        return result;
    }

    // Index of the cell containing x.
    int locate(const Vec3& x) const {
        for (std::size_t i = 0; i < cells.size(); ++i) {
            if (fbsi::bbox_overlaps(cells[i], x, 0.0) && fbsi::contains(cells[i], x)) return static_cast<int>(i);
        }
        throw std::runtime_error("point outside the mesh");
    }

    // Virtual exterior cells for the boundary faces the particle crosses.
    std::vector<Cell> exterior_cells(const Vec3& center, double radius) const {
        std::vector<Cell> result;
        for (std::size_t b = 0; b < boundary_faces.size(); ++b) {
            std::vector<Vec3> polygon;
            Cell box;  // bounding box of the face
            box.bbox_min = {Cell::inf, Cell::inf, Cell::inf};
            box.bbox_max = {-Cell::inf, -Cell::inf, -Cell::inf};
            for (int v : boundary_faces[b]) {
                const Vec3& p = vertices[v];
                polygon.push_back(p);
                box.bbox_min = {std::min(box.bbox_min.x, p.x), std::min(box.bbox_min.y, p.y), std::min(box.bbox_min.z, p.z)};
                box.bbox_max = {std::max(box.bbox_max.x, p.x), std::max(box.bbox_max.y, p.y), std::max(box.bbox_max.z, p.z)};
            }
            if (!fbsi::bbox_overlaps(box, center, radius)) continue;
            Cell ghost = fbsi::make_exterior_cell(polygon, cells[boundary_owner[b]].centroid);
            if (std::abs(fbsi::signed_distance(ghost.faces[0], center)) < radius) result.push_back(ghost);
        }
        return result;
    }

    double mean_equivalent_radius() const {
        double sum = 0.0;
        const double pi = std::acos(-1.0);
        for (const Cell& c : cells) sum += std::cbrt(3.0 * c.volume / (4.0 * pi));
        return sum / cells.size();
    }
};

inline Mesh build(std::vector<Vec3> vertices, const std::vector<std::vector<std::vector<int>>>& cell_faces) {
    Mesh mesh;
    mesh.vertices = std::move(vertices);
    std::map<std::vector<int>, int> count;
    for (const auto& faces : cell_faces) {
        mesh.cells.push_back(fbsi::make_polyhedron(mesh.vertices, faces));
        for (auto face : faces) {
            std::sort(face.begin(), face.end());
            ++count[face];
        }
    }
    for (std::size_t c = 0; c < cell_faces.size(); ++c) {
        for (const auto& face : cell_faces[c]) {
            auto key = face;
            std::sort(key.begin(), key.end());
            if (count[key] == 1) {  // a face used by one cell lies on the boundary
                mesh.boundary_faces.push_back(face);
                mesh.boundary_owner.push_back(static_cast<int>(c));
            }
        }
    }
    return mesh;
}

inline std::vector<Vec3> grid_vertices(int n, const Vec3& lengths) {
    std::vector<Vec3> v;
    for (int i = 0; i <= n; ++i)
        for (int j = 0; j <= n; ++j)
            for (int k = 0; k <= n; ++k)
                v.push_back({lengths.x * i / n, lengths.y * j / n, lengths.z * k / n});
    return v;
}

// Structured hexahedral mesh of a box with n cells per direction.
inline Mesh hex_mesh(int n, const Vec3& lengths = {1.0, 1.0, 1.0}) {
    auto id = [n](int i, int j, int k) { return (i * (n + 1) + j) * (n + 1) + k; };
    std::vector<std::vector<std::vector<int>>> cells;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            for (int k = 0; k < n; ++k) {
                auto v = [&](int a, int b, int c) { return id(i + a, j + b, k + c); };
                cells.push_back({{v(0, 0, 0), v(0, 1, 0), v(0, 1, 1), v(0, 0, 1)},
                                 {v(1, 0, 0), v(1, 1, 0), v(1, 1, 1), v(1, 0, 1)},
                                 {v(0, 0, 0), v(1, 0, 0), v(1, 0, 1), v(0, 0, 1)},
                                 {v(0, 1, 0), v(1, 1, 0), v(1, 1, 1), v(0, 1, 1)},
                                 {v(0, 0, 0), v(1, 0, 0), v(1, 1, 0), v(0, 1, 0)},
                                 {v(0, 0, 1), v(1, 0, 1), v(1, 1, 1), v(0, 1, 1)}});
            }
    return build(grid_vertices(n, lengths), cells);
}

// Structured tetrahedral mesh: each hexahedron split into 6 congruent tetrahedra
// (Kuhn split along the main diagonal).
inline Mesh tet_mesh(int n, const Vec3& lengths = {1.0, 1.0, 1.0}) {
    auto id = [n](const std::array<int, 3>& p) { return (p[0] * (n + 1) + p[1]) * (n + 1) + p[2]; };
    std::vector<std::vector<std::vector<int>>> cells;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            for (int k = 0; k < n; ++k) {
                std::array<int, 3> order = {0, 1, 2};
                do {
                    std::array<int, 3> p = {i, j, k};
                    std::array<int, 4> ids{};
                    ids[0] = id(p);
                    for (int s = 0; s < 3; ++s) {
                        ++p[order[s]];
                        ids[s + 1] = id(p);
                    }
                    cells.push_back({{ids[0], ids[1], ids[2]}, {ids[0], ids[1], ids[3]},
                                     {ids[0], ids[2], ids[3]}, {ids[1], ids[2], ids[3]}});
                } while (std::next_permutation(order.begin(), order.end()));
            }
    return build(grid_vertices(n, lengths), cells);
}

}  // namespace simple_mesh

#endif  // FBSI_EXAMPLES_SIMPLE_MESH_HPP
