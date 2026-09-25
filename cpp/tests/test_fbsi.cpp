// Tests for fbsi.hpp (no test framework needed). Run with: ctest --test-dir build

#include <fbsi/fbsi.hpp>

#include "simple_mesh.hpp"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using fbsi::Cell;
using fbsi::Vec3;

static int failures = 0;

#define CHECK(condition)                                                        \
    do {                                                                        \
        if (!(condition)) {                                                     \
            std::printf("FAILED: %s  (%s:%d)\n", #condition, __FILE__, __LINE__); \
            ++failures;                                                         \
        }                                                                       \
    } while (0)

static bool near(double a, double b, double tol = 1e-12) { return std::abs(a - b) <= tol; }

// Exact volume fraction of a sphere on the inner side (d < 0) of a plane.
static double cap_fraction(double d, double radius) {
    const double h = std::fmin(std::fmax(radius - d, 0.0), 2.0 * radius);
    return h * h * (3.0 * radius - h) / (4.0 * radius * radius * radius);
}

static double sum(const std::vector<double>& v) {
    double s = 0.0;
    for (double x : v) s += x;
    return s;
}

static void test_geometry() {
    const Cell box = fbsi::make_box({0, 0, 0}, {2, 1, 1});
    CHECK(near(box.volume, 2.0));
    CHECK(near(box.centroid.x, 1.0) && near(box.centroid.y, 0.5) && near(box.centroid.z, 0.5));
    const Cell tet = fbsi::make_tetrahedron({0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1});
    CHECK(near(tet.volume, 1.0 / 6.0));
    CHECK(fbsi::contains(tet, tet.centroid));
    CHECK(!fbsi::contains(tet, {1, 1, 1}));
}

static void test_face_weight() {
    CHECK(near(fbsi::face_weight(-2.0, 1.0), 1.0));
    CHECK(near(fbsi::face_weight(0.0, 1.0), 0.5));
    CHECK(near(fbsi::face_weight(2.0, 1.0), 0.0));
    for (double d = -1.2; d <= 1.2; d += 0.01) CHECK(near(fbsi::face_weight(d, 1.0), cap_fraction(d, 1.0)));
}

static void test_two_cells() {
    const Cell a = fbsi::make_box({0, 0, 0}, {1, 1, 1});
    const Cell b = fbsi::make_box({1, 0, 0}, {2, 1, 1});
    std::vector<double> phi;
    fbsi::fbsi_fractions({0.5, 0.5, 0.5}, 0.3, {&a, &b}, phi);
    CHECK(near(phi[0], 1.0) && near(phi[1], 0.0));
    for (double x : {0.75, 0.9, 1.0, 1.05, 1.2}) {
        fbsi::fbsi_fractions({x, 0.5, 0.5}, 0.3, {&a, &b}, phi);
        CHECK(near(phi[0], cap_fraction(x - 1.0, 0.3)));  // single face: exact
    }
    const double total = fbsi::fbsi_fractions({5, 5, 5}, 0.1, {&a, &b}, phi);
    CHECK(total == 0.0);  // no overlap is reported, not hidden
}

static void test_eight_cells_at_vertex() {
    std::vector<Cell> cubes;
    for (int i : {-1, 0})
        for (int j : {-1, 0})
            for (int k : {-1, 0}) cubes.push_back(fbsi::make_box({1.0 * i, 1.0 * j, 1.0 * k}, {i + 1.0, j + 1.0, k + 1.0}));
    std::vector<const Cell*> cells;
    for (const Cell& c : cubes) cells.push_back(&c);
    std::vector<double> phi;
    fbsi::fbsi_fractions({0, 0, 0}, 0.3, cells, phi);
    for (double v : phi) CHECK(near(v, 0.125));
}

static void test_partition_of_unity_and_accuracy() {
    const simple_mesh::Mesh mesh = simple_mesh::tet_mesh(5);
    const double radius = 0.8 * mesh.mean_equivalent_radius();
    std::mt19937_64 rng(3);
    std::uniform_real_distribution<double> uniform(radius, 1.0 - radius);
    double err = 0.0;
    const int n = 50;
    std::vector<double> phi;
    for (int p = 0; p < n; ++p) {
        const Vec3 c{uniform(rng), uniform(rng), uniform(rng)};
        const auto cells = mesh.candidates(c, radius);
        fbsi::fbsi_fractions(c, radius, cells, phi);
        CHECK(near(sum(phi), 1.0, 1e-12));
        const auto ref = fbsi::direct_numerical_fractions(c, radius, cells, 64);
        for (std::size_t i = 0; i < phi.size(); ++i) {
            CHECK(phi[i] >= 0.0);
            err += std::abs(phi[i] - ref[i]);
        }
    }
    CHECK(err / n < 0.15);  // mean L1 error per particle
}

static void test_wall() {
    const simple_mesh::Mesh mesh = simple_mesh::hex_mesh(5);
    const double radius = 0.08;
    std::vector<double> phi;
    for (double x : {0.06, 0.03, 0.0, -0.03}) {
        const Vec3 c{x, 0.31, 0.49};
        auto cells = mesh.candidates(c, radius);
        const std::size_t n_interior = cells.size();
        const std::vector<Cell> ghosts = mesh.exterior_cells(c, radius);
        for (const Cell& g : ghosts) cells.push_back(&g);
        fbsi::fbsi_fractions(c, radius, cells, phi);
        double inside = 0.0;
        for (std::size_t i = 0; i < n_interior; ++i) inside += phi[i];
        CHECK(near(inside, cap_fraction(-x, radius), 1e-9));
    }
}

int main() {
    test_geometry();
    test_face_weight();
    test_two_cells();
    test_eight_cells_at_vertex();
    test_partition_of_unity_and_accuracy();
    test_wall();
    if (failures == 0) std::printf("All FBSI tests passed.\n");
    return failures == 0 ? 0 : 1;
}
