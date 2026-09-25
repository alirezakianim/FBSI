// FBSI quick start: the method on the smallest possible examples.
//
//   ./build/quickstart

#include <fbsi/fbsi.hpp>

#include <cmath>
#include <cstdio>
#include <vector>

using fbsi::Cell;
using fbsi::Vec3;

int main() {
    // ------------------------------------------------------------------
    // 1. A particle moving across the face shared by two cubic cells
    // ------------------------------------------------------------------
    const Cell left = fbsi::make_box({0, 0, 0}, {1, 1, 1});
    const Cell right = fbsi::make_box({1, 0, 0}, {2, 1, 1});
    const std::vector<const Cell*> pair = {&left, &right};
    const double radius = 0.3;

    std::printf("1) Particle (R = 0.3) crossing the face x = 1 between two cells\n");
    std::printf("%6s | %9s %10s | %10s\n", "x_p", "FBSI left", "FBSI right", "exact left");
    std::vector<double> phi;
    for (double x : {0.6, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4}) {
        fbsi::fbsi_fractions({x, 0.5, 0.5}, radius, pair, phi);
        // Exact: volume fraction of a sphere cap of height h on the left of x = 1.
        const double h = std::fmin(std::fmax(radius + (1.0 - x), 0.0), 2.0 * radius);
        const double exact = h * h * (3.0 * radius - h) / (4.0 * radius * radius * radius);
        std::printf("%6.2f | %9.4f %10.4f | %10.4f\n", x, phi[0], phi[1], exact);
    }

    // ------------------------------------------------------------------
    // 2. A particle centred on a vertex shared by eight cells
    // ------------------------------------------------------------------
    std::vector<Cell> cubes;
    for (int i : {-1, 0})
        for (int j : {-1, 0})
            for (int k : {-1, 0}) cubes.push_back(fbsi::make_box({1.0 * i, 1.0 * j, 1.0 * k}, {i + 1.0, j + 1.0, k + 1.0}));
    std::vector<const Cell*> corner;
    for (const Cell& c : cubes) corner.push_back(&c);
    fbsi::fbsi_fractions({0, 0, 0}, 0.3, corner, phi);
    std::printf("\n2) Particle centred on a vertex shared by 8 cubes -> 1/8 each:\n  ");
    for (double v : phi) std::printf(" %.4f", v);
    std::printf("\n");

    // ------------------------------------------------------------------
    // 3. A wall: add a virtual exterior cell behind the boundary face
    // ------------------------------------------------------------------
    // The face x = 0 of `left` is a wall. The exterior cell receives the part of
    // the particle that lies outside the domain.
    const Cell wall = fbsi::make_exterior_cell({{0, 0, 0}, {0, 1, 0}, {0, 1, 1}, {0, 0, 1}}, left.centroid);
    const std::vector<const Cell*> with_wall = {&left, &right, &wall};
    std::printf("\n3) Particle (R = 0.3) approaching the wall x = 0\n");
    std::printf("%6s | %14s | %6s\n", "x_p", "inside domain", "exact");
    for (double x : {0.4, 0.2, 0.1, 0.0}) {
        fbsi::fbsi_fractions({x, 0.5, 0.5}, radius, with_wall, phi);
        const double h = std::fmin(radius + x, 2.0 * radius);
        const double exact = h * h * (3.0 * radius - h) / (4.0 * radius * radius * radius);
        std::printf("%6.2f | %14.4f | %6.4f\n", x, phi[0] + phi[1], exact);
    }
    return 0;
}
