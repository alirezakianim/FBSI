// Accuracy and cost of FBSI compared with the Point-Centroid Method (PCM),
// both measured against the direct-numerical (voxel) reference.
//
//   ./build/mesh_benchmark [hex|tet] [particles] [beta]
//   ./build/mesh_benchmark tet 2000 0.8

#include <fbsi/fbsi.hpp>

#include "simple_mesh.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

using fbsi::Cell;
using fbsi::Vec3;

namespace {

struct Stats {
    double r2, mae;
};

Stats compare(const std::vector<double>& truth, const std::vector<double>& pred) {
    double mean = 0.0;
    for (double t : truth) mean += t;
    mean /= truth.size();
    double ss_res = 0.0, ss_tot = 0.0, abs_err = 0.0;
    for (std::size_t i = 0; i < truth.size(); ++i) {
        ss_res += (truth[i] - pred[i]) * (truth[i] - pred[i]);
        ss_tot += (truth[i] - mean) * (truth[i] - mean);
        abs_err += std::abs(truth[i] - pred[i]);
    }
    return {1.0 - ss_res / ss_tot, abs_err / truth.size()};
}

double seconds_since(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

}  // namespace

int main(int argc, char** argv) {
    const std::string type = argc > 1 ? argv[1] : "tet";
    const int n_particles = argc > 2 ? std::atoi(argv[2]) : 1000;
    const double beta = argc > 3 ? std::atof(argv[3]) : 0.8;

    const simple_mesh::Mesh mesh = type == "hex" ? simple_mesh::hex_mesh(10) : simple_mesh::tet_mesh(10);
    const double radius = beta * mesh.mean_equivalent_radius();
    std::printf("%s mesh, %zu cells, beta = %.2f, R_p = %.4f, %d particles\n", type.c_str(), mesh.cells.size(),
                beta, radius, n_particles);

    // Random particles fully inside the unit box, and their candidate cells.
    std::mt19937_64 rng(1);
    std::uniform_real_distribution<double> uniform(radius, 1.0 - radius);
    std::vector<Vec3> centers(n_particles);
    std::vector<std::vector<const Cell*>> candidates(n_particles);
    for (int p = 0; p < n_particles; ++p) {
        centers[p] = {uniform(rng), uniform(rng), uniform(rng)};
        candidates[p] = mesh.candidates(centers[p], radius);
    }

    // --- FBSI (timed) --------------------------------------------------
    const int repeats = 20;
    std::vector<std::vector<double>> phi_fbsi(n_particles);
    auto start = std::chrono::steady_clock::now();
    for (int r = 0; r < repeats; ++r)
        for (int p = 0; p < n_particles; ++p) fbsi::fbsi_fractions(centers[p], radius, candidates[p], phi_fbsi[p]);
    const double t_fbsi = seconds_since(start) / repeats;

    // --- PCM (timed): whole particle to the cell containing the centre -----
    std::vector<std::vector<double>> phi_pcm(n_particles);
    start = std::chrono::steady_clock::now();
    for (int r = 0; r < repeats; ++r)
        for (int p = 0; p < n_particles; ++p) {
            phi_pcm[p].assign(candidates[p].size(), 0.0);
            for (std::size_t c = 0; c < candidates[p].size(); ++c) {
                if (fbsi::contains(*candidates[p][c], centers[p])) {
                    phi_pcm[p][c] = 1.0;
                    break;
                }
            }
        }
    const double t_pcm = seconds_since(start) / repeats;

    // --- Direct-numerical reference -------------------------------------
    std::printf("computing the direct-numerical reference (N_V = 96) ...\n");
    std::vector<double> truth, pred_fbsi, pred_pcm;
    for (int p = 0; p < n_particles; ++p) {
        const std::vector<double> ref = fbsi::direct_numerical_fractions(centers[p], radius, candidates[p], 96);
        for (std::size_t c = 0; c < ref.size(); ++c) {
            if (ref[c] > 1e-5 || phi_fbsi[p][c] > 1e-5) {  // same pair selection as the paper
                truth.push_back(ref[c]);
                pred_fbsi.push_back(phi_fbsi[p][c]);
                pred_pcm.push_back(phi_pcm[p][c]);
            }
        }
    }

    const Stats s_fbsi = compare(truth, pred_fbsi);
    const Stats s_pcm = compare(truth, pred_pcm);
    std::printf("\n%zu particle-cell pairs\n\n", truth.size());
    std::printf("%-6s %8s %11s %16s\n", "method", "R^2", "mean |err|", "time/particle");
    std::printf("%-6s %8.4f %11.2e %13.0f ns\n", "FBSI", s_fbsi.r2, s_fbsi.mae, 1e9 * t_fbsi / n_particles);
    std::printf("%-6s %8.4f %11.2e %13.0f ns\n", "PCM", s_pcm.r2, s_pcm.mae, 1e9 * t_pcm / n_particles);
    std::printf("\nFBSI cost relative to PCM: %.1fx (same candidate lists)\n", t_fbsi / t_pcm);
    return 0;
}
