// v1_naive and v1 (OpenMP) must reproduce v0 bit for bit.
//
// Each cell performs exactly the same floating-point operations in the same order in all
// versions (layout and loop changes only move data), and step() contains no reduction, so the
// expected difference is exactly zero for any thread count. Any non-zero difference is a bug
// (e.g. a race or a wrong neighbour), not round-off.
//
// The case is deliberately not uniform in x: channel walls plus a solid block, a force with
// both components and a non-zero initial velocity, on a grid whose sizes are not multiples
// of the thread counts.
#include <omp.h>

#include <cmath>
#include <cstdio>
#include <vector>

#include "geometry.hpp"
#include "solver_v0.hpp"
#include "solver_v1.hpp"
#include "solver_v1_naive.hpp"

namespace {

using namespace lbm;

constexpr int kNx = 53, kNy = 31, kSteps = 500;
constexpr real kTau = real(0.8), kFx = real(1e-5), kFy = real(2e-6);

Mask make_geometry() {
  Mask solid = make_channel(kNx, kNy);
  for (int y = 10; y < 17; ++y)
    for (int x = 12; x < 16; ++x) solid[static_cast<std::size_t>(y) * kNx + x] = 1;
  return solid;
}

template <class Solver>
std::vector<real> run() {
  Solver s(kNx, kNy, kTau, kFx, kFy, make_geometry());
  s.init_equilibrium(real(1), real(0.02), real(0.01));
  s.run(kSteps);
  const auto& g = s.populations();
  return {g.begin(), g.end()};
}

template <class Solver>
int compare(const char* name, const std::vector<real>& ref) {
  int failures = 0;
  for (int threads : {1, 2, 3, 4, 8}) {
    omp_set_num_threads(threads);
    const std::vector<real> g = run<Solver>();
    std::size_t differing = 0;
    double max_diff = 0.0;
    for (std::size_t k = 0; k < ref.size(); ++k) {
      if (g[k] != ref[k]) {
        ++differing;
        max_diff = std::fmax(max_diff, std::fabs(static_cast<double>(g[k]) - ref[k]));
      }
    }
    const bool ok = differing == 0;
    std::printf("  [%s] %-8s threads=%d  differing values=%zu  max |diff|=%.3e\n",
                ok ? "PASS" : "FAIL", name, threads, differing, max_diff);
    if (!ok) ++failures;
  }
  return failures;
}

}  // namespace

int main() {
  std::printf("v1 vs v0, %dx%d, %d steps, real = %s\n", kNx, kNy, kSteps, real_name);
  const std::vector<real> ref = run<SolverV0>();
  const int failures = compare<SolverV1Naive>("v1_naive", ref) + compare<SolverV1>("v1", ref);
  std::printf("%s\n", failures == 0 ? "ALL PASSED" : "FAILED");
  return failures == 0 ? 0 : 1;
}
