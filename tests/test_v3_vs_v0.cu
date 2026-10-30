// Every v3 variant must reproduce v0 bit for bit.
//
// Same reasoning as test_v2_vs_v0: same floating-point operations in the same order, IEEE
// arithmetic on the GPU, --fmad=false. The layout (SoA) and the streaming direction (pull)
// only move data; in pull mode the stored post-collision populations are streamed back
// (copies only) before the comparison. Any difference is a bug.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "geometry.hpp"
#include "solver_v0.hpp"
#include "solver_v3.hpp"

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

}  // namespace

int main() {
  std::printf("v3 vs v0, %dx%d, %d steps, real = %s\n", kNx, kNy, kSteps, real_name);
  const std::vector<real> ref = run<SolverV0>();

  int failures = 0;
  for (const char* variant : {"soa_push", "soa_pull", "soa_pull_opt"}) {
    setenv("LBM_V3_VARIANT", variant, 1);
    const std::vector<real> g = run<SolverV3>();
    std::size_t differing = 0, first = 0;
    double max_diff = 0.0;
    for (std::size_t k = 0; k < ref.size(); ++k) {
      if (g[k] != ref[k]) {
        if (differing == 0) first = k;
        ++differing;
        max_diff = std::fmax(max_diff, std::fabs(static_cast<double>(g[k]) - ref[k]));
      }
    }
    const bool ok = differing == 0;
    std::printf("  [%s] %-13s differing values=%zu / %zu  max |diff|=%.3e\n",
                ok ? "PASS" : "FAIL", variant, differing, ref.size(), max_diff);
    if (!ok) {
      std::printf("    first difference: cell %zu, direction %zu\n", first / d2q9::Q,
                  first % d2q9::Q);
      ++failures;
    }
  }
  std::printf("%s\n", failures == 0 ? "ALL PASSED" : "FAILED");
  return failures == 0 ? 0 : 1;
}
