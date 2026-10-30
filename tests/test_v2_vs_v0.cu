// v2 (naive CUDA) must reproduce v0 bit for bit.
//
// v2 performs the same floating-point operations as v0 in the same order. GPU single and
// double precision add, multiply and divide are IEEE round-to-nearest (no -use_fast_math),
// and v2 is compiled with --fmad=false (no fused multiply-add contraction), so the expected
// difference is exactly zero. Any non-zero difference is a bug, not round-off.
//
// Same non-uniform case as test_v1_vs_v0: channel walls plus a solid block, a force with
// both components, a non-zero initial velocity, grid sizes that are not multiples of the
// block size (so that the out-of-range threads of the last blocks are exercised).
#include <cmath>
#include <cstdio>
#include <vector>

#include "geometry.hpp"
#include "solver_v0.hpp"
#include "solver_v2.hpp"

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
  std::printf("v2 vs v0, %dx%d, %d steps, real = %s\n", kNx, kNy, kSteps, real_name);
  const std::vector<real> ref = run<SolverV0>();
  const std::vector<real> g = run<SolverV2>();

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
  std::printf("  [%s] differing values=%zu / %zu  max |diff|=%.3e\n", ok ? "PASS" : "FAIL",
              differing, ref.size(), max_diff);
  if (!ok)
    std::printf("  first difference: cell %zu, direction %zu\n", first / d2q9::Q, first % d2q9::Q);
  std::printf("%s\n", ok ? "ALL PASSED" : "FAILED");
  return ok ? 0 : 1;
}
