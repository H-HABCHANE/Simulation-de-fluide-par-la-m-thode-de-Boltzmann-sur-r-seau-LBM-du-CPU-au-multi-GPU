// Open boundaries (inflow / outflow) and cylinder on the GPU: every v3 variant must reproduce
// v0 bit for bit (same reasoning as test_cylinder_v1: the boundary rules only copy values).
#include <cstdio>
#include <cstdlib>

#include "cylinder_case.hpp"
#include "solver_v0.hpp"
#include "solver_v3.hpp"

int main() {
  using namespace cylinder_case;
  std::printf("cylinder, v3 vs v0, %dx%d, D=%d, tau=%.4f, %d steps, real = %s\n", kNx, kNy, kD,
              static_cast<double>(tau()), kSteps, real_name);
  const auto ref = run<SolverV0>();
  int failures = 0;
  for (const char* variant : {"soa_push", "soa_pull", "soa_pull_opt"}) {
    setenv("LBM_V3_VARIANT", variant, 1);
    double max_diff = 0.0;
    const std::size_t n = compare(run<SolverV3>(), ref, max_diff);
    std::printf("  [%s] %-13s differing values=%zu  max |diff|=%.3e\n", n == 0 ? "PASS" : "FAIL",
                variant, n, max_diff);
    if (n != 0) ++failures;
  }
  std::printf("%s\n", failures == 0 ? "ALL PASSED" : "FAILED");
  return failures == 0 ? 0 : 1;
}
