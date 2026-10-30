// Open boundaries (inflow / outflow) and cylinder: v1 must reproduce v0 bit for bit.
// Inlet and outlet rules only copy values (boundary.hpp), so the pull scheme of v1 sees exactly
// the same inputs as the push scheme of v0. Also checks that the flow stays finite.
#include <omp.h>

#include <cmath>
#include <cstdio>

#include "cylinder_case.hpp"
#include "solver_v0.hpp"
#include "solver_v1.hpp"

int main() {
  using namespace cylinder_case;
  std::printf("cylinder, v1 vs v0, %dx%d, D=%d, tau=%.4f, %d steps, real = %s\n", kNx, kNy, kD,
              static_cast<double>(tau()), kSteps, real_name);
  const auto ref = run<SolverV0>();
  int failures = 0;
  bool finite = true;
  for (real v : ref) finite = finite && std::isfinite(static_cast<double>(v));
  std::printf("  [%s] v0 populations finite\n", finite ? "PASS" : "FAIL");
  if (!finite) ++failures;

  for (int threads : {1, 3, 4}) {
    omp_set_num_threads(threads);
    double max_diff = 0.0;
    const std::size_t n = compare(run<SolverV1>(), ref, max_diff);
    std::printf("  [%s] v1 threads=%d  differing values=%zu  max |diff|=%.3e\n",
                n == 0 ? "PASS" : "FAIL", threads, n, max_diff);
    if (n != 0) ++failures;
  }
  std::printf("%s\n", failures == 0 ? "ALL PASSED" : "FAILED");
  return failures == 0 ? 0 : 1;
}
