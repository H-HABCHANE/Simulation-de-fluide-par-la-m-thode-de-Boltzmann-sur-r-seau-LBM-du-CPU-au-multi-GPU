// Checks that the 'real' type matches the LBM_REAL_DOUBLE build option.
#include <cstdio>

#include "real.hpp"

int main() {
#ifdef LBM_REAL_DOUBLE
  constexpr bool ok = sizeof(lbm::real) == sizeof(double);
#else
  constexpr bool ok = sizeof(lbm::real) == sizeof(float);
#endif
  std::printf("real = %s: %s\n", lbm::real_name, ok ? "OK" : "FAIL");
  return ok ? 0 : 1;
}
