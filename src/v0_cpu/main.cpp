// v0: sequential CPU reference (placeholder until the solver is implemented).
#include <cstdio>

#include "real.hpp"

int main() {
  std::printf("lbm_v0: real = %s (%zu bytes)\n", lbm::real_name, sizeof(lbm::real));
  return 0;
}
