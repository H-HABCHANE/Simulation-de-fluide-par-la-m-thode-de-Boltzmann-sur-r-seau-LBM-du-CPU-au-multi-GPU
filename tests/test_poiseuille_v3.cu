// Poiseuille flow test run on the GPU with every v3 variant (checks in poiseuille_check.hpp).
#include <cstdlib>

#include "poiseuille_check.hpp"
#include "solver_v3.hpp"

int main() {
  int result = 0;
  for (const char* variant : {"soa_push", "soa_pull", "soa_pull_opt"}) {
    setenv("LBM_V3_VARIANT", variant, 1);
    const std::string name = std::string("v3 ") + variant;
    result |= poiseuille::run_all<lbm::SolverV3>(name.c_str());
  }
  return result;
}
