// v2: naive CUDA solver (block size from LBM_BLOCK, default 32x8).
// Same command line as lbm_v0.
#include "driver.hpp"
#include "solver_v2.hpp"

int main(int argc, char** argv) { return lbm::run_driver<lbm::SolverV2>(argc, argv, "v2"); }
