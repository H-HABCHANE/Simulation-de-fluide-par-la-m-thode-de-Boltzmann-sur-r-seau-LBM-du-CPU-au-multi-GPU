// v3: optimised CUDA solver (variant from LBM_V3_VARIANT, block size from LBM_BLOCK).
// Same command line as lbm_v0.
#include "driver.hpp"
#include "solver_v3.hpp"

int main(int argc, char** argv) { return lbm::run_driver<lbm::SolverV3>(argc, argv, "v3"); }
