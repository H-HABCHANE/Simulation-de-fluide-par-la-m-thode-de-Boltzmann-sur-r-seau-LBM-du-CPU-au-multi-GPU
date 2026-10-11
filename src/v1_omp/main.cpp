// v1: OpenMP CPU solver (thread count from OMP_NUM_THREADS).
// Same command line as lbm_v0.
#include "driver.hpp"
#include "solver_v1.hpp"

int main(int argc, char** argv) { return lbm::run_driver<lbm::SolverV1>(argc, argv, "v1"); }
