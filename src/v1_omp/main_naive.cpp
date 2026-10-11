// v1_naive: OpenMP port of v0 without optimisation (thread count from OMP_NUM_THREADS).
// Same command line as lbm_v0.
#include "driver.hpp"
#include "solver_v1_naive.hpp"

int main(int argc, char** argv) {
  return lbm::run_driver<lbm::SolverV1Naive>(argc, argv, "v1_naive");
}
