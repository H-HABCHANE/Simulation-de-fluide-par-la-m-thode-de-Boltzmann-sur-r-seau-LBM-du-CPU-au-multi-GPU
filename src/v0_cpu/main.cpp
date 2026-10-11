// v0: sequential CPU reference solver.
// Usage: lbm_v0 [--nx N] [--ny N] [--tau T] [--umax U] [--steps N] [--out DIR]
//        lbm_v0 --bench [--nx N] [--ny N] [--steps N] [--warmup N] [--reps R]
#include "driver.hpp"
#include "solver_v0.hpp"

int main(int argc, char** argv) { return lbm::run_driver<lbm::SolverV0>(argc, argv, "v0"); }
