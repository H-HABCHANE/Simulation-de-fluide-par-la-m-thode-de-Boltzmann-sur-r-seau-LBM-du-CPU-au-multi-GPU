// Poiseuille flow test run on the GPU with v2 (checks in poiseuille_check.hpp).
#include "poiseuille_check.hpp"
#include "solver_v2.hpp"

int main() { return poiseuille::run_all<lbm::SolverV2>("v2"); }
