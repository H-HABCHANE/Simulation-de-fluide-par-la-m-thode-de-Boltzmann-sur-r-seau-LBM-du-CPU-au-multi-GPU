// Poiseuille flow test for v0 (checks in poiseuille_check.hpp).
#include "poiseuille_check.hpp"
#include "solver_v0.hpp"

int main() { return poiseuille::run_all<lbm::SolverV0>("v0"); }
