#pragma once

#include <string>

#include "real.hpp"

namespace lbm {

// Run parameters, all in lattice units.
struct Params {
  std::string case_name = "poiseuille";
  int nx = 64;
  int ny = 34;         // includes the two solid wall rows for the channel case
  real tau = real(0.9);
  real u_max = real(0.05);  // target peak velocity, sets the driving force
  int steps = 20000;
  std::string out_dir = "output";
};

// Parses "--key value" pairs (--case, --nx, --ny, --tau, --umax, --steps, --out).
// Throws std::invalid_argument on unknown keys or invalid values.
Params parse_args(int argc, char** argv);

void print_params(const Params& p);

}  // namespace lbm
