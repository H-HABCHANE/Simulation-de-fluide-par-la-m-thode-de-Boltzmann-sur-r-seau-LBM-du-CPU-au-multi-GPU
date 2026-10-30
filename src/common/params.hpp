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

  // Cylinder case: inlet velocity = u_max, tau derived from the Reynolds number.
  double re = 100.0;
  int diameter = 40;
  int probe_every = 50;   // steps between two velocity probe samples
  int frame_every = 0;    // steps between two vorticity frames (0: none)
  int frame_start = 0;    // first step at which frames are written

  // Benchmark mode: no file output; `warmup` untimed steps, then `reps` timed runs of `steps`.
  bool bench = false;
  int warmup = 100;
  int reps = 5;
};

// Parses "--key value" pairs (--case, --nx, --ny, --tau, --umax, --steps, --out,
// --warmup, --reps, --re, --diameter, --probe-every, --frame-every, --frame-start)
// and the --bench flag.
// Throws std::invalid_argument on unknown keys or invalid values.
Params parse_args(int argc, char** argv);

void print_params(const Params& p);

}  // namespace lbm
