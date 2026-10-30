#include "params.hpp"

#include <cstdio>
#include <stdexcept>

namespace lbm {

Params parse_args(int argc, char** argv) {
  Params p;
  for (int a = 1; a < argc; a += 2) {
    const std::string key = argv[a];
    if (key == "--bench") {  // flag without value
      p.bench = true;
      --a;
      continue;
    }
    if (a + 1 >= argc) throw std::invalid_argument("missing value for " + key);
    const std::string val = argv[a + 1];

    if (key == "--case") p.case_name = val;
    else if (key == "--nx") p.nx = std::stoi(val);
    else if (key == "--ny") p.ny = std::stoi(val);
    else if (key == "--tau") p.tau = static_cast<real>(std::stod(val));
    else if (key == "--umax") p.u_max = static_cast<real>(std::stod(val));
    else if (key == "--steps") p.steps = std::stoi(val);
    else if (key == "--out") p.out_dir = val;
    else if (key == "--warmup") p.warmup = std::stoi(val);
    else if (key == "--reps") p.reps = std::stoi(val);
    else if (key == "--re") p.re = std::stod(val);
    else if (key == "--diameter") p.diameter = std::stoi(val);
    else if (key == "--probe-every") p.probe_every = std::stoi(val);
    else if (key == "--frame-every") p.frame_every = std::stoi(val);
    else if (key == "--frame-start") p.frame_start = std::stoi(val);
    else throw std::invalid_argument("unknown option " + key);
  }

  if (p.nx < 1 || p.ny < 3) throw std::invalid_argument("grid too small");
  if (p.tau <= real(0.5)) throw std::invalid_argument("tau must be > 0.5 (positive viscosity)");
  if (p.steps < 0) throw std::invalid_argument("steps must be >= 0");
  if (p.warmup < 0 || p.reps < 1) throw std::invalid_argument("need warmup >= 0 and reps >= 1");
  if (p.re <= 0 || p.diameter < 2 || p.probe_every < 1 || p.frame_every < 0)
    throw std::invalid_argument("invalid cylinder parameters");
  return p;
}

void print_params(const Params& p) {
  std::printf("case=%s nx=%d ny=%d tau=%g umax=%g steps=%d real=%s out=%s\n",
              p.case_name.c_str(), p.nx, p.ny, static_cast<double>(p.tau),
              static_cast<double>(p.u_max), p.steps, real_name, p.out_dir.c_str());
}

}  // namespace lbm
