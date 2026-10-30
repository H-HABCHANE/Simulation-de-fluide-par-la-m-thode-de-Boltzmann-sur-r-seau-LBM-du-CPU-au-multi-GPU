#pragma once

// Small cylinder flow shared by the cross-version tests of the open (inflow / outflow)
// boundaries: 120 x 48 grid, D = 10, inlet velocity 0.05, Re = 20 (tau = 0.575), 1000 steps.
// The flow starts from the uniform inlet state, as in the full simulation, so that the push
// (v0) and pull (v1, v3) states are identical from the first step.
#include <vector>

#include "d2q9.hpp"
#include "geometry.hpp"

namespace cylinder_case {

using namespace lbm;

inline constexpr int kNx = 120, kNy = 48, kD = 10, kSteps = 1000;
inline constexpr real kU = real(0.05);
inline constexpr double kRe = 20.0;

inline real tau() { return real(0.5 + 3.0 * static_cast<double>(kU) * kD / kRe); }
inline Mask geometry() { return make_cylinder(kNx, kNy, 3.0 * kD, kNy / 2 + 1, kD); }

template <class Solver>
std::vector<real> run() {
  Solver s(kNx, kNy, tau(), real(0), real(0), geometry());
  s.set_inflow_outflow(kU);
  s.init_equilibrium(real(1), kU, real(0));
  s.run(kSteps);
  const auto& g = s.populations();
  return {g.begin(), g.end()};
}

// Number of differing values and largest absolute difference.
inline std::size_t compare(const std::vector<real>& a, const std::vector<real>& b,
                           double& max_diff) {
  std::size_t n = 0;
  max_diff = 0.0;
  for (std::size_t k = 0; k < a.size(); ++k)
    if (a[k] != b[k]) {
      ++n;
      const double d = static_cast<double>(a[k]) - b[k];
      if (d > max_diff) max_diff = d;
      if (-d > max_diff) max_diff = -d;
    }
  return n;
}

}  // namespace cylinder_case
