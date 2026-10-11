// Poiseuille flow test for v0.
//
// Channel periodic in x, halfway bounce-back walls at y = 0.5 and y = ny - 1.5
// (height H = ny - 2), flow driven by a constant body force (Guo scheme).
// The steady velocity profile is compared with the analytical parabola.
//
// Checks:
//   A. tau = 1/2 + sqrt(3)/4: BGK + halfway bounce-back is exact for this tau,
//      so the remaining error comes from floating-point arithmetic only.
//   B. tau = 1: the wall slip error is O(1/H^2).
//      double: the observed order between H = 16, 32, 64 must be close to 2.
//      float:  the order is not measurable (see below); only the error level is checked.
//   C. Mass is conserved (no source in the collision, bounce-back is conservative).
//
// Float tolerances: late in the transient, the per-step change of the populations falls
// below their float resolution and the run stagnates before the exact steady state. The
// measured stagnation error is ~1e-4 relative, the same size as the H = 64 discretisation
// error, so a convergence order cannot be measured in float. Tolerances below are about
// 3x the measured values.
#include <cmath>
#include <cstdio>
#include <vector>

#include "geometry.hpp"
#include "solver_v0.hpp"

namespace {

using namespace lbm;

#ifdef LBM_REAL_DOUBLE
constexpr bool kCheckOrder = true;
constexpr double kSteadyTol = 1e-10;  // relative profile change over kCheckEvery steps
constexpr double kExactTol = 1e-8;    // check A
constexpr double kErrorTol = 2e-3;    // check B, L2 error at every H
constexpr double kMassTol = 1e-13;    // check C, relative mass drift
#else
constexpr bool kCheckOrder = false;
constexpr double kSteadyTol = 1e-7;
constexpr double kExactTol = 2e-4;
constexpr double kErrorTol = 2e-3;
// Rounding errors on the populations add up like a random walk:
// eps * |g| * sqrt(steps * cells * 9) / cells ~ 2e-8 relative for these runs (measured: up to 1e-7).
constexpr double kMassTol = 1e-6;
#endif
constexpr int kCheckEvery = 1000;
constexpr int kNx = 4;  // the flow does not depend on x
constexpr double kUmax = 0.05;

struct Result {
  double l2_error;
  double mass_drift;
  long steps;
  bool steady;
};

// x-averaged ux profile over the fluid rows y = 1 .. ny - 2.
std::vector<double> profile(const SolverV0& s) {
  std::vector<real> rho, ux, uy;
  s.macroscopic(rho, ux, uy);
  std::vector<double> p(s.ny() - 2, 0.0);
  for (int y = 1; y < s.ny() - 1; ++y) {
    for (int x = 0; x < s.nx(); ++x) p[y - 1] += ux[static_cast<std::size_t>(y) * s.nx() + x];
    p[y - 1] /= s.nx();
  }
  return p;
}

double rel_l2(const std::vector<double>& a, const std::vector<double>& b) {
  double num = 0.0, den = 0.0;
  for (std::size_t k = 0; k < a.size(); ++k) {
    num += (a[k] - b[k]) * (a[k] - b[k]);
    den += b[k] * b[k];
  }
  return std::sqrt(num / den);
}

Result run_poiseuille(int H, double tau) {
  const int ny = H + 2;
  const real force = poiseuille_force(H, real(tau), real(kUmax));
  const double nu = static_cast<double>(d2q9::viscosity(real(tau)));

  SolverV0 s(kNx, ny, real(tau), force, real(0), make_channel(kNx, ny));
  s.init_equilibrium(real(1), real(0), real(0));
  const double mass0 = s.total_mass();

  // Safety cap: 20 viscous times H^2 / nu (about 200 decay times of the slowest mode).
  const long max_steps = static_cast<long>(20.0 * H * H / nu);
  std::vector<double> prev = profile(s);
  long steps = 0;
  bool steady = false;
  while (steps < max_steps) {
    s.run(kCheckEvery);
    steps += kCheckEvery;
    const std::vector<double> cur = profile(s);
    const bool done = rel_l2(cur, prev) < kSteadyTol;
    prev = cur;
    if (done) {
      steady = true;
      break;
    }
  }

  std::vector<double> exact(H);
  for (int k = 0; k < H; ++k)
    exact[k] = poiseuille_velocity(k + 0.5, H, nu, static_cast<double>(force));

  return {rel_l2(prev, exact), (s.total_mass() - mass0) / mass0, steps, steady};
}

int failures = 0;

void check(bool ok, const char* what) {
  std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what);
  if (!ok) ++failures;
}

void report(int H, double tau, const Result& r) {
  std::printf("  H=%3d tau=%.4f  L2=%.3e  steps=%ld%s  mass drift=%.2e\n", H, tau, r.l2_error,
              r.steps, r.steady ? "" : " (not steady)", r.mass_drift);
}

}  // namespace

int main() {
  std::printf("Poiseuille test, v0, real = %s\n", real_name);

  std::printf("A. exact tau\n");
  const double tau_exact = 0.5 + std::sqrt(3.0) / 4.0;
  const Result a = run_poiseuille(32, tau_exact);
  report(32, tau_exact, a);
  check(a.l2_error < kExactTol, "L2 error at round-off level");
  check(std::fabs(a.mass_drift) < kMassTol, "mass conserved");

  std::printf("B. tau = 1, H = 16, 32, 64\n");
  const int Hs[3] = {16, 32, 64};
  double err[3];
  for (int k = 0; k < 3; ++k) {
    const Result r = run_poiseuille(Hs[k], 1.0);
    report(Hs[k], 1.0, r);
    err[k] = r.l2_error;
    check(r.l2_error < kErrorTol, "L2 error below tolerance");
    check(std::fabs(r.mass_drift) < kMassTol, "mass conserved");
  }
  for (int k = 0; k < 2; ++k) {
    const double order = std::log2(err[k] / err[k + 1]);
    std::printf("  order H=%d -> %d: %.3f%s\n", Hs[k], Hs[k + 1], order,
                kCheckOrder ? "" : " (not checked in float)");
    if (kCheckOrder) check(order > 1.9 && order < 2.1, "second-order convergence");
  }

  std::printf("%s\n", failures == 0 ? "ALL PASSED" : "FAILED");
  return failures == 0 ? 0 : 1;
}
