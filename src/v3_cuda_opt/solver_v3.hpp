#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "d2q9.hpp"
#include "geometry.hpp"

namespace lbm {

// v3: optimised CUDA solver, structure-of-arrays layout g[i * N + cell].
//
// The optimisation steps are kept as variants of one templated kernel, selected at run time
// with the environment variable LBM_V3_VARIANT, so that they can all be measured in the same
// session (GPU clocks on shared cloud machines vary between sessions):
//   soa_push      SoA layout, push streaming (as v2 otherwise): coalesced reads;
//   soa_pull      SoA layout, pull streaming: each thread writes its own cell (coalesced
//                 writes), reads are shifted by at most one cell and served by the caches;
//   soa_pull_opt  (default) pull without integer modulo: periodic neighbours by comparison.
//
// Pull stores post-collision populations R; v0 stores post-streaming ones S = stream(R).
// Streaming is a permutation of copies, so R is initialised with the inverse permutation of
// v0's initial state and every collision sees the same inputs as v0. Compiled with
// --fmad=false, all variants are bitwise identical to v0 (tested).
class SolverV3 {
 public:
  enum class Variant { SoaPush, SoaPull, SoaPullOpt };

  SolverV3(int nx, int ny, real tau, real force_x, real force_y, Mask solid);
  ~SolverV3();
  SolverV3(const SolverV3&) = delete;
  SolverV3& operator=(const SolverV3&) = delete;

  void init_equilibrium(real rho0, real ux0, real uy0);
  void step();
  void run(int steps);
  double timed_run(int steps);  // elapsed GPU time in seconds (CUDA events)

  // These copy the populations back to the host; populations() returns the post-streaming
  // populations in v0's layout ((y * nx + x) * 9 + i), for comparisons.
  void macroscopic(std::vector<real>& rho, std::vector<real>& ux, std::vector<real>& uy) const;
  double total_mass() const;
  std::vector<real> populations() const;

  int threads() const { return 0; }
  std::string bench_info() const;
  static const char* variant_name(Variant v);

  int nx() const { return nx_; }
  int ny() const { return ny_; }
  const Mask& solid() const { return solid_; }

 private:
  std::size_t cells() const { return static_cast<std::size_t>(nx_) * ny_; }
  bool pull() const { return variant_ != Variant::SoaPush; }

  int nx_, ny_;
  real tau_, fx_, fy_;
  Mask solid_;
  Variant variant_ = Variant::SoaPullOpt;
  int block_x_ = 32, block_y_ = 8;
  real* g_d_ = nullptr;
  real* g_new_d_ = nullptr;
  std::uint8_t* solid_d_ = nullptr;
};

}  // namespace lbm
