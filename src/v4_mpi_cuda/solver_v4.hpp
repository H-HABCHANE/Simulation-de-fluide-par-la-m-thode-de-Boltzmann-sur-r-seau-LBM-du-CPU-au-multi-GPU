#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "decomposition.hpp"
#include "engine.hpp"
#include "geometry.hpp"

namespace lbm {

// v4: multi-GPU solver, one MPI rank per GPU, domain decomposed in strips of rows (y).
// Numerics of v3 (SoA, pull, no modulo): bitwise identical to v0 for any number of ranks.
//
// Every rank builds the same global geometry and keeps its strip plus one ghost row on each
// side; after each step it exchanges the 3 populations that cross each strip boundary
// (engine.hpp). Halos go through page-locked host buffers (no CUDA-aware MPI assumed).
// Overlap (default, LBM_V4_OVERLAP=0 to disable): the two boundary rows are computed first,
// then the interior rows run in a second stream while the halos are copied and exchanged.
//
// The interface is the one the shared driver expects; collective calls (all ranks must call
// them): every method except the accessors. Output methods gather the global fields on every
// rank; is_root() tells the driver which rank writes files.
class SolverV4 {
 public:
  SolverV4(int nx, int ny, real tau, real force_x, real force_y, Mask solid);
  ~SolverV4();

  void set_inflow_outflow(real u_in);  // call before init_equilibrium()
  void init_equilibrium(real rho0, real ux0, real uy0);
  void step();
  void run(int steps);
  double timed_run(int steps);  // wall time, maximum over the ranks

  void macroscopic(std::vector<real>& rho, std::vector<real>& ux, std::vector<real>& uy) const;
  double total_mass() const;
  std::vector<real> populations() const;  // global, post-streaming, v0 layout
  void populations_at(int x, int y, real* out) const;

  int threads() const { return 0; }
  std::string bench_info() const;
  bool is_root() const { return strip_.rank == 0; }

  int nx() const { return nx_; }
  int ny() const { return ny_; }
  const Mask& solid() const { return solid_; }

 private:
  std::size_t strip_cells() const { return static_cast<std::size_t>(nx_) * (strip_.ny_local + 2); }
  int global_row(int local_row) const { return (strip_.y0 - 1 + local_row + ny_) % ny_; }
  void exchange_halo();
  // Post-streaming populations of the owned rows of this rank, v0 layout.
  std::vector<real> local_populations() const;

  int nx_, ny_;
  real tau_, fx_, fy_;
  Mask solid_;        // global
  Mask solid_strip_;  // owned rows + ghost rows
  Strip strip_;
  bool open_x_ = false;
  bool overlap_ = true;
  real eq_in_[d2q9::Q] = {};
  StripParams params_{};
  std::unique_ptr<StripEngine> engine_;
};

}  // namespace lbm
