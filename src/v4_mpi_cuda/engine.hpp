#pragma once

// Compute engine of one MPI rank: owns the populations of its strip (owned rows + 2 ghost
// rows, see strip_update.hpp) and the halo buffers. Two implementations, chosen at build time:
// GPU (engine_gpu.cu, LBM_ENABLE_CUDA=ON) and CPU (engine_cpu.cpp), the latter to test the MPI
// logic on machines without a GPU (laptop, CI).
//
// One time step, as driven by SolverV4::step():
//   compute_rows(first, last, kBoundary)   owned rows 1 and ny_local (those the neighbours need)
//   compute_rows(2, ny_local-1, kInterior) the rest; on the GPU it runs in a second stream,
//                                          overlapping the halo exchange
//   pack_halo()                            boundary-row populations -> send buffers (host)
//   MPI exchange (SolverV4)                send_up -> rank above, send_down -> rank below
//   unpack_halo()                          receive buffers -> ghost rows of the new state
//   finish_step()                          wait for everything, swap the two buffers
//
// Halo content (pull scheme): the ghost row below needs the populations going up
// (c_y = +1: i = 2, 5, 6) of the top row of the rank below; the ghost row above needs the
// populations going down (c_y = -1: i = 4, 7, 8) of the bottom row of the rank above.
// Each message is 3 * nx values.

#include <memory>
#include <string>
#include <vector>

#include "geometry.hpp"
#include "strip_update.hpp"

namespace lbm {

inline constexpr int kHaloUp[3] = {2, 5, 6};    // c_y = +1, sent to the rank above
inline constexpr int kHaloDown[3] = {4, 7, 8};  // c_y = -1, sent to the rank below

class StripEngine {
 public:
  enum Stream { kBoundary = 0, kInterior = 1 };
  virtual ~StripEngine() = default;

  // Whole strip, ghost rows included, structure of arrays (N = nx * (ny_local + 2) per plane).
  virtual void upload_state(const std::vector<real>& g) = 0;  // into both buffers
  virtual void download_state(std::vector<real>& g) const = 0;
  virtual void upload_solid(const Mask& solid_strip) = 0;
  virtual void set_params(const StripParams& p) = 0;

  virtual void compute_rows(int first, int last, Stream s) = 0;  // local rows, inclusive
  virtual void pack_halo() = 0;    // waits for the boundary rows, fills send buffers
  virtual void unpack_halo() = 0;  // receive buffers -> ghost rows of the new state
  virtual void finish_step() = 0;  // waits for all work, swaps the buffers

  // Host halo buffers, 3 * nx values each (page-locked on the GPU engine).
  virtual real* send_up() = 0;
  virtual real* send_down() = 0;
  virtual real* recv_from_down() = 0;
  virtual real* recv_from_up() = 0;

  virtual std::string backend() const = 0;      // "cpu" or "gpu"
  virtual std::string device_name() const = 0;  // GPU model, empty on the CPU engine
  virtual std::string device_id() const = 0;    // unique id of the device (PCI bus id)
};

// Defined in engine_cpu.cpp or engine_gpu.cu. local_rank selects the GPU (local_rank % count).
std::unique_ptr<StripEngine> make_engine(int nx, int ny_local, int local_rank);

}  // namespace lbm
