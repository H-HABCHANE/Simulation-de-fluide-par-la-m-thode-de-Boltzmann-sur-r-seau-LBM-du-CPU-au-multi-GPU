// CPU engine of v4: plain sequential loops over the strip. It exists to run and test the MPI
// decomposition and halo exchange without a GPU; it is not meant to be fast.
#include <algorithm>
#include <cstring>
#include <utility>

#include "engine.hpp"

namespace lbm {

namespace {

class CpuEngine final : public StripEngine {
 public:
  CpuEngine(int nx, int ny_local)
      : nx_(nx),
        nyl_(ny_local),
        N_(static_cast<std::size_t>(nx) * (ny_local + 2)),
        g_(N_ * d2q9::Q, real(0)),
        g_new_(N_ * d2q9::Q, real(0)),
        halo_(4 * 3 * static_cast<std::size_t>(nx), real(0)) {}

  void upload_state(const std::vector<real>& g) override { g_ = g_new_ = g; }
  void download_state(std::vector<real>& g) const override { g = g_; }
  void upload_solid(const Mask& solid_strip) override { solid_ = solid_strip; }
  void set_params(const StripParams& p) override { params_ = p; }

  void compute_rows(int first, int last, Stream) override {
    for (int y = first; y <= last; ++y)
      for (int x = 0; x < nx_; ++x)
        strip_update_cell(g_.data(), g_new_.data(), solid_.data(), x, y, params_);
  }

  void pack_halo() override {
    for (int k = 0; k < 3; ++k) {
      std::memcpy(send_up() + k * nx_, row(g_new_, kHaloUp[k], nyl_), nx_ * sizeof(real));
      std::memcpy(send_down() + k * nx_, row(g_new_, kHaloDown[k], 1), nx_ * sizeof(real));
    }
  }

  void unpack_halo() override {
    for (int k = 0; k < 3; ++k) {
      std::memcpy(row(g_new_, kHaloUp[k], 0), recv_from_down() + k * nx_, nx_ * sizeof(real));
      std::memcpy(row(g_new_, kHaloDown[k], nyl_ + 1), recv_from_up() + k * nx_,
                  nx_ * sizeof(real));
    }
  }

  void finish_step() override { std::swap(g_, g_new_); }

  real* send_up() override { return halo_.data(); }
  real* send_down() override { return halo_.data() + 3 * nx_; }
  real* recv_from_down() override { return halo_.data() + 6 * nx_; }
  real* recv_from_up() override { return halo_.data() + 9 * nx_; }

  std::string backend() const override { return "cpu"; }
  std::string device_name() const override { return ""; }
  std::string device_id() const override { return ""; }

 private:
  real* row(std::vector<real>& g, int i, int y) {
    return g.data() + i * N_ + static_cast<std::size_t>(y) * nx_;
  }

  int nx_, nyl_;
  std::size_t N_;
  std::vector<real> g_, g_new_, halo_;
  Mask solid_;
  StripParams params_{};
};

}  // namespace

std::unique_ptr<StripEngine> make_engine(int nx, int ny_local, int /*local_rank*/) {
  return std::make_unique<CpuEngine>(nx, ny_local);
}

}  // namespace lbm
