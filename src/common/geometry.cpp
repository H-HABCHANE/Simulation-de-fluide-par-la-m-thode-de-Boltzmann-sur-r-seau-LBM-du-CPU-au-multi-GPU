#include "geometry.hpp"

namespace lbm {

Mask make_channel(int nx, int ny) {
  Mask solid(static_cast<std::size_t>(nx) * ny, 0);
  for (int x = 0; x < nx; ++x) {
    solid[x] = 1;
    solid[static_cast<std::size_t>(ny - 1) * nx + x] = 1;
  }
  return solid;
}

}  // namespace lbm
