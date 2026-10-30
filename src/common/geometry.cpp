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

Mask make_cylinder(int nx, int ny, double xc, double yc, double D) {
  Mask solid(static_cast<std::size_t>(nx) * ny, 0);
  const double r2 = 0.25 * D * D;
  for (int y = 0; y < ny; ++y)
    for (int x = 0; x < nx; ++x) {
      const double dx = x - xc, dy = y - yc;
      if (dx * dx + dy * dy <= r2) solid[static_cast<std::size_t>(y) * nx + x] = 1;
    }
  return solid;
}

}  // namespace lbm
