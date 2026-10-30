#include "io.hpp"

#include <cstring>
#include <fstream>
#include <stdexcept>

namespace lbm {

namespace {

template <typename T>
void write_npy_impl(const std::string& path, const std::vector<T>& a, int ny, int nx,
                    const char* descr) {
  if (a.size() != static_cast<std::size_t>(nx) * ny)
    throw std::invalid_argument("write_npy: size does not match shape");

  // Header dict, padded with spaces and ended by '\n' so that the data starts
  // on a 64-byte boundary (magic 6 + version 2 + length 2 + dict).
  std::string dict = std::string("{'descr': '") + descr +
                     "', 'fortran_order': False, 'shape': (" + std::to_string(ny) + ", " +
                     std::to_string(nx) + "), }";
  const std::size_t prefix = 10;
  const std::size_t total = ((prefix + dict.size() + 1 + 63) / 64) * 64;
  dict.append(total - prefix - dict.size() - 1, ' ');
  dict.push_back('\n');

  std::ofstream out(path, std::ios::binary);
  if (!out) throw std::runtime_error("cannot open " + path);
  out.write("\x93NUMPY", 6);
  const char version[2] = {1, 0};
  out.write(version, 2);
  const std::uint16_t len = static_cast<std::uint16_t>(dict.size());
  const char len_le[2] = {static_cast<char>(len & 0xff), static_cast<char>(len >> 8)};
  out.write(len_le, 2);
  out.write(dict.data(), static_cast<std::streamsize>(dict.size()));
  // Data is written in native byte order; the descr assumes little-endian (x86, ARM).
  out.write(reinterpret_cast<const char*>(a.data()),
            static_cast<std::streamsize>(a.size() * sizeof(T)));
  if (!out) throw std::runtime_error("write failed: " + path);
}

}  // namespace

void write_npy(const std::string& path, const std::vector<float>& a, int ny, int nx) {
  write_npy_impl(path, a, ny, nx, "<f4");
}
void write_npy(const std::string& path, const std::vector<double>& a, int ny, int nx) {
  write_npy_impl(path, a, ny, nx, "<f8");
}
void write_npy(const std::string& path, const std::vector<std::uint8_t>& a, int ny, int nx) {
  write_npy_impl(path, a, ny, nx, "|u1");
}

std::vector<real> vorticity(const std::vector<real>& ux, const std::vector<real>& uy,
                            const Mask& solid, int nx, int ny, bool periodic_x) {
  std::vector<real> w(ux.size(), real(0));
  for (int y = 0; y < ny; ++y) {
    const int yp = (y + 1) % ny, ym = (y - 1 + ny) % ny;
    for (int x = 0; x < nx; ++x) {
      const std::size_t c = static_cast<std::size_t>(y) * nx + x;
      if (solid[c]) continue;
      int xp = (x + 1) % nx, xm = (x - 1 + nx) % nx;
      real h = real(2);
      if (!periodic_x && x == 0) { xm = 0; h = real(1); }
      if (!periodic_x && x == nx - 1) { xp = nx - 1; h = real(1); }
      const real duy_dx = (uy[static_cast<std::size_t>(y) * nx + xp] -
                           uy[static_cast<std::size_t>(y) * nx + xm]) / h;
      const real dux_dy = (ux[static_cast<std::size_t>(yp) * nx + x] -
                           ux[static_cast<std::size_t>(ym) * nx + x]) / real(2);
      w[c] = duy_dx - dux_dy;
    }
  }
  return w;
}

}  // namespace lbm
