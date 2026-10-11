#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "geometry.hpp"
#include "real.hpp"

namespace lbm {

// Writes a 2D row-major array of shape (ny, nx) in NumPy .npy format (version 1.0).
void write_npy(const std::string& path, const std::vector<float>& a, int ny, int nx);
void write_npy(const std::string& path, const std::vector<double>& a, int ny, int nx);
void write_npy(const std::string& path, const std::vector<std::uint8_t>& a, int ny, int nx);

// Vorticity dUy/dx - dUx/dy with second-order central differences (periodic wrap).
// Set to zero on solid cells.
std::vector<real> vorticity(const std::vector<real>& ux, const std::vector<real>& uy,
                            const Mask& solid, int nx, int ny);

}  // namespace lbm
