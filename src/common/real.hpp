#pragma once

// Floating-point type used by every version of the solver.
// Selected at configure time with -DLBM_REAL_DOUBLE=ON (default: float).
namespace lbm {

#ifdef LBM_REAL_DOUBLE
using real = double;
#else
using real = float;
#endif

constexpr const char* real_name = sizeof(real) == sizeof(double) ? "double" : "float";

}  // namespace lbm
