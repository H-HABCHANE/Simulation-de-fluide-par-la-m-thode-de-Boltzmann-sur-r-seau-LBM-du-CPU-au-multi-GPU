#pragma once

// 1D domain decomposition in y: each rank owns a strip of consecutive rows. Rows are contiguous
// in memory (x varies fastest), so a halo row is one contiguous block per population plane.
// When ny is not a multiple of the number of ranks, the first ny % size ranks get one more
// row. Periodicity in y makes the ranks a ring.

#include <stdexcept>

namespace lbm {

struct Strip {
  int rank = 0, size = 1;
  int y0 = 0;        // first global row owned by this rank
  int ny_local = 0;  // number of owned rows
  int up = 0;        // rank owning the rows above (y0 + ny_local, periodic)
  int down = 0;      // rank owning the rows below (y0 - 1, periodic)
};

inline Strip make_strip(int ny, int rank, int size) {
  Strip s;
  s.rank = rank;
  s.size = size;
  const int base = ny / size, extra = ny % size;
  s.ny_local = base + (rank < extra ? 1 : 0);
  s.y0 = rank * base + (rank < extra ? rank : extra);
  s.up = (rank + 1) % size;
  s.down = (rank - 1 + size) % size;
  // Two owned rows at least: the bottom and top rows (those exchanged with the neighbours)
  // must be distinct for the overlap of computation and communication.
  if (s.ny_local < 2) throw std::invalid_argument("each MPI rank needs at least 2 rows");
  return s;
}

}  // namespace lbm
