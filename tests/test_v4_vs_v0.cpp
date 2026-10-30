// v4 (MPI, strips of rows) must reproduce v0 bit for bit, whatever the number of ranks.
//
// The decomposition does not change any operation on a cell: halo rows only carry copies of
// the neighbours' populations. Run with mpirun -np 1, 2, 3, 4 (3 gives strips of unequal
// height). Two cases: periodic channel with an obstacle and a body force, and the cylinder
// with inflow / outflow boundaries; each with and without overlap of the halo exchange.
#include <mpi.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "cylinder_case.hpp"
#include "solver_v0.hpp"
#include "solver_v4.hpp"

namespace {

using namespace lbm;

// Periodic case of test_v1_vs_v0: 53 x 31 channel, solid block, force in x and y.
constexpr int kNx = 53, kNy = 31, kSteps = 500;
constexpr real kTau = real(0.8), kFx = real(1e-5), kFy = real(2e-6);

Mask obstacle_geometry() {
  Mask solid = make_channel(kNx, kNy);
  for (int y = 10; y < 17; ++y)
    for (int x = 12; x < 16; ++x) solid[static_cast<std::size_t>(y) * kNx + x] = 1;
  return solid;
}

template <class Solver>
std::vector<real> run_obstacle() {
  Solver s(kNx, kNy, kTau, kFx, kFy, obstacle_geometry());
  s.init_equilibrium(real(1), real(0.02), real(0.01));
  s.run(kSteps);
  const auto& g = s.populations();
  return {g.begin(), g.end()};
}

}  // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (rank == 0) std::printf("v4 vs v0, %d MPI ranks, real = %s\n", size, real_name);

  // Reference results, computed once by rank 0.
  std::vector<real> ref_obstacle, ref_cylinder;
  if (rank == 0) {
    ref_obstacle = run_obstacle<SolverV0>();
    ref_cylinder = cylinder_case::run<SolverV0>();
  }

  int failures = 0;
  for (const char* overlap : {"1", "0"}) {
    setenv("LBM_V4_OVERLAP", overlap, 1);
    const std::vector<real> obstacle = run_obstacle<SolverV4>();      // collective
    const std::vector<real> cylinder = cylinder_case::run<SolverV4>();  // collective
    if (rank == 0) {
      double d1 = 0.0, d2 = 0.0;
      const std::size_t n1 = cylinder_case::compare(obstacle, ref_obstacle, d1);
      const std::size_t n2 = cylinder_case::compare(cylinder, ref_cylinder, d2);
      std::printf("  [%s] overlap=%s obstacle (%dx%d): differing values=%zu  max |diff|=%.3e\n",
                  n1 == 0 ? "PASS" : "FAIL", overlap, kNx, kNy, n1, d1);
      std::printf("  [%s] overlap=%s cylinder (%dx%d): differing values=%zu  max |diff|=%.3e\n",
                  n2 == 0 ? "PASS" : "FAIL", overlap, cylinder_case::kNx, cylinder_case::kNy,
                  n2, d2);
      failures += (n1 != 0) + (n2 != 0);
    }
  }
  MPI_Bcast(&failures, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (rank == 0) std::printf("%s\n", failures == 0 ? "ALL PASSED" : "FAILED");
  MPI_Finalize();
  return failures == 0 ? 0 : 1;
}
