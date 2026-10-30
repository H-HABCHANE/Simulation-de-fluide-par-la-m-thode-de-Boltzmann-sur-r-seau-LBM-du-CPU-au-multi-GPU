// v4: multi-GPU (MPI + CUDA) solver; with LBM_ENABLE_CUDA=OFF, the same MPI code on CPU.
// Usage: mpirun -np P lbm_v4 [same options as lbm_v0]   (LBM_V4_OVERLAP=0 disables overlap)
#include <mpi.h>

#include "driver.hpp"
#include "solver_v4.hpp"

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  const int status = lbm::run_driver<lbm::SolverV4>(argc, argv, "v4");
  MPI_Finalize();
  return status;
}
