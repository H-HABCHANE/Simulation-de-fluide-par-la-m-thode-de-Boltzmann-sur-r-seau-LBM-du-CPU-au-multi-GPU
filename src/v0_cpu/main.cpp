// v0: sequential CPU reference solver.
// Usage: lbm_v0 [--case poiseuille] [--nx N] [--ny N] [--tau T] [--umax U] [--steps N] [--out DIR]
#include <chrono>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <fstream>

#include "geometry.hpp"
#include "io.hpp"
#include "params.hpp"
#include "solver_v0.hpp"

int main(int argc, char** argv) {
  using namespace lbm;
  try {
    const Params p = parse_args(argc, argv);
    if (p.case_name != "poiseuille") {
      std::fprintf(stderr, "unknown case '%s' (available: poiseuille)\n", p.case_name.c_str());
      return 1;
    }
    print_params(p);

    const int H = p.ny - 2;
    const real force = poiseuille_force(H, p.tau, p.u_max);
    SolverV0 solver(p.nx, p.ny, p.tau, force, real(0), make_channel(p.nx, p.ny));
    solver.init_equilibrium(real(1), real(0), real(0));
    const double mass0 = solver.total_mass();

    const auto t0 = std::chrono::steady_clock::now();
    solver.run(p.steps);
    const auto t1 = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(t1 - t0).count();

    // Indicative only: the benchmark script (bench/) does warm-up and repetitions.
    const double mlups = static_cast<double>(p.nx) * p.ny * p.steps / seconds / 1e6;
    std::printf("time=%.3f s  MLUPS=%.2f (indicative)  mass drift=%.3e\n", seconds, mlups,
                (solver.total_mass() - mass0) / mass0);

    std::vector<real> rho, ux, uy;
    solver.macroscopic(rho, ux, uy);
    const std::vector<real> vort = vorticity(ux, uy, solver.solid(), p.nx, p.ny);

    std::filesystem::create_directories(p.out_dir);
    const std::string dir = p.out_dir + "/";
    write_npy(dir + "rho.npy", rho, p.ny, p.nx);
    write_npy(dir + "ux.npy", ux, p.ny, p.nx);
    write_npy(dir + "uy.npy", uy, p.ny, p.nx);
    write_npy(dir + "vorticity.npy", vort, p.ny, p.nx);
    write_npy(dir + "solid.npy", solver.solid(), p.ny, p.nx);

    std::ofstream meta(dir + "meta.json");
    meta << "{\"version\": \"v0\", \"case\": \"" << p.case_name << "\", \"nx\": " << p.nx
         << ", \"ny\": " << p.ny << ", \"tau\": " << static_cast<double>(p.tau)
         << ", \"umax\": " << static_cast<double>(p.u_max)
         << ", \"force_x\": " << static_cast<double>(force) << ", \"steps\": " << p.steps
         << ", \"real\": \"" << real_name << "\"}\n";
    std::printf("fields written to %s\n", p.out_dir.c_str());
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}
