add_executable(test_real test_real.cpp)
target_link_libraries(test_real PRIVATE lbm_common)
add_test(NAME real_type COMMAND test_real)

add_executable(test_poiseuille test_poiseuille.cpp)
target_link_libraries(test_poiseuille PRIVATE lbm_v0_lib)
add_test(NAME poiseuille_v0 COMMAND test_poiseuille)

if(TARGET lbm_v1_lib)
  add_executable(test_v1_vs_v0 test_v1_vs_v0.cpp)
  target_link_libraries(test_v1_vs_v0 PRIVATE lbm_v0_lib lbm_v1_naive_lib lbm_v1_lib)
  add_test(NAME v1_matches_v0 COMMAND test_v1_vs_v0)

  add_executable(test_cylinder_v1 test_cylinder_v1.cpp)
  target_link_libraries(test_cylinder_v1 PRIVATE lbm_v0_lib lbm_v1_lib)
  add_test(NAME cylinder_v1_matches_v0 COMMAND test_cylinder_v1)
endif()

# GPU tests: built with LBM_ENABLE_CUDA=ON, they need a GPU to run.
if(TARGET lbm_v2_lib)
  add_executable(test_v2_vs_v0 test_v2_vs_v0.cu)
  target_link_libraries(test_v2_vs_v0 PRIVATE lbm_v0_lib lbm_v2_lib)
  add_test(NAME v2_matches_v0 COMMAND test_v2_vs_v0)

  add_executable(test_poiseuille_v2 test_poiseuille_v2.cu)
  target_link_libraries(test_poiseuille_v2 PRIVATE lbm_v2_lib)
  add_test(NAME poiseuille_v2 COMMAND test_poiseuille_v2)
endif()

if(TARGET lbm_v3_lib)
  add_executable(test_v3_vs_v0 test_v3_vs_v0.cu)
  target_link_libraries(test_v3_vs_v0 PRIVATE lbm_v0_lib lbm_v3_lib)
  add_test(NAME v3_matches_v0 COMMAND test_v3_vs_v0)

  add_executable(test_poiseuille_v3 test_poiseuille_v3.cu)
  target_link_libraries(test_poiseuille_v3 PRIVATE lbm_v3_lib)
  add_test(NAME poiseuille_v3 COMMAND test_poiseuille_v3)

  add_executable(test_cylinder_v3 test_cylinder_v3.cu)
  target_link_libraries(test_cylinder_v3 PRIVATE lbm_v0_lib lbm_v3_lib)
  add_test(NAME cylinder_v3_matches_v0 COMMAND test_cylinder_v3)
endif()

# MPI tests: v4 against v0 with 1 to 4 ranks (3 gives strips of unequal height).
if(TARGET lbm_v4_lib)
  add_executable(test_v4_vs_v0 test_v4_vs_v0.cpp)
  target_link_libraries(test_v4_vs_v0 PRIVATE lbm_v0_lib lbm_v4_lib)
  foreach(np 1 2 3 4)
    add_test(NAME v4_matches_v0_np${np}
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np} ${LBM_MPIEXEC_FLAGS}
                     ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_v4_vs_v0> ${MPIEXEC_POSTFLAGS})
  endforeach()
endif()
