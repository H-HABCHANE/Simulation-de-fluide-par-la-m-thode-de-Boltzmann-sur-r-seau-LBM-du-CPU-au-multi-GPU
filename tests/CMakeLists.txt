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
endif()
