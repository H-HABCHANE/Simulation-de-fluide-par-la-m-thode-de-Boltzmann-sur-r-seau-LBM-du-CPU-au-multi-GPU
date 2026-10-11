add_executable(test_real test_real.cpp)
target_link_libraries(test_real PRIVATE lbm_common)
add_test(NAME real_type COMMAND test_real)
