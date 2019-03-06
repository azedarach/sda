add_executable(test_sda
  test_main.cpp
  test_eigen_backend.cpp)

target_include_directories(test_sda
  PRIVATE catch)

if(ENABLE_COMPILER_WARNINGS)
  target_compile_options(test_sda
    PRIVATE
    ${CXX_WARNING_FLAGS})
endif()

target_link_libraries(test_sda
  SDA::sda)

include(Catch)
catch_discover_tests(test_sda)

add_executable(SDA::test_sda ALIAS test_sda)

add_custom_target(check COMMAND "ctest")
add_dependencies(check test_sda)

