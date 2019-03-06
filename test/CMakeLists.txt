set(SDA_TEST_SOURCES
  test_main.cpp)

if(TARGET Eigen3::Eigen)
  set(SDA_TEST_SOURCES
    ${SDA_TEST_SOURCES}
    test_eigen_backend.cpp)
endif()

add_executable(test_sda
  ${SDA_TEST_SOURCES})

target_include_directories(test_sda
  PRIVATE catch)

if(ENABLE_COMPILER_WARNINGS)
  target_compile_options(test_sda
    PRIVATE
    ${CXX_WARNING_FLAGS})
endif()

if(TARGET Eigen3::Eigen)
  target_compile_definitions(test_sda
    PRIVATE HAVE_EIGEN)
endif()

target_link_libraries(test_sda
  SDA::sda)

include(Catch)
catch_discover_tests(test_sda)

add_executable(SDA::test_sda ALIAS test_sda)

add_custom_target(check COMMAND "ctest")
add_dependencies(check test_sda)

