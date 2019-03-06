cmake_minimum_required(VERSION 3.0.0 FATAL_ERROR)

project(SDA VERSION 0.1.0 LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)

include(SDABuildOptions)

if(NOT SDA_INSTALL_INCLUDE_DIR)
  set(SDA_INSTALL_INCLUDE_DIR include
    CACHE PATH
    "The directory relative to CMAKE_INSTALL_PREFIX where SDA headers are installed")
endif()

if(NOT SDA_INSTALL_CMAKE_DIR)
  set(SDA_INSTALL_CMAKE_DIR share/sda
    CACHE PATH
    "The directory relative to CMAKE_INSTALL_PREFIX where SHA CMake files are installed")
endif()

find_package(Eigen3 3.1)

if(ENABLE_COMPILER_WARNINGS)
  include(add_cxx_flag_if_supported)
  include(CompilerWarnings)
endif()

add_library(sda INTERFACE)
target_include_directories(sda INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:${SDA_INSTALL_INCLUDE_DIR}>)

if(ENABLE_COMPILER_WARNINGS)
  target_compile_options(sda INTERFACE
    ${CXX_WARNING_FLAGS})
endif()

install(DIRECTORY sda DESTINATION ${SDA_INSTALL_INCLUDE_DIR})

install(TARGETS
  sda
  EXPORT SDATargets)

if(TARGET Eigen3::Eigen)
  target_link_libraries(sda
    INTERFACE
    Eigen3::Eigen)
endif()

add_library(SDA::sda ALIAS sda)

export(TARGETS sda
  NAMESPACE SDA::
  FILE "${PROJECT_BINARY_DIR}/${SDA_INSTALL_CMAKE_DIR}/SDATargets.cmake")

export(PACKAGE SDA)

if(ENABLE_TESTING)
  enable_testing()
  add_subdirectory(test EXCLUDE_FROM_ALL)
endif()

include(SDAGenerateConfig)
generate_sda_config_files()
