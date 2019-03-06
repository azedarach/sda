cmake_minimum_required(VERSION 3.0.0 FATAL_ERROR)

project(SDA VERSION 0.1.0 LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE "Release")
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)

include(SDABuildOptions)

if(NOT SDA_INSTALL_INCLUDE_DIR)
	set(SDA_INSTALL_INCLUDE_DIR include/sda
		CACHE PATH
		"The directory relative to CMAKE_INSTALL_PREFIX where SDA headers are installed")
endif()

find_package(Eigen3 3.1)
find_package(CUDA)

if(ENABLE_COMPILER_WARNINGS)
	include(add_cxx_flag_if_supported)
	include(CompilerWarnings)
endif()

add_library(sda INTERFACE)
target_include_directories(sda INTERFACE
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/sda>
	$<INSTALL_INTERFACE:${SDA_INSTALL_INCLUDE_DIR}>)

if(ENABLE_COMPILER_WARNINGS)
	target_compile_options(sda INTERFACE
		${CXX_WARNING_FLAGS})
endif()

install(TARGETS
	sda
	EXPORT SDATargets)

add_library(SDA::sda ALIAS sda)

if(ENABLE_TESTING)
	enable_testing()
	add_subdirectory(test EXCLUDE_FROM_ALL)
endif()