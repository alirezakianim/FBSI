cmake_minimum_required(VERSION 3.14)
project(fbsi VERSION 1.0.2 LANGUAGES CXX)

# Header-only library: link to fbsi::fbsi or just copy include/fbsi/fbsi.hpp.
add_library(fbsi INTERFACE)
add_library(fbsi::fbsi ALIAS fbsi)
target_include_directories(fbsi INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_compile_features(fbsi INTERFACE cxx_std_17)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(FBSI_BUILD_EXAMPLES "Build the examples" ON)
option(FBSI_BUILD_TESTS "Build the tests" ON)

if(FBSI_BUILD_EXAMPLES)
    foreach(example quickstart mesh_benchmark)
        add_executable(${example} examples/${example}.cpp)
        target_link_libraries(${example} PRIVATE fbsi::fbsi)
    endforeach()
endif()

if(FBSI_BUILD_TESTS)
    enable_testing()
    add_executable(fbsi_tests tests/test_fbsi.cpp)
    target_include_directories(fbsi_tests PRIVATE examples)
    target_link_libraries(fbsi_tests PRIVATE fbsi::fbsi)
    add_test(NAME fbsi_tests COMMAND fbsi_tests)
endif()

install(DIRECTORY include/ DESTINATION include)
