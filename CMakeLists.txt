cmake_minimum_required(VERSION 3.10)
project(SNN_simulator)

# Force C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add the src folder to the include path
include_directories(src)

# Create an executable for the math tests
add_executable(run_math_tests tests/test_math.cpp)
add_executable(run_alif_tests tests/test_alif.cpp)
add_executable(run_eprop_tests tests/test_eprop.cpp)
add_executable(run_network_tests tests/test_network.cpp)
add_executable(run_benchmarks tests/test_benchmarks.cpp)
add_executable(run_encoder tests/test_encoder.cpp)
add_executable(run_integration tests/test_integration.cpp)
add_executable(run_shd tests/run_shd.cpp)