cmake_minimum_required(VERSION 3.0)
project(c_utils)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(PythonInterp 3.5 REQUIRED)
find_package(PythonLibs 3.5 REQUIRED)
find_package(pybind11 REQUIRED)
find_package(OpenMP REQUIRED)

pybind11_add_module(c_gen_depth_and_normal src/c_gen_depth_and_normal.cpp)
pybind11_add_module(c_gen_virtual_scan src/c_gen_virtual_scan.cpp)

