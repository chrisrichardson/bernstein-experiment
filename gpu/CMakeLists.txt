cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

set(CUDAToolkit_ROOT "/lsc/opt/cuda-12.6")
set(CMAKE_CUDA_COMPILER "/lsc/opt/cuda-12.6/bin/nvcc")

add_compile_definitions(PROJECT_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

project(BernsteinTest
        VERSION 1.0
        DESCRIPTION "BernsteinTest"
        LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
# set(CMAKE_CUDA_ARCHITECTURES 86)
set(CMAKE_CUDA_ARCHITECTURES 80)

# -x cu argument 
set_source_files_properties(main.cpp PROPERTIES LANGUAGE CUDA)
add_executable(BernsteinTest main.cpp)
