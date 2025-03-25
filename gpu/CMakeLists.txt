cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

project(BernsteinTest
        VERSION 1.0
        DESCRIPTION "BernsteinTest"
        LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES 86)

# -x cu argument 
set_source_files_properties(main.cpp PROPERTIES LANGUAGE CUDA)
add_executable(BernsteinTest main.cpp)
