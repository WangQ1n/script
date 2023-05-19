cmake_minimum_required(VERSION 3.2)
project(func_test)
add_compile_options(-std=c++14)
#TODO test if confilct with global CMakeListx.txt
# set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb  -Wl,-rpath=.")
set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3  -Wl,-rpath=.")

find_package(OpenCV  REQUIRED)
find_package(CUDA REQUIRED)
if (CUDA_FOUND)
  message("-- You find the cuda success! --")
  message("Version: " ${CUDA_VERSION})
  message("Library: " ${CUDA_CUDA_LIBRARY})
  message("Runtime: " ${CUDA_CUDART_LIBRARY})
endif()

include_directories(
        ${CMAKE_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
)

include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

cuda_add_library(gpu_speed_test SHARED
include/gpu_speed_test.h
src/gpu_speed_test.cu
)

add_executable(func_test
        src/main.cpp
        
)


target_link_libraries(func_test
        ${OpenCV_LIBS}
        gpu_speed_test
)

