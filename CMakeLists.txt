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
find_package(PCL  REQUIRED)
if (PCL_FOUND)
  message("-- You find the PCL success! --")
  message("Version: " ${PCL_VERSION})
  else()
  message("-- You find the PCL failed! --")
endif ()
find_package(Eigen3 QUIET)
if (Eigen3_FOUND)
  message("-- You find the Eigen3 success! --")
  message("Version: " ${Eigen3_VERSION})
  else()
  message("-- You find the Eigen3 failed! --")
endif ()
include_directories(
        ${CMAKE_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
        ${PCL_INCLUDE_DIRS}
)

link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

# cuda_add_library(gpu_speed_test SHARED
# include/gpu_speed_test.h
# src_cpp/gpu_speed_test.cu
# )

add_executable(mutilthread_test
        src_cpp/mutilthread.cpp         
)

target_link_libraries(mutilthread_test
        ${OpenCV_LIBS}
)

add_executable(func_test
src_cpp/func.cpp 
)
target_link_libraries(func_test
  ${OpenCV_LIBS}
)

add_executable(fillPoly_test
src_cpp/fillPolye_error_test.cpp 
)
target_link_libraries(fillPoly_test
  ${OpenCV_LIBS}
)

add_executable(main_test
        src_cpp/main.cpp 
        src_cpp/utils.cpp        
)

target_link_libraries(main_test
        ${OpenCV_LIBS}
        ${CUDA_cublas_LIBRARY}
        cudart
)

