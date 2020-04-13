cmake_minimum_required(VERSION 3.10)
set(PROJECT_TARGET gstCuda)
set(CMAKE_C_COMPILER "/usr/bin/gcc")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -std=c++11")

set(CMAKE_VERBOSE_MAKEFILE 0)
file(GLOB SOURCES cudaWarp-fisheye.cu main.cpp)

find_package(CUDA REQUIRED)
cuda_add_executable(${PROJECT_TARGET} ${SOURCES})

# CUDA
target_link_libraries(${PROJECT_TARGET} ${CUDA_LIBRARIES})

# OPENCV
find_package( OpenCV REQUIRED core highgui )
target_link_libraries(${PROJECT_TARGET} ${OpenCV_LIBS})

# ARRAYFIRE
# target_include_directories(${PROJECT_TARGET} PUBLIC /opt/arrayfire/include)
# link_directories(${PROJECT_TARGET} /opt/arrayfire-3/lib)
# target_link_libraries(${PROJECT_TARGET} "afcpu;forge")
