cmake_minimum_required(VERSION 2.6)
project(registration)
 
find_package(PCL 1.8 REQUIRED)

find_package(OpenMP)

if (OPENMP_FOUND)

set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

endif()

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
 
add_executable(normal_distribution_transform normal_distribution_transform.cpp)

include_directories(include)
add_library(ndt_omp src/pclomp/voxel_grid_covariance_omp.cpp src/pclomp/ndt_omp.cpp src/pclomp/gicp_omp.cpp)
add_dependencies(normal_distribution_transform ndt_omp)
target_link_libraries(normal_distribution_transform ${PCL_LIBRARIES} ndt_omp)

install(TARGETS normal_distribution_transform RUNTIME DESTINATION bin)
 
