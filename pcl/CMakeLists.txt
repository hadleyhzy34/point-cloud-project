cmake_minimum_required(VERSION 2.6)
project(pcl)
 
find_package(PCL 1.2 REQUIRED)
 
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
 
add_executable(pcl_fpfh pcl_fpfh.cpp)
 
target_link_libraries (pcl_fpfh ${PCL_LIBRARIES})
 
install(TARGETS pcl_fpfh RUNTIME DESTINATION bin)
 
