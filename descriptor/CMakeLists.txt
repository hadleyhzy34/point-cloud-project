cmake_minimum_required(VERSION 2.6)
project(descriptor)
 
find_package(PCL 1.2 REQUIRED)
 
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
 
add_executable(pcl_shot pcl_shot.cpp)
 
target_link_libraries (pcl_shot ${PCL_LIBRARIES})
 
install(TARGETS pcl_shot RUNTIME DESTINATION bin)
 
