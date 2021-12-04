cmake_minimum_required(VERSION 2.6)
project(registration)
 
find_package(PCL 1.5 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
 
add_executable(normal_distribution_transform normal_distribution_transform.cpp)

target_link_libraries(normal_distribution_transform ${PCL_LIBRARIES})
install(TARGETS normal_distribution_transform RUNTIME DESTINATION bin)
 
