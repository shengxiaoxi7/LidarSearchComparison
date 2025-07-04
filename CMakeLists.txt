cmake_minimum_required(VERSION 3.10)

project(LidarSearchComparison)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_options(-Wall -Wextra -O3)

# 添加头文件路径
include_directories(
  ${PROJECT_SOURCE_DIR}/libs/nanoflann
  ${PROJECT_SOURCE_DIR}/libs/ikdtree
)

# 构建简单版本
add_executable(LidarSearchComparison_simple
  src/main_simple.cpp
)
target_link_libraries(LidarSearchComparison_simple pthread)

# 查找 ROS 和 PCL 包
find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  pcl_ros
  pcl_conversions
  rosbag
)

find_package(PCL REQUIRED)

# 如果 ROS 和 PCL 环境存在，构建完整版本
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

add_executable(LidarSearchComparison
  src/main.cpp
  libs/ikdtree/ikd_tree.cpp
)

target_compile_definitions(LidarSearchComparison PRIVATE
  ${PCL_DEFINITIONS}
)

target_link_libraries(LidarSearchComparison
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  pthread
)
