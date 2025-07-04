cmake_minimum_required(VERSION 3.10)

# 项目名称
project(LidarSearchComparison)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 添加编译选项
add_compile_options(-Wall -Wextra -O3)

# 添加头文件路径
include_directories(
  ${PROJECT_SOURCE_DIR}/libs/nanoflann   # nanoflann 库头文件路径
  ${PROJECT_SOURCE_DIR}/libs/ikdtree    # ikdtree 库头文件路径
)

# 简化版本（不依赖ROS和PCL）
add_executable(LidarSearchComparison_simple src/main_simple.cpp libs/ikdtree/ikd_tree.cpp)
target_link_libraries(LidarSearchComparison_simple pthread)

# 如果有ROS环境，编译完整版本
find_package(catkin QUIET COMPONENTS
  roscpp
  sensor_msgs
  pcl_ros
  pcl_conversions
  rosbag
)

find_package(PCL QUIET)

if(catkin_FOUND AND PCL_FOUND)
  message(STATUS "Found ROS and PCL, building full version")
  
  # 添加头文件路径
  include_directories(
    ${catkin_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
  )
  
  # 完整版本（依赖ROS和PCL）
  add_executable(LidarSearchComparison
    src/main.cpp
    libs/ikdtree/ikd_tree.cpp  # 添加ikdtree源文件
  )
  
  # 添加定义
  target_compile_definitions(LidarSearchComparison PRIVATE
    ${PCL_DEFINITIONS}
  )
  
  # 链接依赖的库
  target_link_libraries(LidarSearchComparison
    ${catkin_LIBRARIES}   # ROS 库
    ${PCL_LIBRARIES}      # PCL 库
    pthread
  )
else()
  message(STATUS "ROS or PCL not found, only building simple version")
endif()
