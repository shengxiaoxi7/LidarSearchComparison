cmake_minimum_required(VERSION 3.10)

project(LidarSearchComparison)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_options(-Wall -Wextra -O3)

# 包含头文件路径
include_directories(
  ${PROJECT_SOURCE_DIR}/include/ikdtree   # ikdtree 库头文件路径
  ${PROJECT_SOURCE_DIR}/include/nanoflann  # nanoflann 库头文件路径
)

# 查找 PCL 包
find_package(PCL REQUIRED)

# 查找 ROS 包
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  sensor_msgs
  pcl_ros
  rosbag
)

# 添加头文件路径
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

# 编译 main
add_executable(main
  src/main.cpp
  include/ikdtree/ikd_tree.cpp   # 确保这里包含了 ikd_tree.cpp
)
target_link_libraries(main
  ${PCL_LIBRARIES}     # 链接 PCL 库
  pthread              # 链接 pthread
  ${catkin_LIBRARIES}  # 链接 ROS 包
)

# 编译简单版本
# add_executable(LidarSearchComparison_simple
#   src/main_simple.cpp
#   include/ikdtree/ikd_tree.cpp
# )
# target_link_libraries(LidarSearchComparison_simple
#   pthread              # 链接 pthread
# )
