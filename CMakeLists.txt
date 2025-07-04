cmake_minimum_required(VERSION 3.10)  # 设置CMake的最小版本要求

project(LidarSearchComparison)  # 定义项目名称

set(CMAKE_CXX_STANDARD 14)  # 设置C++标准为 C++14

# 设置头文件目录
include_directories(include)

# 设置源文件目录和目标程序
add_executable(LidarSearchComparison src/main.cpp)

