cmake_minimum_required(VERSION 3.10)

project(LidarSearchComparison)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 14)

# 包含头文件目录
include_directories(
    ${PROJECT_SOURCE_DIR}/libs/ikd-tree     # 添加 ikd-tree 头文件目录
    ${PROJECT_SOURCE_DIR}/libs/nanoflann    # 添加 nanoflann 头文件目录
    ${PROJECT_SOURCE_DIR}/include           # 添加项目的 include 目录（如果有的话）
)

# 添加源文件
add_executable(LidarSearchComparison 
    src/main.cpp

    
)

# 如果 ikd-tree 有静态库或动态库文件（.a/.so），您可以使用以下命令链接库
# target_link_libraries(LidarSearchComparison ikd-tree)
