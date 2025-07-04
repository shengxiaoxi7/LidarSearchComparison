#!/bin/bash

# LiDAR Search Comparison 项目构建和运行脚本

PROJECT_DIR="/home/syx/wfzf/code/ikdtree_nanoflann/src/LidarSearchComparison"
BUILD_DIR="$PROJECT_DIR/build"

echo "=== LiDAR Search Comparison Build & Run Script ==="

# 检查项目目录
if [ ! -d "$PROJECT_DIR" ]; then
    echo "错误: 项目目录不存在: $PROJECT_DIR"
    exit 1
fi

cd "$PROJECT_DIR"

# 清理并创建构建目录
echo "清理旧的构建文件..."
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# 运行 CMake 配置
echo "运行 CMake 配置..."
if ! cmake ..; then
    echo "错误: CMake 配置失败"
    exit 1
fi

# 编译项目
echo "编译项目..."
if ! make -j4; then
    echo "错误: 编译失败"
    exit 1
fi

echo "编译成功!"

# 检查可执行文件
if [ -f "./LidarSearchComparison_simple" ]; then
    echo ""
    echo "=== 运行简化版本 (不依赖ROS/PCL) ==="
    ./LidarSearchComparison_simple
fi

if [ -f "./LidarSearchComparison" ]; then
    echo ""
    echo "=== 运行完整版本 (需要ROS/PCL) ==="
    ./LidarSearchComparison
fi

echo ""
echo "=== 构建和运行完成 ==="
