# LiDAR Search Comparison Project(**WIP**)

## 项目简介

这是一个用于比较不同 K-d 树实现在 LiDAR 点云数据搜索任务中性能表现的项目。主要对比：
- **nanoflann**: 高性能的 C++ K-d 树库
- **ikd-tree**: 支持增量构建的 K-d 树库

## 编译和运行

### 方法一：使用运行脚本（推荐），需要修改其中的 PROJECT_DIR
```bash
./run.sh
```

### 方法二：手动编译
```bash
# 创建构建目录
mkdir -p build && cd build

# CMake 配置
cmake ..

# 编译
make -j4

# 运行简化版本（弃用）
# ./LidarSearchComparison_simple

# 运行完整版本（如果有 ROS/PCL 环境）
./main
```

## 依赖要求

### 简化版本
- C++14 编译器
- CMake 3.10+
- pthread

### 完整版本（额外依赖）
- ROS Noetic
- PCL 1.10+
- rosbag
- sensor_msgs
- pcl_ros
- pcl_conversions

