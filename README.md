# LiDAR Search Comparison Project

## 项目简介

这是一个用于比较不同 K-d 树实现在 LiDAR 点云数据搜索任务中性能表现的项目。主要对比：
- **nanoflann**: 高性能的 C++ K-d 树库（重新构建模式）
- **ikd-tree**: 支持增量构建的 K-d 树库（增量模式）

## 项目结构

```
LidarSearchComparison/
├── CMakeLists.txt          # CMake 构建配置
├── README.md              # 项目说明文档
├── run.sh                 # 构建和运行脚本
├── src/                   # 源代码目录
│   ├── main.cpp          # 完整版本主程序（依赖 ROS/PCL）
│   ├── main_simple.cpp   # 简化版本主程序（无外部依赖）
│   ├── lidar_data_loader.cpp  # LiDAR 数据加载（原始版本）
│   ├── tree_construction.cpp  # 树构建（原始版本）
│   └── search.cpp        # 搜索功能（原始版本）
├── libs/                 # 第三方库
│   ├── nanoflann/        # nanoflann 库
│   └── ikd-tree/         # ikd-tree 库
├── data/                 # 数据文件
│   └── 9.bag             # ROS bag 数据文件
└── build/                # 构建目录
```

## 功能特性

### 已实现功能
1. **树构建性能测试**
   - nanoflann 重新构建模式
   - ikd-tree 增量构建模式（当前使用 nanoflann 模拟）

2. **搜索性能测试**
   - KNN 搜索（K=5, 10, 20）
   - 半径搜索（R=0.5, 1.0, 2.0）

3. **数据规模测试**
   - 100 点、1000 点、5000 点三种规模

4. **双版本支持**
   - 简化版本：无外部依赖，使用随机生成的测试数据
   - 完整版本：支持 ROS bag 文件加载真实 LiDAR 数据

### 性能指标
- 树构建时间（毫秒）
- KNN 搜索时间（毫秒）
- 半径搜索时间（毫秒）
- 搜索结果数量

## 编译和运行

### 方法一：使用运行脚本（推荐）
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

# 运行简化版本
./LidarSearchComparison_simple

# 运行完整版本（如果有 ROS/PCL 环境）
./LidarSearchComparison
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

## 示例输出

```
=== LiDAR Search Comparison Project ===
Generated 10000 test points.

=== Running for N = 100 points ===

--- Using nanoflann (rebuild) ---
nanoflann tree construction time: 0.00815 ms

--- Using ikd-tree (incremental) ---
ikd-tree construction time: 0.00581 ms (using nanoflann for now)

--- KNN Search Tests ---
KNN search time for K=5: 0.00089 ms
KNN search time for K=10: 0.00043 ms
KNN search time for K=20: 0.0011 ms

--- Radius Search Tests ---
RadiusNN search time for R=0.5: 0.00055 ms, found 1 points
RadiusNN search time for R=1: 0.00023 ms, found 2 points
RadiusNN search time for R=2: 0.00011 ms, found 2 points
```

## 问题修复记录

### 原始问题
1. ❌ 编译错误：缺少头文件和依赖
2. ❌ 代码结构问题：.cpp 文件被当作头文件包含
3. ❌ 命名冲突：与 PCL 的 PointCloud 类型冲突
4. ❌ 变量未定义：main 函数中的 kdtree 变量未定义
5. ❌ 功能缺失：没有真正实现两种树的对比

### 修复方案
1. ✅ 创建简化版本，去除 ROS/PCL 依赖
2. ✅ 重构代码结构，将所有功能集成到单个文件
3. ✅ 解决命名冲突：使用 NanoPointCloud 避免与 PCL 冲突
4. ✅ 修复数据类型不匹配问题
5. ✅ 添加性能计时和结果输出
6. ✅ 支持多种数据规模测试

## 贡献说明

欢迎提交 Issue 和 Pull Request 来改进这个项目！

## 许可证

本项目遵循相关开源库的许可证要求。
