#include <iostream>
#include <chrono>
#include <vector>
#include <pcl/point_types.h>
#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <boost/foreach.hpp>
#include <nanoflann.hpp>
#include <Eigen/Dense>
#include "../libs/ikdtree/ikd_tree.h"

using namespace std;
using namespace pcl;

// 从 lidar_data_loader.cpp 引入的结构和函数
struct PerformanceTestData {
    std::vector<std::vector<pcl::PointXYZ>> consecutive_frames;  
    std::vector<pcl::PointXYZ> merged_points;                   
    int frame_count;
    int total_points;
};

// 前向声明
PerformanceTestData prepare_performance_test_data(const std::string& rosbag_file, int N);
void benchmark_tree_construction_rebuild_nanoflann(const std::vector<pcl::PointXYZ>& points, const std::string& method_name);
void benchmark_tree_construction_rebuild_ikdtree(const std::vector<pcl::PointXYZ>& points, const std::string& method_name);
void benchmark_tree_construction_incremental_ikdtree(const std::vector<std::vector<pcl::PointXYZ>>& frames, const std::string& method_name);
void benchmark_search_performance_nanoflann(const std::vector<pcl::PointXYZ>& points, const pcl::PointXYZ& query_point, const std::string& method_name);
void benchmark_search_performance_ikdtree(const std::vector<pcl::PointXYZ>& points, const pcl::PointXYZ& query_point, const std::string& method_name);

int main() {
    std::cout << "=== LiDAR Search Comparison: ikdtree vs nanoflann ===" << std::endl;
    std::cout << "Core Objective: Compare build/search efficiency between ikdtree and nanoflann" << std::endl;
    
    // 针对 N=3, N=6, N=30 进行对比测试
    for (int N : {3, 6, 30}) {
        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "TESTING WITH N = " << N << " CONSECUTIVE FRAMES" << std::endl;
        std::cout << std::string(60, '=') << std::endl;
        
        // 准备测试数据：连续的N帧LiDAR scan
        auto test_data = prepare_performance_test_data("../data/9.bag", N);
        
        if (test_data.frame_count == 0) {
            std::cout << "No data loaded for N=" << N << ", skipping..." << std::endl;
            continue;
        }
        
        // 获取查询点（使用第一帧的第一个点）
        pcl::PointXYZ query_point;
        if (!test_data.consecutive_frames.empty() && !test_data.consecutive_frames[0].empty()) {
            query_point = test_data.consecutive_frames[0][0];
        } else {
            std::cout << "No valid query point, skipping search tests..." << std::endl;
            continue;
        }
        
        std::cout << "\n--- 1. TREE CONSTRUCTION COMPARISON (REBUILD MODE) ---" << std::endl;
        std::cout << "Testing reconstruction of trees using " << test_data.total_points << " points from " << N << " frames" << std::endl;
        
        // 1. nanoflann重新构建树的测试
        benchmark_tree_construction_rebuild_nanoflann(test_data.merged_points, "nanoflann");
        
        // 2. ikdtree重新构建树的测试 (暂时跳过，有崩溃问题)
        std::cout << "\n[ikdtree] Rebuild mode: SKIPPED (debugging segmentation fault)" << std::endl;
        
        std::cout << "\n--- 2. TREE CONSTRUCTION COMPARISON (INCREMENTAL MODE) ---" << std::endl;
        std::cout << "Testing incremental construction using " << N << " consecutive frames" << std::endl;
        std::cout << "Note: nanoflann does not support true incremental updates - it requires full rebuilds" << std::endl;
        std::cout << "Only ikdtree supports efficient incremental updates, but ikdtree is currently disabled due to segfault" << std::endl;
       
        std::cout << "\n--- 3. SEARCH PERFORMANCE COMPARISON ---" << std::endl;
        std::cout << "Testing KNN and RadiusNN search with different parameters" << std::endl;
        
        // 4. 搜索性能对比测试
        benchmark_search_performance_nanoflann(test_data.merged_points, query_point, "nanoflann");
        std::cout << "\n[ikdtree] Search performance: SKIPPED (debugging segmentation fault)" << std::endl;
    }
    
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "PERFORMANCE COMPARISON COMPLETED" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    return 0;
}

// =============================================================================
// 以下是所有功能实现，针对您的核心目标优化
// =============================================================================

// 从 rosbag 按帧加载数据
std::vector<std::vector<pcl::PointXYZ>> load_lidar_frames_from_rosbag(const std::string& rosbag_file, int max_frames = -1) {
    std::vector<std::vector<pcl::PointXYZ>> frames;
    
    try {
        rosbag::Bag bag;
        bag.open(rosbag_file, rosbag::bagmode::Read);
        
        rosbag::View view(bag);
        
        if (view.size() == 0) {
            std::cerr << "Warning: Bag file is empty: " << rosbag_file << std::endl;
            bag.close();
            return frames;
        }
        
        int frame_count = 0;
        
        BOOST_FOREACH(rosbag::MessageInstance const m, view) {
            if (m.getTopic() == "/cloud_registered") {
                sensor_msgs::PointCloud2::ConstPtr msg = m.instantiate<sensor_msgs::PointCloud2>();
                if (msg != nullptr) {
                    try {
                        PointCloud<PointXYZ> cloud;
                        pcl::fromROSMsg(*msg, cloud);
                        
                        if (!cloud.empty()) {
                            std::vector<pcl::PointXYZ> frame_points(cloud.points.begin(), cloud.points.end());
                            frames.push_back(frame_points);
                            frame_count++;
                            
                            if (max_frames > 0 && frame_count >= max_frames) {
                                break;
                            }
                        }
                    } catch (const std::exception& e) {
                        std::cerr << "Error converting PointCloud2 message: " << e.what() << std::endl;
                        continue;
                    }
                }
            }
        }
        
        bag.close();
        std::cout << "Loaded " << frames.size() << " consecutive frames" << std::endl;
        
    } catch (const rosbag::BagException& e) {
        std::cerr << "Error opening bag file " << rosbag_file << ": " << e.what() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Unexpected error: " << e.what() << std::endl;
    }
    
    return frames;
}

// 准备性能测试数据
PerformanceTestData prepare_performance_test_data(const std::string& rosbag_file, int N) {
    PerformanceTestData test_data;
    
    auto frames = load_lidar_frames_from_rosbag(rosbag_file, N);
    
    test_data.consecutive_frames = frames;
    test_data.frame_count = frames.size();
    test_data.total_points = 0;
    
    // 合并所有帧的点云
    for (const auto& frame : frames) {
        test_data.merged_points.insert(test_data.merged_points.end(), frame.begin(), frame.end());
        test_data.total_points += frame.size();
    }
    
    std::cout << "Prepared " << test_data.frame_count << " frames with " << test_data.total_points << " total points" << std::endl;
    
    return test_data;
}

// nanoflann 适配器
struct NanoPointCloud {
    std::vector<pcl::PointXYZ> points;
    inline size_t kdtree_get_point_count() const { return points.size(); }
    inline double kdtree_get_pt(const size_t idx, const size_t dim) const {
        if (dim == 0) return points[idx].x;
        else if (dim == 1) return points[idx].y;
        else return points[idx].z;
    }
    template <class BBOX>
    bool kdtree_get_bbox(BBOX &bb) const { (void)bb; return false; }
};

typedef nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<float, NanoPointCloud>, NanoPointCloud, 3> my_kd_tree_t;

// 1. nanoflann重新构建树的性能测试
void benchmark_tree_construction_rebuild_nanoflann(const std::vector<pcl::PointXYZ>& points, const std::string& method_name) {
    std::cout << "\n[" << method_name << "] Rebuild mode with " << points.size() << " points:" << std::endl;
    
    NanoPointCloud cloud;
    cloud.points = points;
    
    // 多次测试取平均值
    const int test_rounds = 5;
    double total_time = 0.0;
    
    for (int i = 0; i < test_rounds; ++i) {
        auto start = chrono::high_resolution_clock::now();
        
        // 重新构建树
        my_kd_tree_t kdtree(3, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));
        
        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double> elapsed = end - start;
        total_time += elapsed.count();
    }
    
    double avg_time = total_time / test_rounds;
    std::cout << "  Average construction time: " << avg_time * 1000 << " ms" << std::endl;
    std::cout << "  Points per millisecond: " << points.size() / (avg_time * 1000) << std::endl;
}


// 2. ikdtree重新构建树的性能测试
void benchmark_tree_construction_rebuild_ikdtree(const std::vector<pcl::PointXYZ>& points, const std::string& method_name) {
    std::cout << "\n[" << method_name << "] Rebuild mode with " << points.size() << " points:" << std::endl;
    
    if (points.empty()) {
        std::cout << "  No points to process" << std::endl;
        return;
    }
    
    // 多次测试取平均值
    const int test_rounds = 5;
    double total_time = 0.0;
    
    for (int i = 0; i < test_rounds; ++i) {
        std::cout << "  Test round " << (i+1) << "/" << test_rounds << std::endl;
        
        try {
            auto start = chrono::high_resolution_clock::now();
            
            // 创建ikdtree并初始化
            KD_TREE<pcl::PointXYZ> ikd_tree(0.5, 0.6, 0.2);
            
            // 转换为ikdtree所需的vector类型
            vector<pcl::PointXYZ, Eigen::aligned_allocator<pcl::PointXYZ>> point_vector(points.begin(), points.end());
            
            std::cout << "    Building tree with " << point_vector.size() << " points" << std::endl;
            ikd_tree.Build(point_vector);
            std::cout << "    Tree built successfully, size: " << ikd_tree.size() << std::endl;
            
            auto end = chrono::high_resolution_clock::now();
            chrono::duration<double> elapsed = end - start;
            total_time += elapsed.count();
            
        } catch (const std::exception& e) {
            std::cerr << "    Error in round " << (i+1) << ": " << e.what() << std::endl;
            return;
        } catch (...) {
            std::cerr << "    Unknown error in round " << (i+1) << std::endl;
            return;
        }
    }
    
    double avg_time = total_time / test_rounds;
    std::cout << "  Average construction time: " << avg_time * 1000 << " ms" << std::endl;
    std::cout << "  Points per millisecond: " << points.size() / (avg_time * 1000) << std::endl;
}

// 3. ikdtree增量式构建树的性能测试
void benchmark_tree_construction_incremental_ikdtree(const std::vector<std::vector<pcl::PointXYZ>>& frames, const std::string& method_name) {
    std::cout << "\n[" << method_name << "] Incremental mode with " << frames.size() << " frames:" << std::endl;
    
    if (frames.empty()) {
        std::cout << "  No frames to process" << std::endl;
        return;
    }
    
    try {
        auto start_total = chrono::high_resolution_clock::now();
        
        // 使用ikdtree进行真正的增量式构建
        KD_TREE<pcl::PointXYZ> ikd_tree(0.5, 0.6, 0.2);
        double total_construction_time = 0.0;
        
        // 首先用第一帧初始化树
        if (!frames[0].empty()) {
            auto start = chrono::high_resolution_clock::now();
            vector<pcl::PointXYZ, Eigen::aligned_allocator<pcl::PointXYZ>> initial_points(frames[0].begin(), frames[0].end());
            ikd_tree.Build(initial_points);
            auto end = chrono::high_resolution_clock::now();
            chrono::duration<double> elapsed = end - start;
            total_construction_time += elapsed.count();
            
            std::cout << "  Frame 1: built initial tree with " << frames[0].size() 
                      << " points, time=" << elapsed.count() * 1000 << " ms" << std::endl;
        }
    
        // 增量式添加后续帧
        for (size_t i = 1; i < frames.size(); ++i) {
            auto start = chrono::high_resolution_clock::now();
            
            // 使用ikdtree的Add_Points功能增量添加
            vector<pcl::PointXYZ, Eigen::aligned_allocator<pcl::PointXYZ>> frame_points(frames[i].begin(), frames[i].end());
            ikd_tree.Add_Points(frame_points, false); // 不进行下采样
            
            auto end = chrono::high_resolution_clock::now();
            chrono::duration<double> elapsed = end - start;
            total_construction_time += elapsed.count();
            
            std::cout << "  Frame " << i+1 << ": added " << frames[i].size() 
                      << " points, total_tree_size=" << ikd_tree.size()
                      << ", time=" << elapsed.count() * 1000 << " ms" << std::endl;
        }
        
        auto end_total = chrono::high_resolution_clock::now();
        chrono::duration<double> total_elapsed = end_total - start_total;
        
        std::cout << "  Total incremental construction time: " << total_construction_time * 1000 << " ms" << std::endl;
        std::cout << "  Average time per frame: " << (total_construction_time / frames.size()) * 1000 << " ms" << std::endl;
        std::cout << "  Final tree size: " << ikd_tree.size() << " points" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "  Error in incremental construction: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "  Unknown error in incremental construction" << std::endl;
    }
}

// 4. nanoflann搜索性能测试
void benchmark_search_performance_nanoflann(const std::vector<pcl::PointXYZ>& points, const pcl::PointXYZ& query_point, const std::string& method_name) {
    std::cout << "\n[" << method_name << "] Search performance with " << points.size() << " points:" << std::endl;
    
    NanoPointCloud cloud;
    cloud.points = points;
    my_kd_tree_t kdtree(3, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));
    
    // KNN 搜索测试 (K=5, 10, 20)
    std::cout << "  KNN Search Results:" << std::endl;
    for (size_t k : {5, 10, 20}) {
        if (k > points.size()) continue;
        
        std::vector<uint32_t> ret_index(k);
        std::vector<float> out_distances(k);
        
        const int search_rounds = 100;  // 多次搜索取平均
        double total_time = 0.0;
        
        for (int i = 0; i < search_rounds; ++i) {
            auto start = chrono::high_resolution_clock::now();
            kdtree.knnSearch(&query_point.x, k, &ret_index[0], &out_distances[0]);
            auto end = chrono::high_resolution_clock::now();
            total_time += chrono::duration<double>(end - start).count();
        }
        
        double avg_time = (total_time / search_rounds) * 1000000;  // 转换为微秒
        std::cout << "    K=" << k << ": " << avg_time << " μs" << std::endl;
    }
    
    // 半径搜索测试 (R=0.5, 1.0, 5.0)
    std::cout << "  Radius Search Results:" << std::endl;
    for (float r : {0.5f, 1.0f, 5.0f}) {
        std::vector<nanoflann::ResultItem<uint32_t, float>> ret_indices_dists;
        
        const int search_rounds = 100;
        double total_time = 0.0;
        int total_found = 0;
        
        for (int i = 0; i < search_rounds; ++i) {
            ret_indices_dists.clear();
            auto start = chrono::high_resolution_clock::now();
            kdtree.radiusSearch(&query_point.x, r * r, ret_indices_dists);
            auto end = chrono::high_resolution_clock::now();
            total_time += chrono::duration<double>(end - start).count();
            total_found += ret_indices_dists.size();
        }
        
        double avg_time = (total_time / search_rounds) * 1000000;  // 转换为微秒
        int avg_found = total_found / search_rounds;
        std::cout << "    R=" << r << ": " << avg_time << " μs, avg_found=" << avg_found << std::endl;
    }
}

// 5. ikdtree搜索性能测试
void benchmark_search_performance_ikdtree(const std::vector<pcl::PointXYZ>& points, const pcl::PointXYZ& query_point, const std::string& method_name) {
    std::cout << "\n[" << method_name << "] Search performance with " << points.size() << " points:" << std::endl;
    
    try {
        // 构建ikdtree
        KD_TREE<pcl::PointXYZ> ikd_tree(0.5, 0.6, 0.2);
        vector<pcl::PointXYZ, Eigen::aligned_allocator<pcl::PointXYZ>> point_vector(points.begin(), points.end());
        ikd_tree.Build(point_vector);
        
        std::cout << "  Tree built successfully for search testing, size: " << ikd_tree.size() << std::endl;
    
    // KNN 搜索测试 (K=5, 10, 20)
    std::cout << "  KNN Search Results:" << std::endl;
    for (size_t k : {5, 10, 20}) {
        if (k > points.size()) continue;
        
        const int search_rounds = 100;  // 多次搜索取平均
        double total_time = 0.0;
        
        for (int i = 0; i < search_rounds; ++i) {
            vector<pcl::PointXYZ, Eigen::aligned_allocator<pcl::PointXYZ>> nearest_points;
            vector<float> point_distances;
            
            auto start = chrono::high_resolution_clock::now();
            ikd_tree.Nearest_Search(query_point, k, nearest_points, point_distances);
            auto end = chrono::high_resolution_clock::now();
            total_time += chrono::duration<double>(end - start).count();
        }
        
        double avg_time = (total_time / search_rounds) * 1000000;  // 转换为微秒
        std::cout << "    K=" << k << ": " << avg_time << " μs" << std::endl;
    }
    
    // 半径搜索测试 (R=0.5, 1.0, 5.0)
    std::cout << "  Radius Search Results:" << std::endl;
    for (float r : {0.5f, 1.0f, 5.0f}) {
        const int search_rounds = 100;
        double total_time = 0.0;
        int total_found = 0;
        
        for (int i = 0; i < search_rounds; ++i) {
            vector<pcl::PointXYZ, Eigen::aligned_allocator<pcl::PointXYZ>> radius_points;
            
            auto start = chrono::high_resolution_clock::now();
            ikd_tree.Radius_Search(query_point, r, radius_points);
            auto end = chrono::high_resolution_clock::now();
            total_time += chrono::duration<double>(end - start).count();
            total_found += radius_points.size();
        }
        
        double avg_time = (total_time / search_rounds) * 1000000;  // 转换为微秒
        int avg_found = total_found / search_rounds;
        std::cout << "    R=" << r << ": " << avg_time << " μs, avg_found=" << avg_found << std::endl;
    }
    
    } catch (const std::exception& e) {
        std::cerr << "  Error in search performance test: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "  Unknown error in search performance test" << std::endl;
    }
}
