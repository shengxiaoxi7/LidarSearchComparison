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

struct PerformanceTestData {
    std::vector<std::vector<pcl::PointXYZ>> consecutive_frames;  
    std::vector<pcl::PointXYZ> merged_points;                   
    int frame_count;
    int total_points;
};

PerformanceTestData prepare_performance_test_data(const std::string& rosbag_file, int N);
void benchmark_tree_construction_rebuild_nanoflann(const std::vector<pcl::PointXYZ>& points, const std::string& method_name);
void benchmark_tree_construction_rebuild_ikdtree(const std::vector<pcl::PointXYZ>& points, const std::string& method_name);
void benchmark_tree_construction_incremental_ikdtree(const std::vector<std::vector<pcl::PointXYZ>>& frames, const std::string& method_name);
void benchmark_search_performance_nanoflann(const std::vector<pcl::PointXYZ>& points, const pcl::PointXYZ& query_point, const std::string& method_name);
void benchmark_search_performance_ikdtree(const std::vector<pcl::PointXYZ>& points, const pcl::PointXYZ& query_point, const std::string& method_name);

int main() {
    std::cout << "=== LiDAR Search Comparison: ikdtree vs nanoflann ===" << std::endl;
    std::cout << "Core Objective: Compare build/search efficiency between ikdtree and nanoflann" << std::endl;

    for (int N : {3, 6, 30}) {
        
        std::cout << "TESTING WITH N = " << N << " CONSECUTIVE FRAMES" << std::endl;

        auto test_data = prepare_performance_test_data("../data/9.bag", N);
        
        if (test_data.frame_count == 0) {
            std::cout << "No data loaded for N=" << N << ", skipping..." << std::endl;
            continue;
        }

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
        // benchmark_tree_construction_rebuild_ikdtree(test_data.merged_points, "ikdtree");
        
        std::cout << "\n--- 2. TREE CONSTRUCTION COMPARISON (INCREMENTAL MODE) ---" << std::endl;
        std::cout << "Testing incremental construction using " << N << " consecutive frames" << std::endl;
        // benchmark_tree_construction_incremental_ikdtree(test_data.consecutive_frames, "ikdtree_incremental");

        std::cout << "\n--- 3. SEARCH PERFORMANCE COMPARISON ---" << std::endl;
        std::cout << "Testing KNN and RadiusNN search with different parameters" << std::endl;
        
        // 3. 搜索性能对比测试
        benchmark_search_performance_nanoflann(test_data.merged_points, query_point, "nanoflann");
        // benchmark_search_performance_ikdtree(test_data.merged_points, query_point, "ikdtree");
    }
    
    std::cout << "PERFORMANCE COMPARISON COMPLETED" << std::endl;
    
    return 0;
}

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

PerformanceTestData prepare_performance_test_data(const std::string& rosbag_file, int N) {
    PerformanceTestData test_data;
    
    auto frames = load_lidar_frames_from_rosbag(rosbag_file, N);
    
    test_data.consecutive_frames = frames;
    test_data.frame_count = frames.size();
    test_data.total_points = 0;

    for (const auto& frame : frames) {
        test_data.merged_points.insert(test_data.merged_points.end(), frame.begin(), frame.end());
        test_data.total_points += frame.size();
    }
    
    std::cout << "Prepared " << test_data.frame_count << " frames with " << test_data.total_points << " total points" << std::endl;
    
    return test_data;
}

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

void benchmark_tree_construction_rebuild_nanoflann(const std::vector<pcl::PointXYZ>& points, const std::string& method_name) {
    std::cout << "\n[" << method_name << "] Rebuild mode with " << points.size() << " points:" << std::endl;
    
    NanoPointCloud cloud;
    cloud.points = points;

    const int test_rounds = 5;
    double total_time = 0.0;
    
    for (int i = 0; i < test_rounds; ++i) {
        auto start = chrono::high_resolution_clock::now();

        my_kd_tree_t kdtree(3, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));
        
        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double> elapsed = end - start;
        total_time += elapsed.count();
    }
    
    double avg_time = total_time / test_rounds;
    std::cout << "  Average construction time: " << avg_time * 1000 << " ms" << std::endl;
    std::cout << "  Points per millisecond: " << points.size() / (avg_time * 1000) << std::endl;
}


void benchmark_tree_construction_rebuild_ikdtree(const std::vector<pcl::PointXYZ>& points, const std::string& method_name) {
    std::cout << "\n[" << method_name << "] Rebuild mode with " << points.size() << " points:" << std::endl;
    
    if (points.empty()) {
        std::cout << "  No points to process" << std::endl;
        return;
    }

    const int test_rounds = 5;
    double total_time = 0.0;
    
    for (int i = 0; i < test_rounds; ++i) {
        std::cout << "  Test round " << (i+1) << "/" << test_rounds << std::endl;
        
        try {
            auto start = chrono::high_resolution_clock::now();

            KD_TREE<pcl::PointXYZ> ikd_tree(0.5, 0.6, 0.2);

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

void benchmark_tree_construction_incremental_ikdtree(const std::vector<std::vector<pcl::PointXYZ>>& frames, const std::string& method_name) {
    std::cout << "\n[" << method_name << "] Incremental mode with " << frames.size() << " frames:" << std::endl;
    
    if (frames.empty()) {
        std::cout << "  No frames to process" << std::endl;
        return;
    }
    
    try {
        auto start_total = chrono::high_resolution_clock::now();

        KD_TREE<pcl::PointXYZ> ikd_tree(0.5, 0.6, 0.2);
        double total_construction_time = 0.0;

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

        for (size_t i = 1; i < frames.size(); ++i) {
            auto start = chrono::high_resolution_clock::now();

            vector<pcl::PointXYZ, Eigen::aligned_allocator<pcl::PointXYZ>> frame_points(frames[i].begin(), frames[i].end());
            ikd_tree.Add_Points(frame_points, false);
            
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

void benchmark_search_performance_nanoflann(const std::vector<pcl::PointXYZ>& points, const pcl::PointXYZ& query_point, const std::string& method_name) {
    std::cout << "\n[" << method_name << "] Search performance with " << points.size() << " points:" << std::endl;
    
    NanoPointCloud cloud;
    cloud.points = points;
    my_kd_tree_t kdtree(3, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));

    std::cout << "  KNN Search Results:" << std::endl;
    for (size_t k : {5, 10, 20}) {
        if (k > points.size()) continue;
        
        std::vector<uint32_t> ret_index(k);
        std::vector<float> out_distances(k);
        
        const int search_rounds = 100;
        double total_time = 0.0;
        
        for (int i = 0; i < search_rounds; ++i) {
            auto start = chrono::high_resolution_clock::now();
            kdtree.knnSearch(&query_point.x, k, &ret_index[0], &out_distances[0]);
            auto end = chrono::high_resolution_clock::now();
            total_time += chrono::duration<double>(end - start).count();
        }
        
        double avg_time = (total_time / search_rounds) * 1000; 
        std::cout << "    K=" << k << ": " << avg_time << " ms" << std::endl;
    }

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
        
        double avg_time = (total_time / search_rounds) * 1000;
        int avg_found = total_found / search_rounds;
        std::cout << "    R=" << r << ": " << avg_time << " ms, avg_found=" << avg_found << std::endl;
    }
}

void benchmark_search_performance_ikdtree(const std::vector<pcl::PointXYZ>& points, const pcl::PointXYZ& query_point, const std::string& method_name) {
    std::cout << "\n[" << method_name << "] Search performance with " << points.size() << " points:" << std::endl;
    
    try {

        KD_TREE<pcl::PointXYZ> ikd_tree(0.5, 0.6, 0.2);
        vector<pcl::PointXYZ, Eigen::aligned_allocator<pcl::PointXYZ>> point_vector(points.begin(), points.end());
        ikd_tree.Build(point_vector);
        
        std::cout << "  Tree built successfully for search testing, size: " << ikd_tree.size() << std::endl;

    std::cout << "  KNN Search Results:" << std::endl;
    for (size_t k : {5, 10, 20}) {
        if (k > points.size()) continue;
        
        const int search_rounds = 100;
        double total_time = 0.0;
        
        for (int i = 0; i < search_rounds; ++i) {
            vector<pcl::PointXYZ, Eigen::aligned_allocator<pcl::PointXYZ>> nearest_points;
            vector<float> point_distances;
            
            auto start = chrono::high_resolution_clock::now();
            ikd_tree.Nearest_Search(query_point, k, nearest_points, point_distances);
            auto end = chrono::high_resolution_clock::now();
            total_time += chrono::duration<double>(end - start).count();
        }
        
        double avg_time = (total_time / search_rounds) * 1000;
        std::cout << "    K=" << k << ": " << avg_time << " ms" << std::endl;
    }

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
        
        double avg_time = (total_time / search_rounds) * 1000;
        int avg_found = total_found / search_rounds;
        std::cout << "    R=" << r << ": " << avg_time << " ms, avg_found=" << avg_found << std::endl;
    }
    
    } catch (const std::exception& e) {
        std::cerr << "  Error in search performance test: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "  Unknown error in search performance test" << std::endl;
    }
}
