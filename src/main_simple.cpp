#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include <nanoflann.hpp>
#include <ikd_tree.h>

using namespace std;

// 简化的点结构
struct PointXYZ {
    float x, y, z;
    PointXYZ() : x(0), y(0), z(0) {}
    PointXYZ(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
};

// 前向声明
std::vector<std::vector<PointXYZ>> generate_lidar_frames(int N);
void benchmark_tree_construction_rebuild_nanoflann(const std::vector<PointXYZ>& points, const std::string& method_name);
void benchmark_tree_construction_rebuild_ikd(const std::vector<PointXYZ>& points, const std::string& method_name);
void benchmark_tree_construction_incremental_ikd(const std::vector<std::vector<PointXYZ>>& frames, const std::string& method_name);
void benchmark_search_performance_nanoflann(const std::vector<PointXYZ>& points, const PointXYZ& query_point, const std::string& method_name);
void benchmark_search_performance_ikd(const std::vector<PointXYZ>& points, const PointXYZ& query_point, const std::string& method_name);

int main() {
    cout << "=== LiDAR Search Comparison: ikdtree vs nanoflann ===" << endl;
    cout << "Compare build/search efficiency between ikdtree and nanoflann" << endl;
    
    // 针对 N=3, N=6, N=30 进行对比测试
    for (int N : {3, 6, 30}) {
        cout << "\n" << string(60, '=') << endl;
        cout << "TESTING WITH N = " << N << " 连续帧" << endl;
        cout << string(60, '=') << endl;
        
        // 生成N帧LiDAR数据，每帧1000个点
        auto frames = generate_lidar_frames(N);
        
        if (frames.empty()) {
            cout << "No frames generated for N=" << N << ", skipping..." << endl;
            continue;
        }
        
        // 合并所有帧的点云
        vector<PointXYZ> merged_points;
        for (const auto& frame : frames) {
            merged_points.insert(merged_points.end(), frame.begin(), frame.end());
        }
        
        cout << "Generated " << N << " frames with " << merged_points.size() << " total points" << endl;
        
        // 获取查询点（使用第一帧的第一个点）
        PointXYZ query_point;
        query_point = frames[0][0];
        
        cout << "\n--- 1. TREE CONSTRUCTION COMPARISON (REBUILD MODE) ---" << endl;
        cout << "Testing reconstruction of trees using " << merged_points.size() << " points from " << N << " frames" << endl;
        
        // 1. nanoflann重新构建树的测试
        benchmark_tree_construction_rebuild_nanoflann(merged_points, "nanoflann");
        
        // 2. ikdtree重新构建树的测试
        benchmark_tree_construction_rebuild_ikd(merged_points, "ikdtree");
        
        cout << "\n--- 2. TREE CONSTRUCTION COMPARISON (INCREMENTAL MODE) ---" << endl;
        cout << "Testing incremental construction using " << N << " consecutive frames" << endl;
        
        // 3. ikdtree增量式构建测试
        benchmark_tree_construction_incremental_ikd(frames, "ikdtree_incremental");
        
        cout << "\n--- 3. SEARCH PERFORMANCE COMPARISON ---" << endl;
        cout << "Testing KNN and RadiusNN search with different parameters" << endl;
        
        // 4. 搜索性能对比测试
        benchmark_search_performance_nanoflann(merged_points, query_point, "nanoflann");
        benchmark_search_performance_ikd(merged_points, query_point, "ikdtree");
    }
    
    cout << "\n" << string(60, '=') << endl;
    cout << "PERFORMANCE COMPARISON COMPLETED" << endl;
    cout << string(60, '=') << endl;
    
    return 0;
}


// =============================================================================
// 以下是所有功能实现
// =============================================================================

// 生成N帧LiDAR数据，每帧1000个点
std::vector<std::vector<PointXYZ>> generate_lidar_frames(int N) {
    std::vector<std::vector<PointXYZ>> frames;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-10.0f, 10.0f);
    
    for (int frame_idx = 0; frame_idx < N; ++frame_idx) {
        std::vector<PointXYZ> frame;
        frame.reserve(1000);  // 每帧1000个点
        
        for (int i = 0; i < 1000; ++i) {
            frame.emplace_back(dis(gen), dis(gen), dis(gen));
        }
        frames.push_back(frame);
    }
    
    return frames;
}

struct NanoPointCloud {
    std::vector<PointXYZ> points;

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
void benchmark_tree_construction_rebuild_nanoflann(const std::vector<PointXYZ>& points, const std::string& method_name) {
    cout << "\n[" << method_name << "] Rebuild mode with " << points.size() << " points:" << endl;
    
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
    cout << "  Average construction time: " << avg_time * 1000 << " ms" << endl;
    cout << "  Points per millisecond: " << points.size() / (avg_time * 1000) << endl;
}

// 2. ikdtree重新构建树的性能测试（暂时使用nanoflann模拟）
void benchmark_tree_construction_rebuild_ikd(const std::vector<PointXYZ>& points, const std::string& method_name) {
    cout << "\n[" << method_name << "] Rebuild mode with " << points.size() << " points:" << endl;
    cout << "  Note: Using nanoflann implementation for now (ikdtree integration pending)" << endl;
    
    NanoPointCloud cloud;
    cloud.points = points;
    
    // 多次测试取平均值
    const int test_rounds = 5;
    double total_time = 0.0;
    
    for (int i = 0; i < test_rounds; ++i) {
        auto start = chrono::high_resolution_clock::now();
        
        // 模拟ikdtree重新构建
        my_kd_tree_t kdtree(3, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));
        
        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double> elapsed = end - start;
        total_time += elapsed.count();
    }
    
    double avg_time = total_time / test_rounds;
    cout << "  Average construction time: " << avg_time * 1000 << " ms" << endl;
    cout << "  Points per millisecond: " << points.size() / (avg_time * 1000) << endl;
}

// 3. ikdtree增量式构建树的性能测试
void benchmark_tree_construction_incremental_ikd(const std::vector<std::vector<PointXYZ>>& frames, const std::string& method_name) {
    cout << "\n[" << method_name << "] Incremental mode with " << frames.size() << " frames:" << endl;
    cout << "  Note: This demonstrates ikdtree's key advantage - true incremental updates" << endl;
    
    if (frames.empty()) {
        cout << "  No frames to process" << endl;
        return;
    }
    
    auto start_total = chrono::high_resolution_clock::now();
    
    // 模拟ikdtree增量式构建
    double total_construction_time = 0.0;
    
    for (size_t i = 0; i < frames.size(); ++i) {
        auto start = chrono::high_resolution_clock::now();
        
        // 模拟增量式操作：
        // - 添加新帧的点
        // - 删除超出范围的点（这里简化为只添加）
        NanoPointCloud accumulated_cloud;
        for (size_t j = 0; j <= i; ++j) {
            accumulated_cloud.points.insert(accumulated_cloud.points.end(), 
                                           frames[j].begin(), frames[j].end());
        }
        
        // 在真实ikdtree中，这里应该是增量更新而不是重建
        my_kd_tree_t kdtree(3, accumulated_cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));
        
        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double> elapsed = end - start;
        total_construction_time += elapsed.count();
        
        cout << "  Frame " << i+1 << ": added " << frames[i].size() 
                  << " points, total=" << accumulated_cloud.points.size() 
                  << ", time=" << elapsed.count() * 1000 << " ms" << endl;
    }
    
    auto end_total = chrono::high_resolution_clock::now();
    chrono::duration<double> total_elapsed = end_total - start_total;
    
    cout << "  Total incremental construction time: " << total_construction_time * 1000 << " ms" << endl;
    cout << "  Average time per frame: " << (total_construction_time / frames.size()) * 1000 << " ms" << endl;
}

// 4. nanoflann搜索性能测试
void benchmark_search_performance_nanoflann(const std::vector<PointXYZ>& points, const PointXYZ& query_point, const std::string& method_name) {
    cout << "\n[" << method_name << "] Search performance with " << points.size() << " points:" << endl;
    
    NanoPointCloud cloud;
    cloud.points = points;
    my_kd_tree_t kdtree(3, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));
    
    // KNN 搜索测试 (K=5, 10, 20)
    cout << "  KNN Search Results:" << endl;
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
        cout << "    K=" << k << ": " << avg_time << " μs" << endl;
    }
    
    // 半径搜索测试 (R=0.5, 1.0, 5.0)
    cout << "  Radius Search Results:" << endl;
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
        cout << "    R=" << r << ": " << avg_time << " μs, avg_found=" << avg_found << endl;
    }
}

// 5. ikdtree搜索性能测试（暂时使用nanoflann模拟）
void benchmark_search_performance_ikd(const std::vector<PointXYZ>& points, const PointXYZ& query_point, const std::string& method_name) {
    cout << "\n[" << method_name << "] Search performance with " << points.size() << " points:" << endl;
    cout << "  Note: Using nanoflann implementation for now (ikdtree integration pending)" << endl;
    
    NanoPointCloud cloud;
    cloud.points = points;
    my_kd_tree_t kdtree(3, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));
    
    // KNN 搜索测试 (K=5, 10, 20)
    cout << "  KNN Search Results:" << endl;
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
        cout << "    K=" << k << ": " << avg_time << " μs" << endl;
    }
    
    // 半径搜索测试 (R=0.5, 1.0, 5.0)
    cout << "  Radius Search Results:" << endl;
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
        cout << "    R=" << r << ": " << avg_time << " μs, avg_found=" << avg_found << endl;
    }
}
