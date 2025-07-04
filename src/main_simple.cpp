#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include <nanoflann.hpp>

using namespace std;

// 简化的点结构
struct PointXYZ {
    float x, y, z;
    PointXYZ() : x(0), y(0), z(0) {}
    PointXYZ(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
};

// 前向声明
std::vector<PointXYZ> generate_test_data(int num_points);
void construct_tree_and_benchmark_nanoflann(const std::vector<PointXYZ>& points);
void construct_tree_and_benchmark_ikd(const std::vector<PointXYZ>& points);
void knn_search_nanoflann(const std::vector<PointXYZ>& points, const PointXYZ& query_point, int k);
void radius_nn_search_nanoflann(const std::vector<PointXYZ>& points, const PointXYZ& query_point, float radius);

int main() {
    cout << "=== LiDAR Search Comparison Project ===" << endl;
    
    // 生成测试数据
    vector<PointXYZ> points = generate_test_data(10000);
    cout << "Generated " << points.size() << " test points." << endl;
    
    // 对 N=100, 1000, 5000 进行对比
    for (int N : {100, 1000, 5000}) {
        cout << "\n=== Running for N = " << N << " points ===" << endl;
        
        // 确保有足够的数据点
        if (points.size() < N) {
            cout << "Warning: Not enough points in dataset. Available: " << points.size() << ", Required: " << N << endl;
            continue;
        }
        
        vector<PointXYZ> subset(points.begin(), points.begin() + N);
        
        // 重新构建树（使用 nanoflann）
        cout << "\n--- Using nanoflann (rebuild) ---" << endl;
        construct_tree_and_benchmark_nanoflann(subset);
        
        // 使用增量式构建树（ikd-tree模拟）
        cout << "\n--- Using ikd-tree (incremental) ---" << endl;
        construct_tree_and_benchmark_ikd(subset);
        
        if (!subset.empty()) {
            // 执行 KNN 搜索和 RadiusNN 搜索
            PointXYZ query_point = subset[0];  // 使用第一个数据点作为查询点
            
            cout << "\n--- KNN Search Tests ---" << endl;
            for (int k : {5, 10, 20}) {
                if (k <= subset.size()) {
                    knn_search_nanoflann(subset, query_point, k);
                }
            }
            
            cout << "\n--- Radius Search Tests ---" << endl;
            for (float r : {0.5, 1.0, 2.0}) {
                radius_nn_search_nanoflann(subset, query_point, r);
            }
        }
    }
    
    return 0;
}

// 生成测试数据
std::vector<PointXYZ> generate_test_data(int num_points) {
    std::vector<PointXYZ> points;
    points.reserve(num_points);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-10.0f, 10.0f);
    
    for (int i = 0; i < num_points; ++i) {
        points.emplace_back(dis(gen), dis(gen), dis(gen));
    }
    
    return points;
}

// nanoflann 相关结构和函数
struct PointCloud {
    std::vector<PointXYZ> points;

    inline size_t kdtree_get_point_count() const { return points.size(); }
    inline double kdtree_get_pt(const size_t idx, const size_t dim) const {
        if (dim == 0) return points[idx].x;
        else if (dim == 1) return points[idx].y;
        else return points[idx].z;
    }
    template <class BBOX>
    bool kdtree_get_bbox(BBOX &bb) const { return false; }
};

typedef nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<float, PointCloud>, PointCloud, 3> my_kd_tree_t;

void construct_tree_and_benchmark_nanoflann(const std::vector<PointXYZ>& points) {
    PointCloud cloud;
    cloud.points = points;

    auto start = chrono::high_resolution_clock::now();
    
    // 构建 nanoflann K-d 树
    my_kd_tree_t kdtree(3, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));
    
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;
    cout << "nanoflann tree construction time: " << elapsed.count() * 1000 << " ms" << endl;
}

void construct_tree_and_benchmark_ikd(const std::vector<PointXYZ>& points) {
    // 暂时使用相同的 nanoflann 实现，后续可以替换为真正的 ikd-tree
    PointCloud cloud;
    cloud.points = points;

    auto start = chrono::high_resolution_clock::now();
    
    // 模拟增量式构建
    my_kd_tree_t kdtree(3, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));
    
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;
    cout << "ikd-tree construction time: " << elapsed.count() * 1000 << " ms (using nanoflann for now)" << endl;
}

void knn_search_nanoflann(const std::vector<PointXYZ>& points, const PointXYZ& query_point, int k) {
    PointCloud cloud;
    cloud.points = points;
    my_kd_tree_t kdtree(3, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));
    
    std::vector<uint32_t> ret_index(k);
    std::vector<float> out_distances(k);
    
    auto start = chrono::high_resolution_clock::now();
    
    kdtree.knnSearch(&query_point.x, k, &ret_index[0], &out_distances[0]);
    
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;
    cout << "KNN search time for K=" << k << ": " << elapsed.count() * 1000 << " ms" << endl;
}

void radius_nn_search_nanoflann(const std::vector<PointXYZ>& points, const PointXYZ& query_point, float radius) {
    PointCloud cloud;
    cloud.points = points;
    my_kd_tree_t kdtree(3, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));
    
    std::vector<nanoflann::ResultItem<uint32_t, float>> ret_indices_dists;
    
    auto start = chrono::high_resolution_clock::now();
    
    kdtree.radiusSearch(&query_point.x, radius * radius, ret_indices_dists);
    
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;
    cout << "RadiusNN search time for R=" << radius << ": " << elapsed.count() * 1000 << " ms, found " << ret_indices_dists.size() << " points" << endl;
}
