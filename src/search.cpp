#include <nanoflann.hpp>
#include <chrono>
#include "tree_construction.cpp"  // 引入树构建代码

void knn_search(const my_kd_tree_t& kdtree, const pcl::PointXYZ& query_point, int k) {
    std::vector<size_t> ret_index(k);
    std::vector<float> out_distances(k);
    
    auto start = chrono::high_resolution_clock::now();
    
    kdtree.knnSearch(&query_point.x, k, &ret_index[0], &out_distances[0]);
    
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;
    cout << "KNN search time for K=" << k << ": " << elapsed.count() << " seconds" << endl;
}

void radius_nn_search(const my_kd_tree_t& kdtree, const pcl::PointXYZ& query_point, float radius) {
    std::vector<size_t> ret_index;
    std::vector<float> out_distances;
    
    auto start = chrono::high_resolution_clock::now();
    
    kdtree.radiusSearch(&query_point.x, radius, ret_index, out_distances);
    
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;
    cout << "RadiusNN search time for R=" << radius << ": " << elapsed.count() << " seconds" << endl;
}
