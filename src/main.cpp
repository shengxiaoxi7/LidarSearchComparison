
#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/PointCloud2.h>
#include "ikd_tree.h"
#include "nanoflann.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <random>
#include <algorithm>
#include "pcl/point_types.h"
#include "pcl/common/common.h"
#include "pcl/point_cloud.h"
#include <pcl/io/pcd_io.h>
#include <pcl_ros/point_cloud.h> 
#include <pcl/visualization/pcl_visualizer.h>

using namespace std;
using PointType = pcl::PointXYZ;
using PointVector = KD_TREE<PointType>::PointVector;
template class KD_TREE<pcl::PointXYZ>;

static int N_ = 100;

struct NanoPointCloud {
    std::vector<pcl::PointXYZ> pts;

    inline size_t kdtree_get_point_count() const { return pts.size(); }
    inline double kdtree_get_pt(const size_t idx, const size_t dim) const {
        if (dim == 0) return pts[idx].x;
        else if (dim == 1) return pts[idx].y;
        else           return pts[idx].z;
    }
    template <class BBOX>
    bool kdtree_get_bbox(BBOX&) const { return false; }
};

// 2) 定义 nanoflann 的树类型
using nano_kdtree = nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<float, NanoPointCloud>, NanoPointCloud, 3 /* dim */>;

void colorize( const PointVector &pc, pcl::PointCloud<pcl::PointXYZRGB> &pc_colored, const std::vector<int> &color) {
    int N = pc.size();

    pc_colored.clear();
    pcl::PointXYZRGB pt_tmp;

    for (int i = 0; i < N; ++i) {
        const auto &pt = pc[i];
        pt_tmp.x = pt.x;
        pt_tmp.y = pt.y;
        pt_tmp.z = pt.z;
        pt_tmp.r = color[0];
        pt_tmp.g = color[1];
        pt_tmp.b = color[2];
        pc_colored.points.emplace_back(pt_tmp);
    }
}

void generate_box(BoxPointType &boxpoint, const PointType &center_pt, vector<float> box_lengths) {
    float &x_dist = box_lengths[0];
    float &y_dist = box_lengths[1];
    float &z_dist = box_lengths[2];

    boxpoint.vertex_min[0] = center_pt.x - x_dist;
    boxpoint.vertex_max[0] = center_pt.x + x_dist;
    boxpoint.vertex_min[1] = center_pt.y - y_dist;
    boxpoint.vertex_max[1] = center_pt.y + y_dist;
    boxpoint.vertex_min[2] = center_pt.z - z_dist;
    boxpoint.vertex_max[2] = center_pt.z + z_dist;
}

void readfromrosbag(const std::string& bag_file, std::vector<std::pair<int, pcl::PointCloud<PointType>::Ptr>>& frames)
{
    rosbag::Bag bag;
    bag.open(bag_file, rosbag::bagmode::Read);
    rosbag::View view(bag, rosbag::TopicQuery("/cloud_registered"));

    int frame_count = 0;
    for (const auto& m : view) {
        sensor_msgs::PointCloud2::ConstPtr msg = m.instantiate<sensor_msgs::PointCloud2>();
        if (msg) {
            pcl::PointCloud<PointType>::Ptr cloud(new pcl::PointCloud<PointType>);
            pcl::fromROSMsg(*msg, *cloud);

            frames.push_back(std::make_pair(frame_count, cloud));
            frame_count++;
        }
    }
    bag.close();
}

void rebuild_nanoflann(std::vector<std::pair<int, pcl::PointCloud<PointType>::Ptr>>& frames, int N) {
    NanoPointCloud cloud;
    for (int i = N_; i < N_ + N; i++) {
        const auto& pc = *frames[i].second;
        cloud.pts.insert(cloud.pts.end(), pc.points.begin(), pc.points.end());
    }
    cout << "Total points in " << N << " frames: " << cloud.pts.size() << std::endl;

    const int rounds = 100;
    long long total_rebuild_time = 0;
    
    for (int r = 0; r < rounds; ++r) {
        auto start = std::chrono::high_resolution_clock::now();
        nano_kdtree kdtree(3, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));
        kdtree.buildIndex();
        auto end = std::chrono::high_resolution_clock::now();
        total_rebuild_time += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    }
    std::cout << "Average rebuilding nanoflann tree for " << N << " frames over " << rounds << " runs took: " << total_rebuild_time / double(rounds) << " µs" << std::endl;

}

void search_nanoflann(std::vector<std::pair<int, pcl::PointCloud<PointType>::Ptr>>& frames, int N) {
    NanoPointCloud cloud;
    for (int i = N_; i < N_ + N; i++) {
        const auto& pc = *frames[i].second;
        cloud.pts.insert(cloud.pts.end(), pc.points.begin(), pc.points.end());
    }
    cout << "Total points in " << N << " frames: " << cloud.pts.size() << std::endl;

    const int rounds = 100;
    nano_kdtree kdtree(3, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));
    kdtree.buildIndex();

    float query_point[3] = {cloud.pts[1000].x, cloud.pts[1000].y, cloud.pts[1000].z};
    cout << "\n" << string(20, '=') << " knn search " << string(20, '=') << endl;
    for (size_t K : {5, 10, 20}) {
        long long total_knn_time = 0;
        std::vector<unsigned int> indices(K);
        std::vector<float> distances(K);

        for (int r = 0; r < rounds; ++r) {
            auto knn_search_start = std::chrono::high_resolution_clock::now();
            kdtree.knnSearch(&query_point[0], K, indices.data(), distances.data());
            auto knn_search_end = std::chrono::high_resolution_clock::now();
            total_knn_time += std::chrono::duration_cast<std::chrono::microseconds>(knn_search_end - knn_search_start).count();
        }
        std::cout << "Average KNN Search with K=" << K << " over " << rounds << " runs took: " << total_knn_time / double(rounds) << " µs" << std::endl;
        // std::cout << "Found " << found << " nearest points for query point" << std::endl;
    }

    cout << "\n" << string(20, '=') << " radius search " << string(20, '=') << endl;
    for (float R : {0.5f, 1.0f, 5.0f}) {
        std::vector<nanoflann::ResultItem<uint32_t, float>> indices_dists;
        nanoflann::SearchParameters params;
        params.sorted = false;
        long long total_radius_time = 0;

        for (int r = 0; r < rounds; ++r) {
            auto radius_search_start = std::chrono::high_resolution_clock::now();
            kdtree.radiusSearch(&query_point[0], R * R, indices_dists, params);
            auto radius_search_end = std::chrono::high_resolution_clock::now();
            total_radius_time += std::chrono::duration_cast<std::chrono::microseconds>(radius_search_end - radius_search_start).count();
        }
        std::cout << "Average Radius Search with R=" << R << " over " << rounds << " runs took: " << total_radius_time / double(rounds) << " µs" << std::endl;
        // std::cout << "Found " << found << " points within radius " << R << std::endl;
    }
}

void rebuild_ikdtree(std::vector<std::pair<int, pcl::PointCloud<PointType>::Ptr>>& frames, int N) {
    KD_TREE<PointType>::Ptr kdtree_ptr(new KD_TREE<PointType>(0.3, 0.6, 0.2));
    KD_TREE<PointType> &ikd_Tree = *kdtree_ptr;

    pcl::PointCloud<PointType>::Ptr cloud(new pcl::PointCloud<PointType>());
    for (int i = N_; i < N_ + N; i++) {
        *cloud += *frames[i].second;
    }
    cout << "Total points in " << N << " frames: " << cloud->points.size() << std::endl;

    const int rounds = 100;
    long long total_rebuild_time = 0;

    for (int r = 0; r < rounds; ++r) {
        auto start = std::chrono::high_resolution_clock::now();
        ikd_Tree.Build(cloud->points);
        auto end = std::chrono::high_resolution_clock::now();
        total_rebuild_time += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    }
    std::cout << "Average rebuilding ikd-tree for " << N << " frames over " << rounds << " runs took: " << total_rebuild_time / double(rounds) << " µs" << std::endl;
}

void search_ikdtree(std::vector<std::pair<int, pcl::PointCloud<PointType>::Ptr>>& frames, int N) {
    KD_TREE<PointType>::Ptr kdtree_ptr(new KD_TREE<PointType>(0.3, 0.6, 0.2));
    KD_TREE<PointType> &ikd_Tree = *kdtree_ptr;

    pcl::PointCloud<PointType>::Ptr cloud(new pcl::PointCloud<PointType>());
    for (int i = N_; i < N_ + N; i++) {
        *cloud += *frames[i].second;
    }
    cout << "Total points in " << N << " frames: " << cloud->points.size() << std::endl;

    const int rounds = 100;
    ikd_Tree.Build(cloud->points);

    cout << "\n" << string(20, '=') << " knn search " << string(20, '=') << endl;
    for (int K : {5, 10, 20}) {
        PointVector Nearest_Points;
        std::vector<float> distances(K);
        long long total_knn_time = 0;
        PointType query_point = cloud->points[1000];

        for (int r = 0; r < rounds; ++r) {
            auto knn_search_start = std::chrono::high_resolution_clock::now();
            ikd_Tree.Nearest_Search(query_point, K, Nearest_Points, distances, 0.5);
            auto knn_search_end = std::chrono::high_resolution_clock::now();
            total_knn_time += std::chrono::duration_cast<std::chrono::microseconds>(knn_search_end - knn_search_start).count();
        }
        std::cout << "Average KNN Search with K=" << K << " over " << rounds << " runs took: " << total_knn_time / double(rounds) << " µs" << std::endl;
    }
    
    cout << "\n" << string(20, '=') << " radius search " << string(20, '=') << endl;
    for (float R : {0.5f, 1.0f, 5.0f}) {
        PointVector Storage;
        long long total_radius_time = 0;
        PointType query_point = cloud->points[1000];

        for (int r = 0; r < rounds; ++r) {
            auto radius_search_start = std::chrono::high_resolution_clock::now();
            ikd_Tree.Radius_Search(query_point, R, Storage);
            auto radius_search_end = std::chrono::high_resolution_clock::now();
            total_radius_time += std::chrono::duration_cast<std::chrono::microseconds>(radius_search_end - radius_search_start).count();
        }
        std::cout << "Average Radius Search with R=" << R << " over " << rounds << " runs took: " << total_radius_time / double(rounds) << " µs" << std::endl;
    }

}

// void incremental_ikdtree(std::vector<std::pair<int, pcl::PointCloud<PointType>::Ptr>>& frames, int N) {

//     KD_TREE<PointType>::Ptr kdtree_ptr(new KD_TREE<PointType>(0.3, 0.6, 0.2));
//     KD_TREE<PointType> &ikd_Tree = *kdtree_ptr;

//     pcl::PointCloud<PointType>::Ptr cloud(new pcl::PointCloud<PointType>());
//     for (int i = N_; i < N_ + N; i++) {
//         *cloud += *frames[i].second;
//     }
//     std::cout << "Total points in " << N << " frames: " << cloud->points.size() << std::endl;

//     ikd_Tree.Build(cloud->points);

//     pcl::PointCloud<PointType>::Ptr add_cloud = frames[N].second; // 获取第 N+1 帧点云
//     pcl::PointCloud<PointType>::Ptr delete_cloud = frames[0].second;

//     auto delete_start = std::chrono::high_resolution_clock::now();
//     ikd_Tree.Delete_Points(delete_cloud->points);  // 删除第 0 帧点云
//     auto delete_end = std::chrono::high_resolution_clock::now();
//     auto delete_duration = std::chrono::duration_cast<std::chrono::microseconds>(delete_end - delete_start).count();
//     std::cout << "Deleting " << delete_cloud->points.size() << " points took: " << delete_duration << " µs" << std::endl;

//     auto add_start = std::chrono::high_resolution_clock::now();
//     ikd_Tree.Add_Points(add_cloud->points, false);  // 增量添加点云
//     auto add_end = std::chrono::high_resolution_clock::now();
//     auto add_duration = std::chrono::duration_cast<std::chrono::microseconds>(add_end - add_start).count();
//     std::cout << "Adding " << add_cloud->points.size() << " points took: " << add_duration << " µs" << std::endl;
//     std::cout << "Total time : " << (add_duration + delete_duration) << " µs" << std::endl;

// }

void incremental_ikdtree(std::vector<std::pair<int, pcl::PointCloud<PointType>::Ptr>>& frames, int N) {

    pcl::PointCloud<PointType>::Ptr cloud(new pcl::PointCloud<PointType>());
    for (int i = N_ - 1; i < N_ + N - 1; i++) {
        *cloud += *frames[i].second;
    }
    std::cout << "Total points in " << N << " frames: " << cloud->points.size() << std::endl;

    const int rounds = 100;
    long long total_Inc_build_time = 0;

    for (int r = 0; r < rounds; ++r) {

        KD_TREE<PointType>::Ptr kdtree_ptr(new KD_TREE<PointType>(0.3, 0.6, 0.2));
        KD_TREE<PointType> &ikd_Tree = *kdtree_ptr;
        ikd_Tree.Build(cloud->points);

        pcl::PointCloud<PointType>::Ptr add_cloud = frames[N].second; // 获取第 N+1 帧点云
        pcl::PointCloud<PointType>::Ptr delete_cloud = frames[0].second;

        auto start = std::chrono::high_resolution_clock::now();
        ikd_Tree.Delete_Points(delete_cloud->points);
        ikd_Tree.Add_Points(add_cloud->points, false);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        total_Inc_build_time += duration;

    }
    std::cout << "Average incremental build time for " << N << " frames over " << rounds << " runs: " 
              << total_Inc_build_time / double(rounds) << " µs" << std::endl;
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "LidarSearchComparison");

    std::string bag_file = "../data/9.bag";
    std::vector<std::pair<int, pcl::PointCloud<PointType>::Ptr>> frames;

    readfromrosbag(bag_file, frames);
    cout << "Loaded " << frames.size() << " frames from bag file." << endl;

    for (int N : {3, 6, 30}){
        cout << "Processing " << N << " frames..." << endl;

        cout << "\n" << string(30, '=') << " 1. nanoflann rebuild and search " << string(30, '=') << endl;
        rebuild_nanoflann(frames, N);
        search_nanoflann(frames, N);

        cout << "\n" << string(30, '=') << " 2. ikdtree rebuild and search " << string(30, '=') << endl;
        rebuild_ikdtree(frames, N);
        search_ikdtree(frames, N);
        
        cout << "\n" << string(30, '=') << " 3. incremental_ikdtree build " << string(30, '=') << endl;
        incremental_ikdtree(frames, N);
    }

    return 0;
}