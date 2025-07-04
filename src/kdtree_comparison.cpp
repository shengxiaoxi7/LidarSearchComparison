#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/voxel_grid.h>
#include <ikd-Tree/ikd_Tree.h>
#include <chrono>
#include <iostream>
#include <vector>
#include <random>

using namespace std::chrono;
using PointType = pcl::PointXYZ;
using PointVector = KD_TREE<PointType>::PointVector;

// 生成随机点云
pcl::PointCloud<pcl::PointXYZ>::Ptr generatePointCloud(size_t num_points) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    cloud->resize(num_points);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-10.0, 10.0);
    
    for (size_t i = 0; i < num_points; ++i) {
        cloud->points[i].x = dis(gen);
        cloud->points[i].y = dis(gen);
        cloud->points[i].z = dis(gen);
    }
    return cloud;
}

// 测试KD-Tree性能
#if 0
void testKdTree(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, 
                size_t num_operations, float voxel_size) {
    // ================== 构建树 ==================
    auto start = high_resolution_clock::now();
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(cloud);
    auto build_time = duration_cast<microseconds>(high_resolution_clock::now() - start).count() / 1000.0;

    // ================== 查询操作 ==================
    start = high_resolution_clock::now();
    std::vector<int> indices;
    std::vector<float> distances;
    for (size_t i = 0; i < num_operations; ++i) {
        kdtree.nearestKSearch(cloud->points[i % cloud->size()], 5, indices, distances);
    }
    auto query_time = duration_cast<microseconds>(high_resolution_clock::now() - start).count() / 1000.0;

    // ================== 插入操作 ==================
    pcl::PointCloud<pcl::PointXYZ>::Ptr insert_cloud = generatePointCloud(num_operations);
    
    start = high_resolution_clock::now();
    pcl::PointCloud<pcl::PointXYZ>::Ptr new_cloud(new pcl::PointCloud<pcl::PointXYZ>(*cloud));
    *new_cloud += *insert_cloud;
    kdtree.setInputCloud(new_cloud); // 需要完全重建
    auto insert_time = duration_cast<microseconds>(high_resolution_clock::now() - start).count() / 1000.0;

    // ================== 删除操作 ==================
    start = high_resolution_clock::now();
    pcl::PointCloud<pcl::PointXYZ>::Ptr smaller_cloud(new pcl::PointCloud<pcl::PointXYZ>(*new_cloud));
    smaller_cloud->resize(smaller_cloud->size() - num_operations);
    kdtree.setInputCloud(smaller_cloud); // 需要完全重建
    auto delete_time = duration_cast<microseconds>(high_resolution_clock::now() - start).count() / 1000.0;

    // ================== 修改操作 ==================
    start = high_resolution_clock::now();
    for (size_t i = 0; i < num_operations; ++i) {
        (*smaller_cloud)[i].x += 0.1f;
    }
    kdtree.setInputCloud(smaller_cloud); // 需要完全重建
    auto update_time = duration_cast<microseconds>(high_resolution_clock::now() - start).count() / 1000.0;

    // ================== 下采样 ==================
    start = high_resolution_clock::now();
    pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
    voxel_filter.setInputCloud(smaller_cloud);
    voxel_filter.setLeafSize(voxel_size, voxel_size, voxel_size);
    pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled(new pcl::PointCloud<pcl::PointXYZ>);
    voxel_filter.filter(*downsampled);
    kdtree.setInputCloud(downsampled); // 需要完全重建
    auto downsample_time = duration_cast<microseconds>(high_resolution_clock::now() - start).count() / 1000.0;

    // 输出结果
    std::cout << "\n======== KD-Tree 性能结果 ========\n";
    std::cout << "构建时间: " << build_time << " ms\n";
    std::cout << "查询时间 (" << num_operations << " 次): " << query_time << " ms\n";
    std::cout << "插入时间 (" << num_operations << " 点): " << insert_time << " ms\n";
    std::cout << "删除时间 (" << num_operations << " 点): " << delete_time << " ms\n";
    std::cout << "修改时间 (" << num_operations << " 点): " << update_time << " ms\n";
    std::cout << "下采样时间 (体素 " << voxel_size << "m): " << downsample_time << " ms\n";
    std::cout << "下采样后点数: " << downsampled->size() << "/" << smaller_cloud->size() << "\n";
}
#endif
//KD_TREE<PointType> ikdtree; 局部创建会段错误，全局或使用new则OK
// 测试IKD-Tree性能
void testIkdTree(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, 
                 size_t num_operations, float voxel_size) {

    //PointVector points;
    //for (size_t i = 0; i < num_operations; ++i) {
    //    points.push_back(cloud->points[i]);
    //}
    //ikdtree.Build(points);

    // ================== 构建树 ==================
    auto start = high_resolution_clock::now();
    KD_TREE<PointType>::Ptr ikdtree_ptr(new KD_TREE<PointType>());
    KD_TREE<PointType>      &ikdtree        = *ikdtree_ptr;
    //KD_TREE<PointType> ikdtree; 局部创建会段错误

    ikdtree.set_downsample_param(voxel_size); 
    ikdtree.Build((*cloud).points);
    auto build_time = duration_cast<microseconds>(high_resolution_clock::now() - start).count() / 1000.0;

    // ================== 查询操作 ==================
    start = high_resolution_clock::now();
    PointVector search_result;
    std::vector<float> distances;
    for (size_t i = 0; i < num_operations; ++i) {
        ikdtree.Nearest_Search(cloud->points[i % cloud->size()], 5, search_result, distances);
    }
    auto query_time = duration_cast<microseconds>(high_resolution_clock::now() - start).count() / 1000.0;

    // ================== 插入操作 ==================
    PointType point;
    PointVector cloud_increment;

    for (size_t i = 0; i < num_operations; ++i) {
        point.x = cloud->points[i].x + 5.0;
        point.y = cloud->points[i].y + 5.0;
        point.z = cloud->points[i].z + 5.0;

        cloud_increment.push_back(point);    
    }
    
    start = high_resolution_clock::now();
    ikdtree.Add_Points(cloud_increment, false); // 增量插入
    auto insert_time = duration_cast<microseconds>(high_resolution_clock::now() - start).count() / 1000.0;

    // ================== 删除操作 ==================
    PointVector cloud_decrement;
    for (size_t i = 0; i < num_operations; ++i) {
        cloud_decrement.push_back(cloud->points[i]);
    }

    start = high_resolution_clock::now();
    ikdtree.Delete_Points(cloud_decrement); // 增量删除
    auto delete_time = duration_cast<microseconds>(high_resolution_clock::now() - start).count() / 1000.0;

    // ================== 修改操作 ==================
    PointVector original_points, updated_points;
    for (size_t i = num_operations; i < 2 * num_operations; ++i) {
        original_points.push_back(cloud->points[i]);
        updated_points.push_back(PointType(
            cloud->points[i].x + 0.1,
            cloud->points[i].y + 0.1,
            cloud->points[i].z + 0.1
        ));
    }
    
    start = high_resolution_clock::now();
    ikdtree.Delete_Points(original_points); // 删除旧点
    ikdtree.Add_Points(updated_points, false); // 添加新点
    auto update_time = duration_cast<microseconds>(high_resolution_clock::now() - start).count() / 1000.0;

    // ================== 下采样 ==================
    start = high_resolution_clock::now();
    //ikdtree.downsample(); // 内置下采样  build&Add自动下采样，没有单独的下采样接口
    auto downsample_time = duration_cast<microseconds>(high_resolution_clock::now() - start).count() / 1000.0;

    // 输出结果
    std::cout << "\n======== IKD-Tree 性能结果 ========\n";
    std::cout << "构建时间: " << build_time << " ms\n";
    std::cout << "查询时间 (" << num_operations << " 次): " << query_time << " ms\n";
    std::cout << "插入时间 (" << num_operations << " 点): " << insert_time << " ms\n";
    std::cout << "删除时间 (" << num_operations << " 点): " << delete_time << " ms\n";
    std::cout << "修改时间 (" << num_operations << " 点): " << update_time << " ms\n";
    std::cout << "下采样时间 (体素 " << voxel_size << "m): " << downsample_time << " ms\n";
    std::cout << "下采样后点数: " << ikdtree.validnum() << "/" << cloud->size() << "\n";
}
// 测试IKD-Tree性能
void testIkdTree(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, 
                 size_t num_operations, float voxel_size) {
    // 转换点云格式
    std::vector<PointType> points;
    for (const auto& p : cloud->points) {
        points.push_back(PointType(p.x, p.y, p.z));
    }
    
    KD_TREE<PointType> ikdtree;
    
    // ================== 构建树 ==================
    auto start = high_resolution_clock::now();
    ikdtree.Build(points);
    auto build_time = duration_cast<microseconds>(high_resolution_clock::now() - start).count() / 1000.0;

    // ================== 查询操作 ==================
    start = high_resolution_clock::now();
    for (size_t i = 0; i < num_operations; ++i) {
        PointType query_point = points[i % points.size()];
        auto results = ikdtree.KNN_Search(query_point, 5);
    }
    auto query_time = duration_cast<microseconds>(high_resolution_clock::now() - start).count() / 1000.0;

    // ================== 插入操作 ==================
    std::vector<PointType> new_points = points;
    for (size_t i = 0; i < num_operations; ++i) {
        new_points.push_back(PointType(
            points[i].x + 5.0, 
            points[i].y + 5.0, 
            points[i].z + 5.0
        ));
    }
    
    start = high_resolution_clock::now();
    ikdtree.Add_Points(new_points, false); // 增量插入，不开启下采样
    auto insert_time = duration_cast<microseconds>(high_resolution_clock::now() - start).count() / 1000.0;

    // ================== 删除操作 ==================
    std::vector<PointType> points_to_delete;
    for (size_t i = 0; i < num_operations; ++i) {
        points_to_delete.push_back(points[i]);
    }
    
    start = high_resolution_clock::now();
    ikdtree.Delete_Points(points_to_delete); // 增量删除
    auto delete_time = duration_cast<microseconds>(high_resolution_clock::now() - start).count() / 1000.0;

    // ================== 修改操作 ==================
    std::vector<PointType> original_points, updated_points;
    for (size_t i = num_operations; i < 2 * num_operations; ++i) {
        original_points.push_back(points[i]);
        updated_points.push_back(PointType(
            points[i].x + 0.1,
            points[i].y + 0.1,
            points[i].z + 0.1
        ));
    }
    
    start = high_resolution_clock::now();
    ikdtree.Delete_Points(original_points); // 删除旧点
    ikdtree.Add_Points(updated_points, false); // 添加新点，不开启下采样
    auto update_time = duration_cast<microseconds>(high_resolution_clock::now() - start).count() / 1000.0;

    // ================== 下采样 ==================
    start = high_resolution_clock::now();
    ikdtree.Downsample(voxel_size); // 内置下采样
    auto downsample_time = duration_cast<microseconds>(high_resolution_clock::now() - start).count() / 1000.0;

    // 输出结果
    std::cout << "\n======== IKD-Tree 性能结果 ========\n";
    std::cout << "构建时间: " << build_time << " ms\n";
    std::cout << "查询时间 (" << num_operations << " 次): " << query_time << " ms\n";
    std::cout << "插入时间 (" << num_operations << " 点): " << insert_time << " ms\n";
    std::cout << "删除时间 (" << num_operations << " 点): " << delete_time << " ms\n";
    std::cout << "修改时间 (" << num_operations << " 点): " << update_time << " ms\n";
    std::cout << "下采样时间 (体素 " << voxel_size << "m): " << downsample_time << " ms\n";
    std::cout << "下采样后点数: " << ikdtree.validnum() << "/" << new_points.size() << "\n";
}

int main() {
    const size_t CLOUD_SIZE = 100000;     // 初始点云大小
    const size_t OPERATIONS = 1000;       // 操作次数
    const float VOXEL_SIZE = 0.1f;        // 下采样体素大小
    
    // 生成测试点云
    auto cloud = generatePointCloud(CLOUD_SIZE);
    std::cout << "生成测试点云: " << CLOUD_SIZE << " 个点\n";
    
    // 测试KD-Tree
    testKdTree(cloud, OPERATIONS, VOXEL_SIZE);
    
    // 测试IKD-Tree
    testIkdTree(cloud, OPERATIONS, VOXEL_SIZE);
    
    return 0;
}
