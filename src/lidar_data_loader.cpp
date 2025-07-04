#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <boost/foreach.hpp>
#include <iostream>
#include <vector>
#include <stdexcept>

using namespace pcl;
using namespace std;

// 从 rosbag 文件加载 LiDAR 点云数据
std::vector<pcl::PointXYZ> load_lidar_data_from_rosbag(const std::string& rosbag_file) {
    std::vector<pcl::PointXYZ> points;
    
    try {
        rosbag::Bag bag;
        
        // 尝试打开 bag 文件
        bag.open(rosbag_file, rosbag::bagmode::Read);
        
        rosbag::View view(bag);
        
        // 检查 bag 文件是否为空
        if (view.size() == 0) {
            std::cerr << "Warning: Bag file is empty: " << rosbag_file << std::endl;
            bag.close();
            return points;
        }
        
        int frame_count = 0;
        int total_points = 0;
        
        BOOST_FOREACH(rosbag::MessageInstance const m, view) {
            // 检查话题是否为 LiDAR 点云数据
            if (m.getTopic() == "/cloud_registered" || m.getTopic() == "/livox/lidar") {
                sensor_msgs::PointCloud2::ConstPtr msg = m.instantiate<sensor_msgs::PointCloud2>();
                if (msg != nullptr) {
                    try {
                        PointCloud<PointXYZ> cloud;
                        pcl::fromROSMsg(*msg, cloud);
                        
                        // 检查点云是否有效
                        if (!cloud.empty()) {
                            points.insert(points.end(), cloud.points.begin(), cloud.points.end());
                            frame_count++;
                            total_points += cloud.points.size();
                            
                            // 可选：限制加载的帧数，避免内存溢出
                            if (frame_count >= 100) {  // 最多加载100帧
                                std::cout << "Reached maximum frame limit (100), stopping..." << std::endl;
                                break;
                            }
                        }
                    } catch (const std::exception& e) {
                        std::cerr << "Error converting PointCloud2 message: " << e.what() << std::endl;
                        continue;  // 跳过这一帧，继续处理下一帧
                    }
                }
            }
        }
        
        bag.close();
        
        std::cout << "Successfully loaded " << frame_count << " frames with " 
                  << total_points << " total points from " << rosbag_file << std::endl;
        
    } catch (const rosbag::BagException& e) {
        std::cerr << "Error opening bag file " << rosbag_file << ": " << e.what() << std::endl;
        return points;  // 返回空的点云
    } catch (const std::exception& e) {
        std::cerr << "Unexpected error: " << e.what() << std::endl;
        return points;
    }
    
    return points;
}

// 按帧加载 LiDAR 点云数据（针对性能对比测试优化）
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
            if (m.getTopic() == "/cloud_registered" || m.getTopic() == "/livox/lidar") {
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
                                std::cout << "Loaded target frame count (" << max_frames << "), stopping..." << std::endl;
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
        
        std::cout << "Successfully loaded " << frames.size() << " consecutive frames from " << rosbag_file << std::endl;
        
        // 只显示前几帧的统计信息，避免输出太多
        int display_count = std::min((int)frames.size(), 10);
        for (int i = 0; i < display_count; ++i) {
            std::cout << "Frame " << i << ": " << frames[i].size() << " points" << std::endl;
        }
        if (frames.size() > display_count) {
            std::cout << "... (showing first " << display_count << " frames)" << std::endl;
        }
        
    } catch (const rosbag::BagException& e) {
        std::cerr << "Error opening bag file " << rosbag_file << ": " << e.what() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Unexpected error: " << e.what() << std::endl;
    }
    
    return frames;
}

// 专门为性能对比测试设计的数据准备函数
struct PerformanceTestData {
    std::vector<std::vector<pcl::PointXYZ>> consecutive_frames;  // 连续的N帧数据
    std::vector<pcl::PointXYZ> merged_points;                   // 合并后的点云（用于重新构建测试）
    int frame_count;
    int total_points;
};

// 准备性能测试数据（针对N=3,6,30的测试需求）
PerformanceTestData prepare_performance_test_data(const std::string& rosbag_file, int N) {
    PerformanceTestData test_data;
    
    // 加载连续的N帧数据
    auto frames = load_lidar_frames_from_rosbag(rosbag_file, N);
    
    if (frames.size() < N) {
        std::cerr << "Warning: Only loaded " << frames.size() << " frames, but " << N << " requested" << std::endl;
    }
    
    test_data.consecutive_frames = frames;
    test_data.frame_count = frames.size();
    test_data.total_points = 0;
    
    // 合并所有帧的点云（用于重新构建测试）
    for (const auto& frame : frames) {
        test_data.merged_points.insert(test_data.merged_points.end(), frame.begin(), frame.end());
        test_data.total_points += frame.size();
    }
    
    std::cout << "\n=== Performance Test Data Prepared ===" << std::endl;
    std::cout << "Frame count: " << test_data.frame_count << std::endl;
    std::cout << "Total points: " << test_data.total_points << std::endl;
    std::cout << "Average points per frame: " << (test_data.frame_count > 0 ? test_data.total_points / test_data.frame_count : 0) << std::endl;
    
    return test_data;
}
