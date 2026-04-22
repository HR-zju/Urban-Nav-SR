/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *
 * Author: Qin Tong (qintonguav@gmail.com)
 *******************************************************/

#include <rclcpp/rclcpp.hpp>
#include "globalOpt.h"
#include <sensor_msgs/msg/nav_sat_fix.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>
#include <iostream>
#include <stdio.h>
#include <queue>
#include <mutex>
#include <functional>

GlobalOptimization globalEstimator;
rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_global_odometry;
rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_global_path;
rclcpp::Publisher<sensor_msgs::msg::NavSatFix>::SharedPtr pub_global_lla;
nav_msgs::msg::Path *global_path;
std::string world_frame_id;
std::string body_frame_id;
double last_vio_t = -1;
std::queue<sensor_msgs::msg::NavSatFix::SharedPtr> gpsQueue;
std::mutex m_buf;

void vio_callback(const nav_msgs::msg::Odometry::SharedPtr pose_msg)
{
    double t = pose_msg->header.stamp.sec + pose_msg->header.stamp.nanosec * 1e-9;
    last_vio_t = t;
    Eigen::Vector3d vio_t(pose_msg->pose.pose.position.x, pose_msg->pose.pose.position.y, pose_msg->pose.pose.position.z);
    Eigen::Quaterniond vio_q;
    vio_q.w() = pose_msg->pose.pose.orientation.w;
    vio_q.x() = pose_msg->pose.pose.orientation.x;
    vio_q.y() = pose_msg->pose.pose.orientation.y;
    vio_q.z() = pose_msg->pose.pose.orientation.z;

    globalEstimator.inputOdom(t, vio_t, vio_q);


    m_buf.lock();
    while(!gpsQueue.empty())
    {
        sensor_msgs::msg::NavSatFix::ConstPtr GPS_msg = gpsQueue.front();
        double gps_t = GPS_msg->header.stamp.sec + GPS_msg->header.stamp.nanosec * 1e-9;
        
        // Auto-align massive time offsets (e.g. days of difference in dataset)
        static bool time_offset_init = false;
        static double time_offset = 0.0;
        if (!time_offset_init) {
            time_offset = gps_t - t;
            time_offset_init = true;
            printf("[Global Fusion] Auto-aligned GPS and VIO time. Offset: %f s\n", time_offset);
        }

        double aligned_gps_t = gps_t - time_offset;

        // 20ms sync tolerance
        if(aligned_gps_t >= t - 0.02 && aligned_gps_t <= t + 0.02)
        {
            double latitude = GPS_msg->latitude;
            double longitude = GPS_msg->longitude;
            double altitude = GPS_msg->altitude;
            double pos_accuracy = GPS_msg->position_covariance[0];
            if(pos_accuracy <= 0)
                pos_accuracy = 1;
            globalEstimator.inputGPS(t, latitude, longitude, altitude, pos_accuracy);
            gpsQueue.pop();
            break;
        }
        else if(aligned_gps_t < t - 0.02)
            gpsQueue.pop();
        else if(aligned_gps_t > t + 0.02)
            break;
    }
    m_buf.unlock();

    Eigen::Vector3d global_t;
    Eigen:: Quaterniond global_q;
    globalEstimator.getGlobalOdom(global_t, global_q);

    nav_msgs::msg::Odometry odometry;
    odometry.header = pose_msg->header;
    odometry.header.frame_id = world_frame_id;
    odometry.child_frame_id = body_frame_id;
    odometry.pose.pose.position.x = global_t.x();
    odometry.pose.pose.position.y = global_t.y();
    odometry.pose.pose.position.z = global_t.z();
    odometry.pose.pose.orientation.x = global_q.x();
    odometry.pose.pose.orientation.y = global_q.y();
    odometry.pose.pose.orientation.z = global_q.z();
    odometry.pose.pose.orientation.w = global_q.w();
    pub_global_odometry->publish(odometry);
    pub_global_path->publish(*global_path);

    double lat, lon, alt;
    if (globalEstimator.XYZ2LLA(global_t, lat, lon, alt))
    {
        sensor_msgs::msg::NavSatFix lla_msg;
        lla_msg.header = pose_msg->header;
        lla_msg.header.frame_id = world_frame_id;
        lla_msg.latitude = lat;
        lla_msg.longitude = lon;
        lla_msg.altitude = alt;
        pub_global_lla->publish(lla_msg);

        globalEstimator.appendKML(lat, lon, alt);
    } else {
        std::cout << "[Global Fusion] Failed to convert XYZ to LLA" << std::endl;
    }
}


void GPS_callback(const sensor_msgs::msg::NavSatFix::SharedPtr GPS_msg)
{
    m_buf.lock();
    gpsQueue.push(GPS_msg);
    m_buf.unlock();
}




int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto n = rclcpp::Node::make_shared("globalEstimator");
    auto qos_profile = rclcpp::QoS(rclcpp::KeepLast(100));

    n->declare_parameter<std::string>("world_frame_id", "world");
    n->declare_parameter<std::string>("body_frame_id", "body");
    n->declare_parameter<std::string>("kml_output_path", "/tmp/global_fusion_path.kml");
    world_frame_id = n->get_parameter("world_frame_id").as_string();
    body_frame_id = n->get_parameter("body_frame_id").as_string();
    globalEstimator.world_frame_id = world_frame_id;
    
    std::string kml_output_path = n->get_parameter("kml_output_path").as_string();
    globalEstimator.openKML(kml_output_path);

    auto sub_GPS = n->create_subscription<sensor_msgs::msg::NavSatFix>("/baton/gnss/spp_fix_raw", rclcpp::QoS(rclcpp::KeepLast(100)), GPS_callback);

    auto sub_vio = n->create_subscription<nav_msgs::msg::Odometry>("/baton/stereo3/odometry", rclcpp::QoS(rclcpp::KeepLast(100)), vio_callback);

    pub_global_path = n->create_publisher<nav_msgs::msg::Path>("global_path", 100);
    pub_global_odometry = n->create_publisher<nav_msgs::msg::Odometry>("global_odometry", 100);
    pub_global_lla = n->create_publisher<sensor_msgs::msg::NavSatFix>("/baton/nav/global_lla", 100);

    global_path = &(globalEstimator.global_path);
    rclcpp::spin(n);
    return 0;
}

