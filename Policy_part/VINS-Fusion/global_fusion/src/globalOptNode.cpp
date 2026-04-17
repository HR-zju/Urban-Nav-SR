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

#include "ros/ros.h"
#include "globalOpt.h"
#include <sensor_msgs/NavSatFix.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>
#include <iostream>
#include <stdio.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <fstream>
#include <queue>
#include <mutex>
#include <iomanip>
#include <vector>
#include <string>
#include <regex>
#include <sstream>

GlobalOptimization globalEstimator;
ros::Publisher pub_global_odometry, pub_global_path, pub_car;
nav_msgs::Path *global_path;
double last_vio_t = -1;
std::queue<sensor_msgs::NavSatFixConstPtr> gpsQueue;
std::mutex m_buf;

std::string route_json_path;
std::string route_output_path;
std::string last_route_json_content;

struct JsonWaypoint
{
    double latitude;
    double longitude;
    double altitude;
    double qw;
    double qx;
    double qy;
    double qz;
};

bool read_file_to_string(const std::string &path, std::string &content)
{
    std::ifstream fin(path.c_str());
    if (!fin.is_open())
        return false;
    std::ostringstream ss;
    ss << fin.rdbuf();
    content = ss.str();
    return true;
}

bool parse_number_field(const std::string &object_str, const std::vector<std::string> &keys, double &value)
{
    for (size_t i = 0; i < keys.size(); ++i)
    {
        const std::string &key = keys[i];
        std::regex field_regex("\\\"" + key + "\\\"\\s*:\\s*([-+]?[0-9]*\\.?[0-9]+(?:[eE][-+]?[0-9]+)?)");
        std::smatch match;
        if (std::regex_search(object_str, match, field_regex))
        {
            value = std::stod(match[1].str());
            return true;
        }
    }
    return false;
}

bool parse_waypoints_from_json(const std::string &json_content, std::vector<JsonWaypoint> &waypoints)
{
    waypoints.clear();
    std::regex object_regex("\\{[^\\{\\}]*\\}");
    std::sregex_iterator it(json_content.begin(), json_content.end(), object_regex);
    std::sregex_iterator end;

    for (; it != end; ++it)
    {
        const std::string obj = it->str();
        JsonWaypoint wp;

        bool has_lat = parse_number_field(obj, {"latitude", "lat", "y"}, wp.latitude);
        bool has_lon = parse_number_field(obj, {"longitude", "lon", "lng", "x"}, wp.longitude);
        if (!has_lat || !has_lon)
            continue;

        if (!parse_number_field(obj, {"altitude", "alt", "z"}, wp.altitude))
            wp.altitude = 99.0;

        if (!parse_number_field(obj, {"qw", "w"}, wp.qw))
            wp.qw = 1.0;
        if (!parse_number_field(obj, {"qx"}, wp.qx))
            wp.qx = 0.0;
        if (!parse_number_field(obj, {"qy"}, wp.qy))
            wp.qy = 0.0;
        if (!parse_number_field(obj, {"qz"}, wp.qz))
            wp.qz = 0.0;

        waypoints.push_back(wp);
    }

    return !waypoints.empty();
}

void publish_car_model(double t, Eigen::Vector3d t_w_car, Eigen::Quaterniond q_w_car)
{
    visualization_msgs::MarkerArray markerArray_msg;
    visualization_msgs::Marker car_mesh;
    car_mesh.header.stamp = ros::Time(t);
    car_mesh.header.frame_id = "world";
    car_mesh.type = visualization_msgs::Marker::MESH_RESOURCE;
    car_mesh.action = visualization_msgs::Marker::ADD;
    car_mesh.id = 0;

    car_mesh.mesh_resource = "package://global_fusion/models/car.dae";

    Eigen::Matrix3d rot;
    rot << 0, 0, -1, 0, -1, 0, -1, 0, 0;
    
    Eigen::Quaterniond Q;
    Q = q_w_car * rot; 
    car_mesh.pose.position.x    = t_w_car.x();
    car_mesh.pose.position.y    = t_w_car.y();
    car_mesh.pose.position.z    = t_w_car.z();
    car_mesh.pose.orientation.w = Q.w();
    car_mesh.pose.orientation.x = Q.x();
    car_mesh.pose.orientation.y = Q.y();
    car_mesh.pose.orientation.z = Q.z();

    car_mesh.color.a = 1.0;
    car_mesh.color.r = 1.0;
    car_mesh.color.g = 0.0;
    car_mesh.color.b = 0.0;

    float major_scale = 2.0;

    car_mesh.scale.x = major_scale;
    car_mesh.scale.y = major_scale;
    car_mesh.scale.z = major_scale;
    markerArray_msg.markers.push_back(car_mesh);
    pub_car.publish(markerArray_msg);
}

void GPS_callback(const sensor_msgs::NavSatFixConstPtr &GPS_msg)
{
    //printf("gps_callback! \n");
    m_buf.lock();
    gpsQueue.push(GPS_msg);
    m_buf.unlock();
}

void convert_json_waypoints_to_utm(const std::vector<JsonWaypoint> &waypoints)
{
    printf("Received route with %lu waypoints\n", waypoints.size());
    
    // Create output file
    std::ofstream fout(route_output_path.c_str());
    if (!fout.is_open()) {
        printf("Failed to open output file!\n");
        return;
    }
    
    fout << "x,y,z,qw,qx,qy,qz\n";
    fout << std::fixed << std::setprecision(15);
    
    // Convert each waypoint
    for (size_t i = 0; i < waypoints.size(); ++i) {
        double latitude = waypoints[i].latitude;
        double longitude = waypoints[i].longitude;
        cout << "latitude: " << latitude << " longitude: " << longitude << std::endl;
        double altitude = 99.0; //route_msg->poses[i].pose.position.z;   // z contains altitude
        
        // Use the same conversion method as in globalOpt.cpp
        double xyz[3];
        globalEstimator.GPS2XYZ(latitude, longitude, altitude, xyz);
        
        // Get quaternion from pose
        double qw = waypoints[i].qw;
        double qx = waypoints[i].qx;
        double qy = waypoints[i].qy;
        double qz = waypoints[i].qz;
        
        // Write to file in the requested format: x,y,z,qw,qx,qy,qz
        fout << xyz[0] << "," << xyz[1] << "," << xyz[2] 
             << "," << qw << "," << qx << "," << qy << "," << qz << std::endl;
        
        printf("Waypoint %lu: Lat=%.6f, Lon=%.6f, Alt=%.2f -> UTM(%.2f, %.2f, %.2f)\n", 
               i+1, latitude, longitude, altitude, xyz[0], xyz[1], xyz[2]);
    }
    
    fout.close();
    printf("Route conversion completed. Results saved to %s\n", route_output_path.c_str());
}

void route_file_timer_callback(const ros::TimerEvent &)
{
    std::string current_content;
    if (!read_file_to_string(route_json_path, current_content))
    {
        ROS_WARN_THROTTLE(5.0, "Route JSON file not readable: %s", route_json_path.c_str());
        return;
    }

    if (current_content == last_route_json_content)
        return;

    std::vector<JsonWaypoint> waypoints;
    if (!parse_waypoints_from_json(current_content, waypoints))
    {
        ROS_WARN("Route JSON changed but no valid waypoints found in %s", route_json_path.c_str());
        last_route_json_content = current_content;
        return;
    }

    ROS_INFO("Detected route JSON update. Reloading %lu waypoints from %s",
             waypoints.size(), route_json_path.c_str());
    convert_json_waypoints_to_utm(waypoints);
    last_route_json_content = current_content;
}

void vio_callback(const nav_msgs::Odometry::ConstPtr &pose_msg)
{
    //printf("vio_callback! \n");
    double t = pose_msg->header.stamp.toSec();
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
        sensor_msgs::NavSatFixConstPtr GPS_msg = gpsQueue.front();
        double gps_t = GPS_msg->header.stamp.toSec();
        printf("vio t: %f, gps t: %f \n", t, gps_t);
        // 10ms sync tolerance
        if(gps_t >= t - 0.01 && gps_t <= t + 0.01)
        {
            //printf("receive GPS with timestamp %f\n", GPS_msg->header.stamp.toSec());
            double latitude = GPS_msg->latitude;
            double longitude = GPS_msg->longitude;
            double altitude = GPS_msg->altitude;
            //int numSats = GPS_msg->status.service;
            double pos_accuracy = GPS_msg->position_covariance[0];
            if(pos_accuracy <= 0)
                pos_accuracy = 1;
            //printf("receive covariance %lf \n", pos_accuracy);
            //if(GPS_msg->status.status > 8)
                globalEstimator.inputGPS(t, latitude, longitude, altitude, pos_accuracy);
            gpsQueue.pop();
            break;
        }
        else if(gps_t < t - 0.01)
            gpsQueue.pop();
        else if(gps_t > t + 0.01)
            break;
    }
    m_buf.unlock();

    Eigen::Vector3d global_t;
    Eigen:: Quaterniond global_q;
    globalEstimator.getGlobalOdom(global_t, global_q);

    nav_msgs::Odometry odometry;
    odometry.header = pose_msg->header;
    odometry.header.frame_id = "world";
    odometry.child_frame_id = "world";
    odometry.pose.pose.position.x = global_t.x();
    odometry.pose.pose.position.y = global_t.y();
    odometry.pose.pose.position.z = global_t.z();
    odometry.pose.pose.orientation.x = global_q.x();
    odometry.pose.pose.orientation.y = global_q.y();
    odometry.pose.pose.orientation.z = global_q.z();
    odometry.pose.pose.orientation.w = global_q.w();
    pub_global_odometry.publish(odometry);
    // printf("pub_global_odometry");

    // printf("pub_global_path");
    // pub_global_path.publish(*global_path);
  //  publish_car_model(t, global_t, global_q);


    // write result to file
    std::ofstream foutC("/home/tony-ws1/output/vio_global.csv", ios::app);
    foutC.setf(ios::fixed, ios::floatfield);
    foutC.precision(0);
    foutC << pose_msg->header.stamp.toSec() * 1e9 << ",";
    foutC.precision(5);
    foutC << global_t.x() << ","
            << global_t.y() << ","
            << global_t.z() << ","
            << global_q.w() << ","
            << global_q.x() << ","
            << global_q.y() << ","
            << global_q.z() << endl;
    foutC.close();
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "globalEstimator");
    ros::NodeHandle n("~");

    global_path = &globalEstimator.global_path;

    n.param<std::string>("route_json_path", route_json_path,
                         "/home/nuvo/SR/foundation-policy/Policy_part/topomaps/LLM_Policy/route_waypoints.json");
    n.param<std::string>("route_output_path", route_output_path,
                         "/home/nuvo/SR/foundation-policy/Policy_part/topomaps/LLM_UTM_odom_poses.txt");

    ros::Subscriber sub_GPS = n.subscribe("/acu_driver/gps_nav_topic", 100, GPS_callback);
    ros::Subscriber sub_vio = n.subscribe("/odom", 100, vio_callback);
    ros::Timer route_file_timer = n.createTimer(ros::Duration(1.0), route_file_timer_callback);
    pub_global_path = n.advertise<nav_msgs::Path>("global_path", 100);
    pub_global_odometry = n.advertise<nav_msgs::Odometry>("global_odometry", 100);
    pub_car = n.advertise<visualization_msgs::MarkerArray>("car_model", 1000);

    ROS_INFO("Watching route JSON file: %s", route_json_path.c_str());
    ROS_INFO("Route UTM output file: %s", route_output_path.c_str());
    ros::spin();
    return 0;
}
