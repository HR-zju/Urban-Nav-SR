#pragma once
#include <rclcpp/rclcpp.hpp>
#include <Eigen/Dense>

#include <unordered_map>
#include <vector>
#include <cmath>
#include <string>
#include <memory>
#include <mutex>
#include <deque>

#include <sensor_msgs/msg/nav_sat_fix.hpp>
#include <gnss_comm/msg/gnss_ephem_msg.hpp>
#include <gnss_comm/msg/gnss_meas_msg.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <system_ctrl/msg/algo_status.hpp>

#include "gnss_comm_impl/gnss_ros.hpp"
#include "gnss_comm_impl/gnss_spp.hpp"
#include "gnss_comm_impl/gnss_utility.hpp"
#include "gnss_comm_impl/kml_exporter.hpp"

class GnssSppNode : public rclcpp::Node
{
public:
    GnssSppNode() : Node("gnss_spp_node")
    {
        kml_exporter_ = std::make_unique<gnss_comm::KmlExporter>("/home/symc/data/kml");
        kml_exporter_raw_ = std::make_unique<gnss_comm::KmlExporter>("/home/symc/data/kml_raw");
        sub_ephem_ = this->create_subscription<gnss_comm::msg::GnssEphemMsg>(
            "/baton/gnss/ephem", 10,
            std::bind(&GnssSppNode::ephem_callback, this, std::placeholders::_1));

        sub_meas_ = this->create_subscription<gnss_comm::msg::GnssMeasMsg>(
            "/baton/gnss/meas", 10,
            std::bind(&GnssSppNode::meas_callback, this, std::placeholders::_1));
        
        sub_odom_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/baton/stereo3/odometry", 10,
            std::bind(&GnssSppNode::odom_callback, this, std::placeholders::_1));

        sub_algo_status_ = this->create_subscription<system_ctrl::msg::AlgoStatus>(
            "/baton/algo_status", 10,
            std::bind(&GnssSppNode::algo_status_callback, this, std::placeholders::_1));

        pub_spp_fix_raw_ = this->create_publisher<sensor_msgs::msg::NavSatFix>(
            "/baton/gnss/spp_fix_raw", 10);
        
        pub_spp_fix_ = this->create_publisher<sensor_msgs::msg::NavSatFix>(
            "/baton/gnss/spp_fix", 10);
        
        // 创建定时器，每隔一小时执行一次清理
        cleanup_timer_ = this->create_wall_timer(
            std::chrono::seconds(CLEANUP_INTERVAL_SEC),
            std::bind(&GnssSppNode::cleanup_ephem_callback, this));
        
        RCLCPP_INFO(this->get_logger(), "GNSS SPP Node initialized. Waiting for data...");
    }

private:

    static constexpr double MAX_EPHEM_AGE_SEC = 7200.0; // 默认星历有效期
    static constexpr double MIN_EARTH_RADIUS  = 6.0e6;
    static constexpr double MAX_EARTH_RADIUS  = 7.0e6;
    static constexpr size_t MIN_SAT_COUNT     = 4;      // SPP 至少需要4颗星
    static constexpr int CLEANUP_INTERVAL_SEC = 3600;       // 清理执行频率：1小时一次
    static constexpr int MAX_HATCH_WINDOW     = 100;    // Hatch滤波最大平滑窗口(历元数)

    void ephem_callback(const gnss_comm::msg::GnssEphemMsg::ConstSharedPtr msg);
    void meas_callback(const gnss_comm::msg::GnssMeasMsg::ConstSharedPtr msg);
    void odom_callback(const nav_msgs::msg::Odometry::ConstSharedPtr msg);
    void algo_status_callback(const system_ctrl::msg::AlgoStatus::ConstSharedPtr msg);
    
private:
    void filter_iono_free(const std::vector<gnss_comm::ObsPtr>& obs_in, const std::vector<gnss_comm::EphemBasePtr>& ephem_in, std::vector<gnss_comm::ObsPtr>& obs_out, std::vector<gnss_comm::EphemBasePtr>& ephem_out);
    void filter_L1_hatch(const std::vector<gnss_comm::ObsPtr>& obs_in, const std::vector<gnss_comm::EphemBasePtr>& ephem_in, std::vector<gnss_comm::ObsPtr>& obs_out, std::vector<gnss_comm::EphemBasePtr>& ephem_out);
    void filter_weak_constellations(const std::vector<gnss_comm::ObsPtr>& obs_in, const std::vector<gnss_comm::EphemBasePtr>& ephem_in, std::vector<gnss_comm::ObsPtr>& obs_out, std::vector<gnss_comm::EphemBasePtr>& ephem_out, const int min_sat_per_sys = 3);
    bool is_valid_ephem_msg(const gnss_comm::msg::GnssEphemMsg &msg, std::string *reason);
    bool is_valid_obs_msg(const gnss_comm::msg::GnssObsMsg &msg, std::string *reason);
    void cleanup_ephem_callback();

    rclcpp::Subscription<gnss_comm::msg::GnssEphemMsg>::SharedPtr sub_ephem_;
    rclcpp::Subscription<gnss_comm::msg::GnssMeasMsg>::SharedPtr  sub_meas_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr      sub_odom_;
    rclcpp::Subscription<system_ctrl::msg::AlgoStatus>::SharedPtr sub_algo_status_;
    
    rclcpp::Publisher<sensor_msgs::msg::NavSatFix>::SharedPtr     pub_spp_fix_raw_;
    rclcpp::Publisher<sensor_msgs::msg::NavSatFix>::SharedPtr     pub_spp_fix_;

    rclcpp::TimerBase::SharedPtr cleanup_timer_;
    std::mutex ephem_mutex_; // 保护 ephem_map_
    std::unordered_map<uint32_t, gnss_comm::EphemBasePtr> ephem_map_;

    struct HatchState {
        double p_smooth = 0.0;
        double cp_prev = 0.0;
        int count = 0;
        double last_time = 0.0;
    };
    std::unordered_map<uint32_t, HatchState> hatch_states_;

    std::unique_ptr<gnss_comm::KmlExporter> kml_exporter_;
    std::unique_ptr<gnss_comm::KmlExporter> kml_exporter_raw_;

    // 融合Stereo3和SPP
    std::deque<Eigen::Vector3d> init_ecef_buffer_; // 滑动窗口缓存
    static constexpr size_t INIT_WINDOW_SIZE = 10; // 需要多少帧连续稳定的数据
    static constexpr double STABILITY_THRESHOLD = 1.0; // 稳定阈值

    bool is_algo_running_ = false;
    bool is_origin_set_   = false;
    Eigen::Vector3d origin_ecef_{0.0, 0.0, 0.0};
    Eigen::Vector3d origin_lla_{0.0, 0.0, 0.0};
    Eigen::Matrix3d R_enu2ecef_{Eigen::Matrix3d::Identity()};
};
