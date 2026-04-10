#pragma once
#include <rclcpp/rclcpp.hpp>
#include <Eigen/Dense>

#include <unordered_map>
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>
#include <memory>
#include <mutex>
#include <deque>

#include <sensor_msgs/msg/nav_sat_fix.hpp>
#include <std_msgs/msg/float64.hpp>
#include <std_msgs/msg/string.hpp>
#include <gnss_comm/msg/gnss_ephem_msg.hpp>
#include <gnss_comm/msg/gnss_meas_msg.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <system_ctrl/msg/algo_status.hpp>

#include "gnss_comm_impl/gnss_ros.hpp"
#include "gnss_comm_impl/gnss_spp.hpp"
#include "gnss_comm_impl/gnss_utility.hpp"
#include "gnss_comm_impl/kml_exporter.hpp"
#include "gnss_comm_impl/stereo_yaw_bridge.hpp"

class GnssSppNode : public rclcpp::Node
{
public:
    enum class GnssStatusState {
        Idle,
        WaitingOrigin,
        Fusing,
    };

    GnssSppNode() : Node("gnss_spp_node")
    {
        const bool enable_kml_export = this->declare_parameter<bool>("enable_kml_export", true);
        const std::string kml_directory = this->declare_parameter<std::string>("kml_directory", "/home/symc/data/kml");
        const std::string kml_raw_directory = this->declare_parameter<std::string>("kml_raw_directory", "/home/symc/data/kml_raw");

        const int yaw_min_samples = this->declare_parameter<int>("yaw_bridge.min_samples", 5);
        const double yaw_min_baseline_m = this->declare_parameter<double>("yaw_bridge.min_baseline_m", 2.0);
        const double yaw_gnss_stale_sec = this->declare_parameter<double>("yaw_bridge.gnss_stale_sec", 1.5);
        const double yaw_update_gain = this->declare_parameter<double>("yaw_bridge.update_gain", 0.03);
        const double yaw_max_step_deg = this->declare_parameter<double>("yaw_bridge.max_step_deg", 2.0);
        const double yaw_max_innovation_deg = this->declare_parameter<double>("yaw_bridge.max_innovation_deg", 25.0);
        const std::vector<double> gnss_initial_calibration_enu = this->declare_parameter<std::vector<double>>("gnss.initial_calibration_enu", std::vector<double>{0.0, 0.0, 0.0});

        enable_bridge_diagnostics_ = this->declare_parameter<bool>("diagnostics.enable_bridge", false);
        bridge_diag_topic_ = this->declare_parameter<std::string>("diagnostics.bridge_topic", "/baton/gnss/bridge_diagnostics");

        if (gnss_initial_calibration_enu.size() == 3) {
            gnss_initial_calibration_enu_ = Eigen::Vector3d(gnss_initial_calibration_enu[0], gnss_initial_calibration_enu[1], gnss_initial_calibration_enu[2]);
        } else {
            RCLCPP_WARN(this->get_logger(), "gnss.initial_calibration_enu must have 3 elements, falling back to [0, 0, 0]");
            gnss_initial_calibration_enu_.setZero();
        }


        gnss_comm_impl::StereoYawBridge::Config yaw_cfg;
        yaw_cfg.min_samples = std::max(1, yaw_min_samples);
        yaw_cfg.min_baseline_m = std::max(0.1, yaw_min_baseline_m);
        yaw_cfg.gnss_stale_sec = std::max(0.1, yaw_gnss_stale_sec);
        yaw_cfg.update_gain = std::max(0.001, yaw_update_gain);
        yaw_cfg.max_step_rad = std::max(0.1, yaw_max_step_deg) * gnss_comm_impl::StereoYawBridge::PI / 180.0;
        yaw_cfg.max_innovation_rad = std::max(1.0, yaw_max_innovation_deg) * gnss_comm_impl::StereoYawBridge::PI / 180.0;
        yaw_bridge_ = gnss_comm_impl::StereoYawBridge(yaw_cfg);

        if (enable_kml_export) {
            kml_exporter_ = std::make_unique<gnss_comm::KmlExporter>(kml_directory);
            kml_exporter_raw_ = std::make_unique<gnss_comm::KmlExporter>(kml_raw_directory);
        }

        if (enable_bridge_diagnostics_) {
            pub_bridge_diag_ = this->create_publisher<std_msgs::msg::String>(bridge_diag_topic_, 10);
        }

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

        pub_gnss_status_ = this->create_publisher<system_ctrl::msg::AlgoStatus>(
            "/baton/gnss/status", 10);

        pub_spp_fix_raw_ = this->create_publisher<sensor_msgs::msg::NavSatFix>(
            "/baton/gnss/spp_fix_raw", 10);
        
        pub_spp_fix_ = this->create_publisher<sensor_msgs::msg::NavSatFix>(
            "/baton/gnss/spp_fix", 10);

        
        pub_heading_ = this->create_publisher<std_msgs::msg::Float64>(
            "/baton/gnss/heading", 10);
        
        // 创建定时器，每隔一小时执行一次清理
        cleanup_timer_ = this->create_wall_timer(
            std::chrono::seconds(CLEANUP_INTERVAL_SEC),
            std::bind(&GnssSppNode::cleanup_ephem_callback, this));

        gnss_status_timer_ = this->create_wall_timer(
            std::chrono::seconds(1),
            std::bind(&GnssSppNode::publish_gnss_status, this));
        
        RCLCPP_INFO(this->get_logger(), "GNSS SPP Node initialized. Waiting for data...");
        publish_gnss_status();
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
    void transition_gnss_status(GnssStatusState next_state);
    void publish_gnss_status();
    static const char *gnss_status_to_string(GnssStatusState state);
    void publish_bridge_diagnostics(const gnss_comm_impl::StereoYawBridge::UpdateResult &result);
    
private:
    void filter_iono_free(const std::vector<gnss_comm::ObsPtr>& obs_in, const std::vector<gnss_comm::EphemBasePtr>& ephem_in, std::vector<gnss_comm::ObsPtr>& obs_out, std::vector<gnss_comm::EphemBasePtr>& ephem_out);
    void filter_L1_hatch(const std::vector<gnss_comm::ObsPtr>& obs_in, const std::vector<gnss_comm::EphemBasePtr>& ephem_in, std::vector<gnss_comm::ObsPtr>& obs_out, std::vector<gnss_comm::EphemBasePtr>& ephem_out);
    void filter_weak_constellations(const std::vector<gnss_comm::ObsPtr>& obs_in, const std::vector<gnss_comm::EphemBasePtr>& ephem_in, std::vector<gnss_comm::ObsPtr>& obs_out, std::vector<gnss_comm::EphemBasePtr>& ephem_out, const int min_sat_per_sys = 3);
    bool is_valid_ephem_msg(const gnss_comm::msg::GnssEphemMsg &msg, std::string *reason);
    bool is_valid_obs_msg(const gnss_comm::msg::GnssObsMsg &msg, std::string *reason);
    void cleanup_ephem_callback();
    Eigen::Vector3d apply_gnss_initial_calibration(const Eigen::Vector3d &ecef_pos) const;

    rclcpp::Subscription<gnss_comm::msg::GnssEphemMsg>::SharedPtr sub_ephem_;
    rclcpp::Subscription<gnss_comm::msg::GnssMeasMsg>::SharedPtr  sub_meas_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr      sub_odom_;
    rclcpp::Subscription<system_ctrl::msg::AlgoStatus>::SharedPtr sub_algo_status_;
    
    rclcpp::Publisher<sensor_msgs::msg::NavSatFix>::SharedPtr     pub_spp_fix_raw_;
    rclcpp::Publisher<sensor_msgs::msg::NavSatFix>::SharedPtr     pub_spp_fix_;
    rclcpp::Publisher<system_ctrl::msg::AlgoStatus>::SharedPtr    pub_gnss_status_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr           pub_bridge_diag_;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr          pub_heading_;

    rclcpp::TimerBase::SharedPtr cleanup_timer_;
    rclcpp::TimerBase::SharedPtr gnss_status_timer_;
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
    GnssStatusState current_gnss_status_ = GnssStatusState::Idle;
    gnss_comm_impl::StereoYawBridge yaw_bridge_;
    bool enable_bridge_diagnostics_ = false;
    std::string bridge_diag_topic_;
    Eigen::Vector3d gnss_initial_calibration_enu_{0.0, 0.0, 0.0};

    Eigen::Vector3d origin_ecef_{0.0, 0.0, 0.0};
    Eigen::Vector3d origin_lla_{0.0, 0.0, 0.0};
    Eigen::Matrix3d R_enu2ecef_{Eigen::Matrix3d::Identity()};
};
