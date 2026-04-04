#include <rclcpp/rclcpp.hpp>
#include <Eigen/Dense>

#include <unordered_map>
#include <vector>
#include <cmath>
#include <string>
#include <memory>
#include <mutex>

#include <sensor_msgs/msg/nav_sat_fix.hpp>
#include <gnss_comm/msg/gnss_ephem_msg.hpp>
#include <gnss_comm/msg/gnss_meas_msg.hpp>

#include "gnss_comm_impl/gnss_ros.hpp"
#include "gnss_comm_impl/gnss_spp.hpp"
#include "gnss_comm_impl/gnss_utility.hpp"

class GnssSppNode : public rclcpp::Node
{
public:
    GnssSppNode() : Node("gnss_spp_node")
    {
        sub_ephem_ = this->create_subscription<gnss_comm::msg::GnssEphemMsg>(
            "/baton/gnss/ephem", 10,
            std::bind(&GnssSppNode::ephem_callback, this, std::placeholders::_1));

        sub_meas_ = this->create_subscription<gnss_comm::msg::GnssMeasMsg>(
            "/baton/gnss/meas", 10,
            std::bind(&GnssSppNode::meas_callback, this, std::placeholders::_1));
        
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

    void ephem_callback(const gnss_comm::msg::GnssEphemMsg::ConstSharedPtr msg);
    void meas_callback(const gnss_comm::msg::GnssMeasMsg::ConstSharedPtr msg);
    
private:
    void filter_iono_free(const std::vector<gnss_comm::ObsPtr>& obs_in, const std::vector<gnss_comm::EphemBasePtr>& ephem_in, std::vector<gnss_comm::ObsPtr>& obs_out, std::vector<gnss_comm::EphemBasePtr>& ephem_out);
    void filter_weak_constellations(const std::vector<gnss_comm::ObsPtr>& obs_in, const std::vector<gnss_comm::EphemBasePtr>& ephem_in, std::vector<gnss_comm::ObsPtr>& obs_out, std::vector<gnss_comm::EphemBasePtr>& ephem_out, const int min_sat_per_sys = 3);
    bool is_valid_ephem_msg(const gnss_comm::msg::GnssEphemMsg &msg, std::string *reason);
    bool is_valid_obs_msg(const gnss_comm::msg::GnssObsMsg &msg, std::string *reason);
    void cleanup_ephem_callback();

    rclcpp::Subscription<gnss_comm::msg::GnssEphemMsg>::SharedPtr sub_ephem_;
    rclcpp::Subscription<gnss_comm::msg::GnssMeasMsg>::SharedPtr  sub_meas_;
    rclcpp::Publisher<sensor_msgs::msg::NavSatFix>::SharedPtr     pub_spp_fix_;

    rclcpp::TimerBase::SharedPtr cleanup_timer_;
    std::mutex ephem_mutex_; // 保护 ephem_map_
    std::unordered_map<uint32_t, gnss_comm::EphemBasePtr> ephem_map_;
};
