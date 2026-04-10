#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/nav_sat_fix.hpp>
#include <system_ctrl/msg/algo_ctrl.hpp>

class Stereo3AutoStartNode : public rclcpp::Node
{
public:
    Stereo3AutoStartNode() : Node("stereo3_auto_start_node")
    {
        fix_topic_ = this->declare_parameter<std::string>("gnss_fix_topic", "/baton/gnss/spp_fix_raw");
        ctrl_topic_ = this->declare_parameter<std::string>("stereo3_ctrl_topic", "/baton/stereo3_ctrl");

        pub_ctrl_ = this->create_publisher<system_ctrl::msg::AlgoCtrl>(ctrl_topic_, 2);
        sub_fix_ = this->create_subscription<sensor_msgs::msg::NavSatFix>(
            fix_topic_, 10,
            std::bind(&Stereo3AutoStartNode::fix_callback, this, std::placeholders::_1));

        RCLCPP_INFO(this->get_logger(), "Waiting for GNSS fix on %s before enabling stereo3...", fix_topic_.c_str());
    }

private:
    void fix_callback(const sensor_msgs::msg::NavSatFix::ConstSharedPtr)
    {
        if (sent_) {
            return;
        }

        system_ctrl::msg::AlgoCtrl cmd;
        cmd.algo_enable = true;
        cmd.algo_reboot = false;
        cmd.algo_reset = false;

        pub_ctrl_->publish(cmd);
        sent_ = true;

        RCLCPP_INFO(this->get_logger(), "GNSS fix received, stereo3 enable command published.");
        rclcpp::shutdown();
    }

    std::string fix_topic_;
    std::string ctrl_topic_;
    rclcpp::Publisher<system_ctrl::msg::AlgoCtrl>::SharedPtr pub_ctrl_;
    rclcpp::Subscription<sensor_msgs::msg::NavSatFix>::SharedPtr sub_fix_;
    bool sent_ = false;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<Stereo3AutoStartNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}