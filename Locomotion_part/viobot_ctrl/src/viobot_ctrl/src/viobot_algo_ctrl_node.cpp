#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/header.hpp>
#include <system_ctrl/msg/algo_ctrl.hpp>
#include <system_ctrl/msg/algo_status.hpp>
#include <iostream>
#include <string>
#include <thread>
#include <mutex>

enum class CmdType {
    EXIT = 0,
    ENABLE = 1,
    DISABLE = 2,
    REBOOT = 3,
    RESET = 4
};

class AlgoCtrlNode : public rclcpp::Node {
public:
    AlgoCtrlNode() : Node("algo_ctrl_node") {
        pub_stereo3_ctrl_ = this->create_publisher<system_ctrl::msg::AlgoCtrl>("/baton/stereo3_ctrl", 2);
        sub_algo_status_ = this->create_subscription<system_ctrl::msg::AlgoStatus>(
            "/baton/algo_status", 2,
            std::bind(&AlgoCtrlNode::algo_status_callback, this, std::placeholders::_1));
    }

    // 1. 全量绘制整个 UI 界面（通常在初始化或用户回车后调用）
    void draw_full_ui() {
        std::lock_guard<std::mutex> lock(ui_mutex_);
        
        // \033[2J: 清空屏幕, \033[H: 将光标移动到左上角 (1行1列)
        std::cout << "\033[2J\033[H";
        
        // 第 1-3 行：顶部固定状态栏
        std::cout << "====================================================\n";
        std::cout << " [ STATUS ] Algo Status: " << current_status_ << "\n";
        std::cout << "====================================================\n";
        
        // 第 4-8 行：菜单区
        std::cout << " Algo Control Menu:\n"
                  << " [1] Enable  [2] Disable\n"
                  << " [3] Reboot  [4] Reset\n"
                  << " [0] Exit\n"
                  << "----------------------------------------------------\n";
        
        // 第 9-10 行：操作反馈区
        std::cout << " [ ACTION ] " << last_action_msg_ << "\n"
                  << "----------------------------------------------------\n";
        
        // 第 11 行：输入提示符
        std::cout << "Enter command: " << std::flush;
    }

    // 2. 局部刷新顶部状态栏
    void update_status_bar(const std::string& new_status) {
        std::lock_guard<std::mutex> lock(ui_mutex_);
        if (current_status_ == new_status) return; // 状态未变则不刷新
        current_status_ = new_status;

        // ANSI：
        // \033[s      : 保存当前光标位置 (保护用户正在输入的区域)
        // \033[2;1H   : 将光标移动到第 2 行，第 1 列
        // \033[2K     : 清空第 2 行的内容
        // \033[u      : 恢复刚才保存的光标位置
        std::cout << "\033[s" 
                  << "\033[2;1H\033[2K" 
                  << " [ STATUS ] Algo Status: " << current_status_ 
                  << "\033[u" 
                  << std::flush;
    }

    void set_action_msg(const std::string& msg) {
        last_action_msg_ = msg;
        // 动作更新后，我们重新绘制整个UI以保持整洁
        draw_full_ui();
    }

    void publish_cmd(int v) {
        system_ctrl::msg::AlgoCtrl algo_set;
        algo_set.algo_enable = false;
        algo_set.algo_reboot = false;
        algo_set.algo_reset = false;

        switch (static_cast<CmdType>(v)) {
            case CmdType::ENABLE:
                set_action_msg("Command Sent: ALGO_ENABLE");
                algo_set.algo_enable = true;
                break;
            case CmdType::DISABLE:
                set_action_msg("Command Sent: ALGO_DISABLE");
                algo_set.algo_enable = false;
                break;
            case CmdType::REBOOT:
                set_action_msg("Command Sent: ALGO_REBOOT");
                algo_set.algo_reboot = true;
                break;
            case CmdType::RESET:
                set_action_msg("Command Sent: ALGO_RESET");
                algo_set.algo_reset = true;
                break;
            default:
                set_action_msg("Error: Unknown command (" + std::to_string(v) + ")");
                return; 
        }

        pub_stereo3_ctrl_->publish(algo_set);
    }

private:
    void algo_status_callback(const system_ctrl::msg::AlgoStatus::SharedPtr msg) {
        // 后台收到状态时，仅局部更新第 2 行
        update_status_bar(msg->algo_status);
    }

    rclcpp::Publisher<system_ctrl::msg::AlgoCtrl>::SharedPtr pub_stereo3_ctrl_;
    rclcpp::Subscription<system_ctrl::msg::AlgoStatus>::SharedPtr sub_algo_status_;

    std::mutex ui_mutex_; // 防止主线程和 ROS 回调线程同时写控制台导致错乱
    std::string current_status_ = "WAITING FOR DATA...";
    std::string last_action_msg_ = "Ready.";
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<AlgoCtrlNode>();

    std::thread spin_thread([&node]() {
        rclcpp::spin(node);
    });

    std::string input_str;
    
    // 初始化清屏并绘制 UI
    node->draw_full_ui();

    while (rclcpp::ok() && std::getline(std::cin, input_str)) {
        if (input_str.empty()) {
            node->draw_full_ui(); // 压了回车但没输入，重新刷屏保持整洁
            continue;
        }

        try {
            int v = std::stoi(input_str);
            if (v == static_cast<int>(CmdType::EXIT)) {
                node->set_action_msg("Exiting control node...");
                break;
            }
            node->publish_cmd(v);
        } catch (const std::exception& e) {
            node->set_action_msg("Invalid input! Please enter a number.");
        }
    }

    // 退出前清空一下屏幕
    std::cout << "\033[2J\033[H" << "Node shutdown gracefully.\n";

    rclcpp::shutdown();
    if (spin_thread.joinable()) {
        spin_thread.join();
    }
    
    return 0;
}