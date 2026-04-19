#include "gnss_comm_impl/app_node.hpp"

#include <sstream>

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<GnssSppNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}

const char *GnssSppNode::gnss_status_to_string(GnssStatusState state)
{
    switch (state) {
        case GnssStatusState::Idle:
            return "gnss_idle";
        case GnssStatusState::WaitingOrigin:
            return "gnss_waiting_origin";
        case GnssStatusState::Fusing:
            return "gnss_fusing";
    }

    return "gnss_idle";
}

void GnssSppNode::publish_gnss_status()
{
    if (!pub_gnss_status_) {
        return;
    }

    system_ctrl::msg::AlgoStatus msg;
    msg.header.stamp = this->now();
    msg.header.frame_id = "gnss_spp_node";
    msg.algo_status = gnss_status_to_string(current_gnss_status_);
    pub_gnss_status_->publish(msg);
}

void GnssSppNode::transition_gnss_status(GnssStatusState next_state)
{
    if (!pub_gnss_status_) {
        return;
    }

    if (current_gnss_status_ == next_state) {
        return;
    }

    current_gnss_status_ = next_state;

    publish_gnss_status();
}

void GnssSppNode::publish_bridge_diagnostics(const gnss_comm_impl::StereoYawBridge::UpdateResult &result)
{
    if (!enable_bridge_diagnostics_ || !pub_bridge_diag_) return;

    std_msgs::msg::String msg;
    std::ostringstream oss;
    oss << "calibrated=" << (result.calibrated ? 1 : 0)
        << ",samples=" << result.calib_samples
        << ",gnss_fresh=" << (result.gnss_fresh ? 1 : 0)
        << ",used_new_gnss=" << (result.used_new_gnss ? 1 : 0)
        << ",has_measurement=" << (result.has_measurement ? 1 : 0)
        << ",reject_count=" << result.reject_count
        << ",yaw_deg=" << result.yaw_offset_rad * 180.0 / M_PI;
    msg.data = oss.str();
    pub_bridge_diag_->publish(msg);
}

Eigen::Vector3d GnssSppNode::apply_gnss_initial_calibration(const Eigen::Vector3d &ecef_pos) const
{
    if (gnss_initial_calibration_enu_.isZero(0.0)) {
        return ecef_pos;
    }

    const Eigen::Vector3d geo_pos = gnss_comm::ecef2geo(ecef_pos);
    const Eigen::Matrix3d R_enu2ecef = gnss_comm::geo2rotation(geo_pos);
    return ecef_pos + R_enu2ecef * gnss_initial_calibration_enu_;
}

void GnssSppNode::ephem_callback(const gnss_comm::msg::GnssEphemMsg::ConstSharedPtr msg)
{
    auto clock = this->get_clock();
    auto logger = this->get_logger();
    std::string reject_reason;

    if (!this->is_valid_ephem_msg(*msg, &reject_reason))
    {
        RCLCPP_WARN_THROTTLE(logger, *clock, 2000,
                                "Rejected invalid ephem msg (sat=%u): %s",
                                msg->sat, reject_reason.c_str());
        return;
    }

    gnss_comm::EphemBasePtr ephem_ptr = gnss_comm::msg2ephem(msg);
    if (!ephem_ptr)
    {
        RCLCPP_WARN_THROTTLE(logger, *clock, 2000,
                                "msg2ephem returned null (sat=%u)", msg->sat);
        return;
    }

    // 写入星历
    std::lock_guard<std::mutex> lock(ephem_mutex_);
    ephem_map_[ephem_ptr->sat] = ephem_ptr;
}

void GnssSppNode::meas_callback(const gnss_comm::msg::GnssMeasMsg::ConstSharedPtr msg)
{
    auto clock = this->get_clock();
    auto logger = this->get_logger();

    if (msg->meas.empty())
    {
        RCLCPP_WARN_THROTTLE(logger, *clock, 2000, "Empty measurement epoch");
        return;
    }

    std::vector<gnss_comm::ObsPtr> obs_vec;
    obs_vec.reserve(msg->meas.size());

    for (const auto &obs_msg : msg->meas)
    {
        // 校验是否有效
        std::string reject_reason;
        if (!this->is_valid_obs_msg(obs_msg, &reject_reason))
        {
            RCLCPP_DEBUG(logger,
                "Filtered invalid obs (sat=%u): %s", obs_msg.sat, reject_reason.c_str());
            continue;
        }

        auto obs = std::make_shared<gnss_comm::Obs>();
        obs->time     = gnss_comm::gpst2time(obs_msg.time.week, obs_msg.time.tow);
        obs->sat      = obs_msg.sat;
        obs->freqs    = obs_msg.freqs;
        obs->CN0      = obs_msg.cn0;
        obs->LLI      = obs_msg.lli;
        obs->code     = obs_msg.code;
        obs->psr      = obs_msg.psr;
        obs->psr_std  = obs_msg.psr_std;
        obs->cp       = obs_msg.cp;
        obs->cp_std   = obs_msg.cp_std;
        obs->dopp     = obs_msg.dopp;
        obs->dopp_std = obs_msg.dopp_std;
        obs->status   = obs_msg.status;

        obs_vec.push_back(std::move(obs));
    }

    if (obs_vec.empty()) return;

    std::vector<gnss_comm::ObsPtr> matched_obs_in;
    std::vector<gnss_comm::EphemBasePtr> matched_ephems_in;
    matched_obs_in.reserve(obs_vec.size());
    matched_ephems_in.reserve(obs_vec.size());

    {
        // 读取星历
        std::lock_guard<std::mutex> lock(ephem_mutex_);
        for (const auto &obs : obs_vec)
        {
            auto it = ephem_map_.find(obs->sat);
            if (it == ephem_map_.end()) continue;

            const auto &ephem = it->second;

            if (ephem->health != 0) continue;

            // 防止直接相加导致的浮点数大数吃小数精度丢失问题
            double dt = (obs->time.time - ephem->toe.time) + 
                        (obs->time.sec - ephem->toe.sec);

            if (std::abs(dt) > MAX_EPHEM_AGE_SEC) continue;

            matched_obs_in.push_back(obs);
            matched_ephems_in.push_back(ephem);
        }
    } // 离开作用域，解锁 mutex

    if (matched_obs_in.empty())
    {
        RCLCPP_WARN_THROTTLE(logger, *clock, 2000, "No matched obs/ephem pairs");
        return;
    }

    // 尝试提取无电离层组合，如果失败则降级为L1单频
    std::vector<gnss_comm::ObsPtr> combined_obs;
    std::vector<gnss_comm::EphemBasePtr> combined_ephems;

    // filter_iono_free(matched_obs_in, matched_ephems_in, combined_obs, combined_ephems);

    // if (combined_obs.size() >= MIN_SAT_COUNT) {
    //     RCLCPP_DEBUG(logger, "Using IF Combination. Sats: %zu", combined_obs.size());
    // } else {
        // RCLCPP_WARN_THROTTLE(logger, *clock, 3000, 
        //     "Not enough dual-freq sats (%zu). Fallback to L1.", combined_obs.size());
        filter_L1_hatch(matched_obs_in, matched_ephems_in, combined_obs, combined_ephems);
    // }

    // 进一步剔除弱势星座防止 SPP 解算发散
    std::vector<gnss_comm::ObsPtr> robust_obs;
    std::vector<gnss_comm::EphemBasePtr> robust_ephems;
    filter_weak_constellations(combined_obs, combined_ephems, robust_obs, robust_ephems, 4);


    if (robust_obs.size() < MIN_SAT_COUNT)
    {
        RCLCPP_WARN_THROTTLE(logger, *clock, 2000,
                                "Not enough valid satellites (need >= %zu), current: %zu",
                                MIN_SAT_COUNT, robust_obs.size());
        return;
    }

    // TODO: 获取并填入真实的电离层参数
    std::vector<double> iono_params(8, 0.0); 

    Eigen::Matrix<double, 7, 1> spp_result =
        gnss_comm::psr_pos(robust_obs, robust_ephems, iono_params);

    Eigen::Vector3d ecef_pos = spp_result.head<3>();

    const Eigen::Vector3d corrected_ecef_pos = apply_gnss_initial_calibration(ecef_pos);

    if (!corrected_ecef_pos.allFinite())
    {
        RCLCPP_WARN_THROTTLE(logger, *clock, 2000, "SPP returned non-finite ECEF position");
        return;
    }

    const double ecef_norm = corrected_ecef_pos.norm();
    if (ecef_norm < MIN_EARTH_RADIUS || ecef_norm > MAX_EARTH_RADIUS)
    {
        RCLCPP_WARN_THROTTLE(logger, *clock, 2000,
                                "SPP diverged. ecef norm: %.2f", ecef_norm);
        return;
    }

    Eigen::Vector3d geo_pos = gnss_comm::ecef2geo(corrected_ecef_pos);

    const double lat_deg = geo_pos[0];
    const double lon_deg = geo_pos[1];
    const double alt = geo_pos[2];

    if (alt < -200.0 || alt > 5000.0) {
        RCLCPP_WARN_THROTTLE(logger, *clock, 2000,
                            "SPP Alt out of bounds: %.2f m. Rejected.", alt);
        return;
    }

    yaw_bridge_.set_latest_raw_ecef(corrected_ecef_pos, this->now().seconds());

    RCLCPP_INFO_THROTTLE(logger, *clock, 2000,
                            "SPP Succeeded | Lat: %.8f, Lon: %.8f, Alt: %.2f m | Sats: %zu",
                            lat_deg, lon_deg, alt, robust_obs.size());
    
    // 发布 NavSatFix (WGS84 经纬高)
    sensor_msgs::msg::NavSatFix fix_msg;

    // fix_msg.header.stamp = msg->header.stamp;
    fix_msg.header.stamp = this->now();
    fix_msg.header.frame_id = "gnss_link";

    fix_msg.status.status = sensor_msgs::msg::NavSatStatus::STATUS_FIX;
    fix_msg.status.service = sensor_msgs::msg::NavSatStatus::SERVICE_GPS; 

    fix_msg.latitude = lat_deg;
    fix_msg.longitude = lon_deg;
    fix_msg.altitude = alt;
    // 由于此处是单点定位(SPP)且未单独提取协方差矩阵，声明为未知协方差
    fix_msg.position_covariance_type = sensor_msgs::msg::NavSatFix::COVARIANCE_TYPE_UNKNOWN;
    pub_spp_fix_raw_->publish(fix_msg);

    // 保存原始SPP轨迹
    if (kml_exporter_raw_) {
        kml_exporter_raw_->add_point(lat_deg, lon_deg, alt);
    }

    // ====== 两阶段 origin 建立 ======
    // 阶段1: SPP 稳定后给 yaw bridge 设临时 origin（用于 yaw 方向计算）
    // 阶段2: yaw 标定完成后，重新收集 SPP 设最终融合 origin（此时 stereo 位移≈0）
    if (is_algo_running_ && !is_origin_set_) {
        init_ecef_buffer_.push_back(corrected_ecef_pos);
        if (init_ecef_buffer_.size() > INIT_WINDOW_SIZE) init_ecef_buffer_.pop_front();

        if (init_ecef_buffer_.size() == INIT_WINDOW_SIZE) {
            Eigen::Vector3d mean_ecef = Eigen::Vector3d::Zero();
            for (const auto& pos : init_ecef_buffer_) {
                mean_ecef += pos;
            }
            mean_ecef /= INIT_WINDOW_SIZE;

            double variance = 0.0;
            for (const auto& pos : init_ecef_buffer_) {
                variance += (pos - mean_ecef).squaredNorm();
            }
            variance /= INIT_WINDOW_SIZE;
            double std_dev = std::sqrt(variance);

            if (std_dev < STABILITY_THRESHOLD) {
                if (!is_yaw_bridge_initialized_) {
                    // === 阶段1: 给 yaw bridge 设临时 origin ===
                    origin_ecef_ = mean_ecef;
                    origin_lla_  = gnss_comm::ecef2geo(origin_ecef_);
                    R_enu2ecef_  = gnss_comm::geo2rotation(origin_lla_);
                    yaw_bridge_.set_origin(origin_ecef_, R_enu2ecef_);
                    is_yaw_bridge_initialized_ = true;
                    init_ecef_buffer_.clear();  // 清空，阶段2 重新收集
                    transition_gnss_status(GnssStatusState::WaitingOrigin);

                    RCLCPP_INFO(this->get_logger(),
                        "Phase1: Yaw bridge origin set. LLA: [%.8f, %.8f, %.2f]. Waiting for yaw calibration...",
                        origin_lla_[0], origin_lla_[1], origin_lla_[2]);

                } else if (is_yaw_calibrated_flag_) {
                    // === 阶段2: yaw 已标定，设最终融合 origin ===
                    origin_ecef_ = mean_ecef;
                    origin_lla_  = gnss_comm::ecef2geo(origin_ecef_);
                    R_enu2ecef_  = gnss_comm::geo2rotation(origin_lla_);
                    // 同步更新 yaw bridge 坐标参考，保留已标定的 yaw
                    yaw_bridge_.update_origin(origin_ecef_, R_enu2ecef_);
                    is_origin_set_ = true;
                    transition_gnss_status(GnssStatusState::Fusing);

                    // 记录此刻 stereo 位置作为零点偏移
                    init_ecef_buffer_.clear();
                    has_prev_enu_pos_raw_ = false;
                    accumulated_enu_pos_.setZero();
                    if (has_latest_odom_) {
                        origin_odom_offset_ = latest_odom_pos_raw_;
                    }

                    RCLCPP_INFO(this->get_logger(),
                        "Phase2: Fusion Origin Set! LLA: [%.8f, %.8f, %.2f]. odom_offset=[%.2f,%.2f,%.2f]",
                        origin_lla_[0], origin_lla_[1], origin_lla_[2],
                        origin_odom_offset_.x(), origin_odom_offset_.y(), origin_odom_offset_.z());
                }
            } else {
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                    "Waiting for GNSS to stabilize... StdDev: %.2f m (need < %.2f m)",
                    std_dev, STABILITY_THRESHOLD);
            }
        }
    }

}

void GnssSppNode::algo_status_callback(const system_ctrl::msg::AlgoStatus::ConstSharedPtr msg)
{
    if (msg->algo_status == "stereo3_running" || msg->algo_status == "stereo4_running") {
        is_algo_running_ = true;
        transition_gnss_status(is_origin_set_ ? GnssStatusState::Fusing : GnssStatusState::WaitingOrigin);
    } else {
        is_algo_running_ = false;
        // reset all states
        is_yaw_bridge_initialized_ = false;
        is_yaw_calibrated_flag_ = false;
        is_origin_set_ = false;
        init_ecef_buffer_.clear();
        yaw_bridge_.reset();
        yaw_bridge_.clear_raw_gnss();
        has_prev_enu_pos_raw_ = false;
        accumulated_enu_pos_.setZero();
        origin_odom_offset_.setZero();
        has_latest_odom_ = false;
        prev_yaw_offset_rad_ = 0.0;
        yaw_stable_count_ = 0;
        transition_gnss_status(GnssStatusState::Idle);
    }
}

void GnssSppNode::odom_callback(const nav_msgs::msg::Odometry::ConstSharedPtr msg)
{
    // Extract position
    double x = msg->pose.pose.position.x;
    double y = msg->pose.pose.position.y;
    double z = msg->pose.pose.position.z;

    // Stereo odom is relative to its initial frame, first map it to a pseudo-ENU frame.
    Eigen::Vector3d enu_pos_raw(-y, x, z);

    // 记录最新 odom 位置
    latest_odom_pos_raw_ = enu_pos_raw;
    has_latest_odom_ = true;

    if (!is_algo_running_) return;

    // ====== 阶段1: yaw bridge 标定 + 收敛等待 ======
    if (is_yaw_bridge_initialized_ && !is_yaw_calibrated_flag_) {
        const auto yaw_result = yaw_bridge_.update(enu_pos_raw, this->now().seconds());
        publish_bridge_diagnostics(yaw_result);

        if (!yaw_result.calibrated) {
            // 尚未达到 min_samples，继续采集
            if (yaw_result.used_new_gnss && !yaw_result.has_measurement) {
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 3000,
                    "Measuring yaw: baseline too short. odom=[%.3f,%.3f,%.3f]m", x, y, z);
            } else {
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 3000,
                    "Calibrating stereo yaw... samples=%d, gnss_fresh=%d",
                    yaw_result.calib_samples, yaw_result.gnss_fresh ? 1 : 0);
            }
        } else {
            // 已标定，只在有新 GNSS 测量导致 yaw 实际更新时才检查收敛
            if (yaw_result.used_new_gnss && yaw_result.has_measurement) {
                const double yaw_change_deg = std::abs(yaw_result.yaw_offset_rad - prev_yaw_offset_rad_) * 180.0 / M_PI;
                prev_yaw_offset_rad_ = yaw_result.yaw_offset_rad;

                if (yaw_change_deg < YAW_CONVERGE_THRESH_DEG) {
                    ++yaw_stable_count_;
                } else {
                    yaw_stable_count_ = 0;
                }

                RCLCPP_INFO(this->get_logger(),
                    "Yaw stabilizing... offset=%.2f deg, change=%.3f deg, stable=%d/%d",
                    yaw_result.yaw_offset_rad * 180.0 / M_PI, yaw_change_deg,
                    yaw_stable_count_, YAW_STABLE_REQUIRED);

                if (yaw_stable_count_ >= YAW_STABLE_REQUIRED) {
                    is_yaw_calibrated_flag_ = true;
                    RCLCPP_INFO(this->get_logger(),
                        "Yaw converged! offset=%.2f deg. Collecting SPP for final origin...",
                        yaw_result.yaw_offset_rad * 180.0 / M_PI);
                }
            }
        }
        return;  // 等待收敛 + 最终 origin 设定
    }

    // ====== 阶段2: 最终 origin 已设定，输出位置 ======
    if (!is_origin_set_) return;

    // 减去 origin 时刻的 stereo 位置，得到相对于 GNSS origin 的位移
    const Eigen::Vector3d enu_pos_rel = enu_pos_raw - origin_odom_offset_;

    // yaw bridge 继续更新（精化 yaw）
    const auto yaw_result = yaw_bridge_.update(enu_pos_raw, this->now().seconds());
    publish_bridge_diagnostics(yaw_result);

    if (!yaw_result.calibrated)
    {
        // yaw 丢失（连续拒绝过多导致 re-bootstrap），暂停输出
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 3000,
            "Yaw lost (re-bootstrapping). samples=%d, reject=%d",
            yaw_result.calib_samples, yaw_result.reject_count);
        return;
    }

    if (yaw_result.used_new_gnss && yaw_result.reject_count > 0)
    {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 3000,
            "Yaw update rejected. reject_count=%d, yaw_offset=%.2f deg",
            yaw_result.reject_count,
            yaw_result.yaw_offset_rad * 180.0 / M_PI);
    }

    const auto &q = msg->pose.pose.orientation;
    const double siny_cosp = 2.0 * (q.w * q.z + q.x * q.y);
    const double cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
    const double yaw_rel = std::atan2(siny_cosp, cosy_cosp);

    const double e_raw_fwd = -std::sin(yaw_rel);
    const double n_raw_fwd = std::cos(yaw_rel);
    const double c = std::cos(yaw_result.yaw_offset_rad);
    const double s = std::sin(yaw_result.yaw_offset_rad);
    const double e_fwd = c * e_raw_fwd - s * n_raw_fwd;
    const double n_fwd = s * e_raw_fwd + c * n_raw_fwd;

    double heading_deg = std::atan2(e_fwd, n_fwd) * 180.0 / M_PI;
    if (heading_deg < 0.0) heading_deg += 360.0;

    std_msgs::msg::Float64 heading_msg;
    heading_msg.data = heading_deg;
    pub_heading_->publish(heading_msg);

    // 直接旋转：计算当前 yaw 下的理想位置
    const Eigen::Vector3d ideal_enu_pos = yaw_bridge_.rotate_raw_to_enu(enu_pos_rel);

    // 渐进修正注入：增量式保平滑 + 渐进追踪直接旋转的理想位置
    if (!has_prev_enu_pos_raw_) {
        accumulated_enu_pos_ = ideal_enu_pos;
        has_prev_enu_pos_raw_ = true;
    } else {
        // 增量式累积（帧间平滑）
        const Eigen::Vector3d delta_raw = enu_pos_rel - prev_enu_pos_raw_;
        accumulated_enu_pos_ += yaw_bridge_.rotate_raw_to_enu(delta_raw);

        // 渐进注入修正量，使累积位置逐渐收敛到理想位置
        const Eigen::Vector3d correction = ideal_enu_pos - accumulated_enu_pos_;
        accumulated_enu_pos_ += YAW_CORRECTION_RATE * correction;
    }
    prev_enu_pos_raw_ = enu_pos_rel;

    // Transform to ECEF
    Eigen::Vector3d delta_ecef = R_enu2ecef_ * accumulated_enu_pos_;
    Eigen::Vector3d cur_ecef = origin_ecef_ + delta_ecef;

    // Convert ECEF back to Geodetic LLA
    Eigen::Vector3d cur_lla = gnss_comm::ecef2geo(cur_ecef);

    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                            "Fused | Lat: %.8f, Lon: %.8f, Alt: %.2f m | acc_enu=[%.2f,%.2f,%.2f]",
                            cur_lla[0], cur_lla[1], cur_lla[2],
                            accumulated_enu_pos_.x(), accumulated_enu_pos_.y(), accumulated_enu_pos_.z());

    // Publish high precision fused nav sat fix
    sensor_msgs::msg::NavSatFix fix_msg;
    fix_msg.header = msg->header;
    fix_msg.header.frame_id = "gnss_link"; 
    fix_msg.status.status = sensor_msgs::msg::NavSatStatus::STATUS_FIX;
    fix_msg.status.service = sensor_msgs::msg::NavSatStatus::SERVICE_GPS;
    fix_msg.latitude = cur_lla[0];
    fix_msg.longitude = cur_lla[1];
    fix_msg.altitude = cur_lla[2];
    fix_msg.position_covariance_type = sensor_msgs::msg::NavSatFix::COVARIANCE_TYPE_UNKNOWN;
    
    pub_spp_fix_->publish(fix_msg);

    // Save to KML
    if (kml_exporter_) {
        kml_exporter_->add_point(cur_lla[0], cur_lla[1], cur_lla[2]);
    }
}
/**
* @brief 提取L1单频数据并使用Hatch滤波进行载波相位平滑伪距
*/
void GnssSppNode::filter_L1_hatch(
    const std::vector<gnss_comm::ObsPtr>& obs_in,
    const std::vector<gnss_comm::EphemBasePtr>& ephem_in,
    std::vector<gnss_comm::ObsPtr>& obs_out,
    std::vector<gnss_comm::EphemBasePtr>& ephem_out)
{
    obs_out.clear();
    ephem_out.clear();

    for (size_t i = 0; i < obs_in.size(); ++i)
    {
        const auto& obs = obs_in[i];
        
        // 检查卫星系统
        const uint32_t sys = gnss_comm::satsys(obs->sat, nullptr);
        if (sys != SYS_GPS && sys != SYS_GLO && sys != SYS_BDS && sys != SYS_GAL)
            continue;

        // 提取 L1 频率
        int l1_idx = -1;
        const double f1 = gnss_comm::L1_freq(obs, &l1_idx);
        if (f1 < 0 || l1_idx < 0) continue;
        if (obs->psr[l1_idx] == 0.0) continue;

        double p1 = obs->psr[l1_idx];
        double std1 = obs->psr_std[l1_idx];
        
        // 将载波相位从 cycle 转换为 m
        double cp1_m = obs->cp[l1_idx] * (LIGHT_SPEED / f1);

        double current_time = obs->time.time + obs->time.sec;
        auto& state = hatch_states_[obs->sat];

        // 检查周跳、相位无效等重置条件
        bool cycle_slip = (obs->LLI[l1_idx] != 0);
        if (obs->cp[l1_idx] == 0.0) cycle_slip = true;

        // 如果发生周跳、时间间隔过大(>2.5秒) 或 刚初始化，则重置滤波器
        if (state.count == 0 || cycle_slip || std::abs(current_time - state.last_time) > 2.5) {
            state.p_smooth = p1;
            state.cp_prev  = cp1_m;
            state.count    = 1;
        } else {
            state.count = std::min(state.count + 1, MAX_HATCH_WINDOW);
            double k = static_cast<double>(state.count);
            // Hatch滤波递推公式: P_smooth(k) = 1/k * P_raw(k) + (k-1)/k * (P_smooth(k-1) + cp(k) - cp(k-1))
            state.p_smooth = (1.0 / k) * p1 + ((k - 1.0) / k) * (state.p_smooth + cp1_m - state.cp_prev);
            state.cp_prev = cp1_m;
        }

        // 粗略估计平滑后的标准差
        double smoothed_std1 = std1 / std::sqrt(static_cast<double>(state.count));
        state.last_time = current_time;

        auto l1_obs = std::make_shared<gnss_comm::Obs>();
        l1_obs->time = obs->time;
        l1_obs->sat  = obs->sat;
        
        l1_obs->freqs.push_back(f1);
        l1_obs->psr.push_back(state.p_smooth);
        l1_obs->psr_std.push_back(smoothed_std1);

        l1_obs->CN0.push_back(obs->CN0[l1_idx]); 
        l1_obs->status.push_back(obs->status[l1_idx]);
        l1_obs->LLI.push_back(0);
        l1_obs->code.push_back(obs->code[l1_idx]);
        l1_obs->cp.push_back(0.0);
        l1_obs->cp_std.push_back(0.0);
        l1_obs->dopp.push_back(0.0);
        l1_obs->dopp_std.push_back(0.0);

        obs_out.push_back(l1_obs);
        ephem_out.push_back(ephem_in[i]);
    }
}

/**
* @brief 提取多频数据并构造最优无电离层组合 (IF) 单频观测值
*/
void GnssSppNode::filter_iono_free(
    const std::vector<gnss_comm::ObsPtr>& obs_in,
    const std::vector<gnss_comm::EphemBasePtr>& ephem_in,
    std::vector<gnss_comm::ObsPtr>& obs_out,
    std::vector<gnss_comm::EphemBasePtr>& ephem_out)
{
    obs_out.clear();
    ephem_out.clear();

    for (size_t i = 0; i < obs_in.size(); ++i)
    {
        const auto& obs = obs_in[i];
        auto find_freq_idx = [&](double target_f) -> int {
            double min_diff = 1e12;
            int idx = -1;

            for (size_t j = 0; j < obs->freqs.size(); ++j) {
                double f = obs->freqs[j];

                if (f < 1e8) continue;
                if (obs->psr[j] == 0.0) continue;

                double diff = std::abs(f - target_f);
                if (diff < min_diff) {
                    min_diff = diff;
                    idx = static_cast<int>(j);
                }
            }
            return idx;
        };
        int idx1 = -1; // 用于记录高频率索引
        int idx2 = -1; // 用于记录低频率索引

        // 获取卫星系统
        int sys = gnss_comm::satsys(obs->sat, nullptr);
        // 根据不同系统分配目标频率
        
        switch (sys)
        {
            case SYS_GPS:
            case SYS_QZS:
            {
                // L1 + L2
                idx1 = find_freq_idx(FREQ1);
                idx2 = find_freq_idx(FREQ2);

                // fallback: L1 + L5
                if (idx1 == -1 || idx2 == -1) {
                    idx1 = find_freq_idx(FREQ1);
                    idx2 = find_freq_idx(FREQ5);
                }
                break;
            }

            case SYS_GAL:
            {
                // E1 + E5a
                idx1 = find_freq_idx(FREQ1);
                idx2 = find_freq_idx(FREQ5);

                // fallback: E1 + E5b
                if (idx1 == -1 || idx2 == -1) {
                    idx1 = find_freq_idx(FREQ1);
                    idx2 = find_freq_idx(FREQ7);
                }
                break;
            }

            case SYS_BDS:
            {
                // 优先：B1 + B2 (兼容老设备)
                idx1 = find_freq_idx(FREQ1_BDS);
                idx2 = find_freq_idx(FREQ2_BDS);

                // fallback: B1 + B3
                if (idx1 == -1 || idx2 == -1) {
                    idx1 = find_freq_idx(FREQ1_BDS);
                    idx2 = find_freq_idx(FREQ3_BDS);
                }

                // fallback: GPS L1/L2（部分接收机混合输出）
                if (idx1 == -1 || idx2 == -1) {
                    idx1 = find_freq_idx(FREQ1);
                    idx2 = find_freq_idx(FREQ2);
                }
                break;
            }

            case SYS_IRN:
            {
                // L5 + S
                idx1 = find_freq_idx(FREQ5);
                idx2 = find_freq_idx(FREQ9);
                break;
            }

            case SYS_SBS:
            {
                // L1 + L5
                idx1 = find_freq_idx(FREQ1);
                idx2 = find_freq_idx(FREQ5);
                break;
            }
            default: break;
        }

        // 确保找到了两个不同的频率
        if (idx1 != -1 && idx2 != -1 && idx1 != idx2)
        {
            double f1 = obs->freqs[idx1];
            double f2 = obs->freqs[idx2];
            double p1 = obs->psr[idx1];
            double p2 = obs->psr[idx2];
            double std1 = obs->psr_std[idx1];
            double std2 = obs->psr_std[idx2];

            double f1_2 = f1 * f1;
            double f2_2 = f2 * f2;
            double denominator = f1_2 - f2_2;

            double alpha = f1_2 / denominator;
            double beta  = -f2_2 / denominator;

            double p_if = alpha * p1 + beta * p2;
            double std_if = std::sqrt(alpha * alpha * std1 * std1 + beta * beta * std2 * std2);

            // Hatch filter 相位平滑伪距
            // 将载波相位从 cycle 转换为 m
            double cp1_m = obs->cp[idx1] * (LIGHT_SPEED / f1);
            double cp2_m = obs->cp[idx2] * (LIGHT_SPEED / f2);
            double cp_if = alpha * cp1_m + beta * cp2_m;

            double current_time = obs->time.time + obs->time.sec;
            auto& state = hatch_states_[obs->sat];

            // 检查周跳、相位无效等重置条件
            bool cycle_slip = (obs->LLI[idx1] != 0) || (obs->LLI[idx2] != 0);
            if (obs->cp[idx1] == 0.0 || obs->cp[idx2] == 0.0) cycle_slip = true;

            // 如果发生周跳、时间间隔过大(>2.5秒) 或 刚初始化，则重置滤波器
            if (state.count == 0 || cycle_slip || std::abs(current_time - state.last_time) > 2.5) {
                state.p_smooth = p_if;
                state.cp_prev  = cp_if;
                state.count    = 1;
            } else {
                state.count = std::min(state.count + 1, MAX_HATCH_WINDOW);
                double k = static_cast<double>(state.count);
                // Hatch滤波递推公式: P_smooth(k) = 1/k * P_raw(k) + (k-1)/k * (P_smooth(k-1) + cp_if(k) - cp_if(k-1))
                state.p_smooth = (1.0 / k) * p_if + ((k - 1.0) / k) * (state.p_smooth + cp_if - state.cp_prev);
                state.cp_prev = cp_if;
            }
            // 粗略估计平滑后的标准差
            double smoothed_std_if = std_if / std::sqrt(static_cast<double>(state.count));
            state.last_time = current_time;

            auto if_obs = std::make_shared<gnss_comm::Obs>();
            if_obs->time = obs->time;
            if_obs->sat  = obs->sat;
            
            // 将组合伪距作为单频数据喂入，标称频率设为最高频 f1
            if_obs->freqs.push_back(f1);
            if_obs->psr.push_back(state.p_smooth);
            if_obs->psr_std.push_back(smoothed_std_if);

            if_obs->CN0.push_back(obs->CN0[idx1]); 
            if_obs->status.push_back(obs->status[idx1]);
            if_obs->LLI.push_back(0);
            if_obs->code.push_back(obs->code[idx1]);
            if_obs->cp.push_back(0.0);
            if_obs->cp_std.push_back(0.0);
            if_obs->dopp.push_back(0.0);
            if_obs->dopp_std.push_back(0.0);

            obs_out.push_back(if_obs);
            ephem_out.push_back(ephem_in[i]);
        } else { // 直接降级为 L1 单频
            obs_out.push_back(obs_in[i]);
            ephem_out.push_back(ephem_in[i]);
        }
    }
}
/**
* @brief 剔除可用卫星数过少的弱势星座，防止 SPP 解算时钟差矩阵秩亏发散
* @param obs_in        输入的观测值集合
* @param ephem_in        输入的星历集合（与 obs_in 一一对应）
* @param obs_out         输出的过滤后的观测值
* @param ephem_out       输出的过滤后的星历
* @param min_sat_per_sys 判定为强势星座的最低卫星数
*/
void GnssSppNode::filter_weak_constellations(
    const std::vector<gnss_comm::ObsPtr>& obs_in,
    const std::vector<gnss_comm::EphemBasePtr>& ephem_in,
    std::vector<gnss_comm::ObsPtr>& obs_out,
    std::vector<gnss_comm::EphemBasePtr>& ephem_out,
    const int min_sat_per_sys)
{
    obs_out.clear();
    ephem_out.clear();

    // 确保观测值和星历完全对齐
    if (obs_in.empty() || obs_in.size() != ephem_in.size()) {
        return;
    }

    // 统计当前历元下，各个星座的可用卫星总数
    std::unordered_map<int, int> sys_sat_count;
    for (const auto& obs : obs_in) {
        int sys = gnss_comm::satsys(obs->sat, nullptr);
        sys_sat_count[sys]++;
    }

    // 基于统计结果，进行精准过滤
    obs_out.reserve(obs_in.size());
    ephem_out.reserve(ephem_in.size());
    
    for (size_t i = 0; i < obs_in.size(); ++i) {
        int sys = gnss_comm::satsys(obs_in[i]->sat, nullptr);
        
        // 只有当该星座的卫星数大于等于阈值时，才放行
        if (sys_sat_count[sys] >= min_sat_per_sys) {
            obs_out.push_back(obs_in[i]);
            ephem_out.push_back(ephem_in[i]);
        } else {
            // 打印 Debug 信息，方便调试时观察是哪个星座被丢弃了
            RCLCPP_DEBUG(this->get_logger(), 
                "Dropped sat %u, system %d is too weak (only %d sats)", 
                obs_in[i]->sat, sys, sys_sat_count[sys]);
        }
    }
}


// =========================================================================
//  Ephemeris message validation
// =========================================================================
bool GnssSppNode::is_valid_ephem_msg(const gnss_comm::msg::GnssEphemMsg &msg, std::string *reason)
{
    auto reject = [&](const std::string &r) -> bool {
        if (reason) *reason = r;
        return false;
    };

    // 1. 卫星 ID 不能为 0
    if (msg.sat == 0) return reject("sat == 0");

    // 2. 检查卫星系统是否支持
    uint32_t sys = gnss_comm::satsys(msg.sat, nullptr);
    if (sys != SYS_GPS && sys != SYS_GLO && sys != SYS_GAL && sys != SYS_BDS) return reject("unsupported satellite system for ephem (sat=" + std::to_string(msg.sat) + ")");

    // 3. GPS 周数合理性 (GPS rollover: week 大约在 1024 ~ 4096 范围)
    if (msg.week == 0 || msg.week > 4096) return reject("week out of range: " + std::to_string(msg.week));

    // 4. TOE 周数和周内秒合理性
    if (msg.toe.week == 0) return reject("toe.week == 0");
    if (msg.toe.tow < 0.0 || msg.toe.tow > WEEK_SECONDS) return reject("toe.tow out of range: " + std::to_string(msg.toe.tow));

    // 5. TOC 周内秒合理性
    if (msg.toc.tow < 0.0 || msg.toc.tow > WEEK_SECONDS) return reject("toc.tow out of range: " + std::to_string(msg.toc.tow));

    // 6. 半长轴 A 必须为正且合理 (地球轨道约 26,000,000~42,200,000 m)
    //    msg.a 存的是 sqrt(a)，所以 a = msg.a * msg.a
    if (msg.a <= 0.0) return reject("sqrt(A) <= 0");
    double semi_major = msg.a * msg.a;  // a field stores sqrt(A) in some conventions
    // 保守范围：20,000 km ~ 50,000 km 半长轴
    if (semi_major < 2.0e7 || semi_major > 5.0e7)
    {
        // 如果 msg.a 本身就在合理范围则通过
        if (msg.a < 2.0e7 || msg.a > 5.0e7)
            return reject("semi-major axis out of range: a=" + std::to_string(msg.a));
    }

    // 7. 离心率 e 必须在 [0, 1) 之间
    if (msg.e < 0.0 || msg.e >= 1.0) return reject("eccentricity out of range: " + std::to_string(msg.e));

    // 8. 轨道倾角 i0 应在合理范围 (0 ~ pi)
    if (msg.i0 < 0.0 || msg.i0 > M_PI) return reject("i0 out of range: " + std::to_string(msg.i0));

    // 9. URA (用户距离精度) 必须为正值
    if (msg.ura <= 0.0) return reject("ura <= 0: " + std::to_string(msg.ura));

    // 10. 星历字段不能包含 NaN / Inf
    auto is_finite = [](double v) { return std::isfinite(v); };
    if (!is_finite(msg.a) || !is_finite(msg.e) || !is_finite(msg.i0) ||
        !is_finite(msg.omg) || !is_finite(msg.omg0) || !is_finite(msg.m0) ||
        !is_finite(msg.delta_n) || !is_finite(msg.omg_dot) || !is_finite(msg.i_dot) ||
        !is_finite(msg.cuc) || !is_finite(msg.cus) || !is_finite(msg.crc) ||
        !is_finite(msg.crs) || !is_finite(msg.cic) || !is_finite(msg.cis) ||
        !is_finite(msg.af0) || !is_finite(msg.af1) || !is_finite(msg.af2))
        return reject("orbit/clock parameter contains NaN or Inf");

    return true;
}

// =========================================================================
//  Observation message validation
// =========================================================================
bool GnssSppNode::is_valid_obs_msg(const gnss_comm::msg::GnssObsMsg &msg, std::string *reason)
{
    auto reject = [&](const std::string &r) -> bool {
        if (reason) *reason = r;
        return false;
    };

    // 1. 卫星 ID 不能为 0
    if (msg.sat == 0) return reject("sat == 0");

    // 2. 检查卫星系统是否支持
    uint32_t sys = gnss_comm::satsys(msg.sat, nullptr);
    if (sys != SYS_GPS && sys != SYS_GLO && sys != SYS_GAL && sys != SYS_BDS) return reject("unsupported satellite system (sat=" + std::to_string(msg.sat) + ")");
        
    // 3. 观测时间有效性
    if (msg.time.week == 0) return reject("time.week == 0");
    if (msg.time.tow < 0.0 || msg.time.tow > WEEK_SECONDS) return reject("time.tow out of range: " + std::to_string(msg.time.tow));

    // 4. 频率数组不能为空（至少要有一个频率的观测）
    if (msg.freqs.empty()) return reject("freqs array is empty");

    const size_t n = msg.freqs.size();
    // 5. 所有观测数组长度必须一致
    if (msg.cn0.size()  != n || msg.lli.size()  != n || msg.code.size() != n ||
        msg.psr.size()  != n || msg.psr_std.size() != n ||
        msg.cp.size()   != n || msg.cp_std.size()  != n ||
        msg.dopp.size() != n || msg.dopp_std.size() != n ||
        msg.status.size() != n)
        return reject("array size mismatch (expected " + std::to_string(n) + " elements for all arrays)");

    // 6. 逐频率检查
    bool has_valid_psr = false;
    for (size_t i = 0; i < n; ++i)
    {
        // 频率值应该是 L-band 范围 (1.1 ~ 2.6 GHz)
        if (msg.freqs[i] < 1.0e9 || msg.freqs[i] > 3.0e9) return reject("freqs[" + std::to_string(i) + "] out of L-band range: " + std::to_string(msg.freqs[i]));

        // CN0 (载噪比) 范围 [30, 65] dB-Hz
        if (msg.cn0[i] < 30.0 || msg.cn0[i] > 65.0) return reject("cn0[" + std::to_string(i) + "] out of range: " + std::to_string(msg.cn0[i]));

        // 伪距标准差不能为负
        if (msg.psr_std[i] < 0.0 || msg.psr_std[i] > 10.0) return reject("psr_std[" + std::to_string(i) + "] out of range: " + std::to_string(msg.psr_std[i]));

        // 载波相位标准差不能为负
        if (msg.cp_std[i] < 0.0) return reject("cp_std[" + std::to_string(i) + "] < 0");

        // 多普勒标准差不能为负
        if (msg.dopp_std[i] < 0.0) return reject("dopp_std[" + std::to_string(i) + "] < 0");

        // 检查 status bit_0（伪距有效位）
        // bool psr_valid = (msg.status[i] & 0x01) != 0;
        // if (!psr_valid) continue;
        
        // 伪距值应在合理范围：约 20,000 km ~ 30,000 km 的轨道对应
        // 地面伪距约 20,000,000 ~ 90,000,000 m (含钟差)
        if (msg.psr[i] < 1.0e7 || msg.psr[i] > 1.0e8) continue;
        has_valid_psr = true;

        // 检查 NaN / Inf
        if (!std::isfinite(msg.psr[i]) || !std::isfinite(msg.cp[i]) || !std::isfinite(msg.dopp[i])) return reject("psr/cp/dopp contains NaN or Inf at index " + std::to_string(i));
    }

    // 7. 至少有一个频率的伪距是有效的
    if (!has_valid_psr) return reject("no valid pseudorange found (all status bit_0 == 0 or psr out of range)");

    return true;
}
void GnssSppNode::cleanup_ephem_callback()
{
    auto logger = this->get_logger();
    int removed_count = 0;

    // 获取当前系统时间并转换为 GNSS 时间 (GPST)
    auto now_ros = this->now();
    double now_unix = now_ros.seconds();
    
    std::lock_guard<std::mutex> lock(ephem_mutex_);
    
    auto it = ephem_map_.begin();
    while (it != ephem_map_.end())
    {
        const auto& ephem = it->second;
        
        // 计算星历年龄
        // 将 ephem->toe 转换为 Unix 时间戳对比
        double ephem_toe_unix = gnss_comm::time2sec(ephem->toe); 
        double age = std::abs(now_unix - ephem_toe_unix);

        // 判定是否过期
        if (age > MAX_EPHEM_AGE_SEC)
        {
            RCLCPP_INFO(logger, "Cleaning up expired ephem: Sat=%u, Age=%.1f hours", 
                        it->first, age / 3600.0);
            it = ephem_map_.erase(it);
            removed_count++;
        } else{
            ++it;
        }
    }

    if (removed_count > 0) RCLCPP_INFO(logger, "Ephem cleanup finished. Removed %d stale entries.", removed_count);
}
