#include "gnss_comm_impl/stereo_yaw_bridge.hpp"

#include <cmath>

namespace gnss_comm_impl
{

StereoYawBridge::StereoYawBridge() : cfg_()
{
}

StereoYawBridge::StereoYawBridge(const Config &cfg) : cfg_(cfg)
{
}

void StereoYawBridge::set_origin(const Eigen::Vector3d &origin_ecef, const Eigen::Matrix3d &R_enu2ecef)
{
    origin_ecef_ = origin_ecef;
    R_enu2ecef_ = R_enu2ecef;
    has_origin_ = true;
    reset();
    clear_raw_gnss();
}

void StereoYawBridge::update_origin(const Eigen::Vector3d &origin_ecef, const Eigen::Matrix3d &R_enu2ecef)
{
    origin_ecef_ = origin_ecef;
    R_enu2ecef_ = R_enu2ecef;
    // 只重置运动锚点（GNSS ENU 坐标已变），保留 yaw 标定状态
    has_motion_anchor_ = false;
}

void StereoYawBridge::set_latest_raw_ecef(const Eigen::Vector3d &raw_ecef, double stamp_sec)
{
    latest_raw_ecef_ = raw_ecef;
    latest_raw_stamp_sec_ = stamp_sec;
    has_latest_raw_ecef_ = true;
    has_new_raw_ecef_for_update_ = true;
}

void StereoYawBridge::clear_raw_gnss()
{
    has_latest_raw_ecef_ = false;
    has_new_raw_ecef_for_update_ = false;
    latest_raw_stamp_sec_ = 0.0;
    has_motion_anchor_ = false;
}

void StereoYawBridge::reset()
{
    is_yaw_calibrated_ = false;
    yaw_calib_samples_ = 0;
    yaw_update_reject_count_ = 0;
    yaw_offset_rad_ = 0.0;
    yaw_sin_sum_ = 0.0;
    yaw_cos_sum_ = 0.0;
    has_motion_anchor_ = false;
    anchor_odom_2d_.setZero();
    anchor_gnss_enu_2d_.setZero();
}

StereoYawBridge::UpdateResult StereoYawBridge::update(const Eigen::Vector3d &enu_pos_raw, double now_sec)
{
    UpdateResult result;
    if (!has_origin_)
    {
        result.calibrated = false;
        return result;
    }

    result.gnss_fresh = has_latest_raw_ecef_ &&
                        (std::abs(now_sec - latest_raw_stamp_sec_) <= cfg_.gnss_stale_sec);
    result.used_new_gnss = result.gnss_fresh && has_new_raw_ecef_for_update_;

    if (result.used_new_gnss)
    {
        double yaw_meas = 0.0;
        result.has_measurement = compute_yaw_measurement(enu_pos_raw, &yaw_meas);
        if (result.has_measurement)
        {
            update_yaw_offset_with_measurement(yaw_meas);
        }
        has_new_raw_ecef_for_update_ = false;
    }

    result.calibrated = is_yaw_calibrated_;
    result.calib_samples = yaw_calib_samples_;
    result.reject_count = yaw_update_reject_count_;
    result.yaw_offset_rad = yaw_offset_rad_;
    return result;
}

Eigen::Vector3d StereoYawBridge::rotate_raw_to_enu(const Eigen::Vector3d &enu_pos_raw) const
{
    const double c = std::cos(yaw_offset_rad_);
    const double s = std::sin(yaw_offset_rad_);

    Eigen::Vector3d enu_pos;
    enu_pos.x() = c * enu_pos_raw.x() - s * enu_pos_raw.y();
    enu_pos.y() = s * enu_pos_raw.x() + c * enu_pos_raw.y();
    enu_pos.z() = enu_pos_raw.z();
    return enu_pos;
}

double StereoYawBridge::normalize_angle(double rad) const
{
    while (rad > PI) rad -= 2.0 * PI;
    while (rad < -PI) rad += 2.0 * PI;
    return rad;
}

bool StereoYawBridge::compute_yaw_measurement(const Eigen::Vector3d &enu_pos_raw, double *yaw_meas)
{
    if (!has_latest_raw_ecef_) return false;

    const Eigen::Vector3d raw_delta_ecef = latest_raw_ecef_ - origin_ecef_;
    const Eigen::Vector3d raw_delta_enu = R_enu2ecef_.transpose() * raw_delta_ecef;

    const Eigen::Vector2d odom_2d_curr(enu_pos_raw.x(), enu_pos_raw.y());
    const Eigen::Vector2d gnss_2d_curr(raw_delta_enu.x(), raw_delta_enu.y());

    if (!has_motion_anchor_)
    {
        anchor_odom_2d_ = odom_2d_curr;
        anchor_gnss_enu_2d_ = gnss_2d_curr;
        has_motion_anchor_ = true;
        return false;
    }

    const Eigen::Vector2d odom_2d = odom_2d_curr - anchor_odom_2d_;
    const Eigen::Vector2d gnss_2d = gnss_2d_curr - anchor_gnss_enu_2d_;

    const double odom_norm = odom_2d.norm();
    const double gnss_norm = gnss_2d.norm();
    if (odom_norm < cfg_.min_baseline_m || gnss_norm < cfg_.min_baseline_m)
    {
        return false;
    }

    const double cross = odom_2d.x() * gnss_2d.y() - odom_2d.y() * gnss_2d.x();
    const double dot = odom_2d.dot(gnss_2d);
    if (yaw_meas) *yaw_meas = std::atan2(cross, dot);

    anchor_odom_2d_ = odom_2d_curr;
    anchor_gnss_enu_2d_ = gnss_2d_curr;
    return true;
}

void StereoYawBridge::update_yaw_offset_with_measurement(double yaw_meas)
{
    if (!is_yaw_calibrated_)
    {
        yaw_sin_sum_ += std::sin(yaw_meas);
        yaw_cos_sum_ += std::cos(yaw_meas);
        ++yaw_calib_samples_;

        if (yaw_calib_samples_ >= cfg_.min_samples)
        {
            yaw_offset_rad_ = normalize_angle(std::atan2(yaw_sin_sum_, yaw_cos_sum_));
            is_yaw_calibrated_ = true;
            yaw_update_reject_count_ = 0;
        }
        return;
    }

    const double innovation = normalize_angle(yaw_meas - yaw_offset_rad_);
    if (std::abs(innovation) > cfg_.hard_reject_rad)
    {
        ++yaw_update_reject_count_;

        if (yaw_update_reject_count_ >= cfg_.max_reject_before_rebootstrap)
        {
            is_yaw_calibrated_ = false;
            yaw_calib_samples_ = 1;
            yaw_sin_sum_ = std::sin(yaw_meas);
            yaw_cos_sum_ = std::cos(yaw_meas);
            yaw_update_reject_count_ = 0;
        }
        return;
    }

    yaw_update_reject_count_ = 0;

    double clipped_innovation = innovation;
    if (clipped_innovation > cfg_.max_innovation_rad)
        clipped_innovation = cfg_.max_innovation_rad;
    else if (clipped_innovation < -cfg_.max_innovation_rad)
        clipped_innovation = -cfg_.max_innovation_rad;

    double step = cfg_.update_gain * clipped_innovation;
    if (step > cfg_.max_step_rad) step = cfg_.max_step_rad;
    if (step < -cfg_.max_step_rad) step = -cfg_.max_step_rad;

    yaw_offset_rad_ = normalize_angle(yaw_offset_rad_ + step);
}

} // namespace gnss_comm_impl
