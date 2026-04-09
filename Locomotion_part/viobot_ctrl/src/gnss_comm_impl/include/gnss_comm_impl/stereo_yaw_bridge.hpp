#pragma once

#include <Eigen/Dense>

namespace gnss_comm_impl
{

class StereoYawBridge
{
public:
    static constexpr double PI = 3.14159265358979323846;

    struct Config
    {
        int min_samples = 5;
        double min_baseline_m = 2.0;
        double gnss_stale_sec = 1.5;
        double update_gain = 0.03;
        double max_step_rad = 2.0 * PI / 180.0;
        double max_innovation_rad = 25.0 * PI / 180.0;
        double hard_reject_rad = 120.0 * PI / 180.0;
        int max_reject_before_rebootstrap = 30;
    };

    struct UpdateResult
    {
        bool gnss_fresh = false;
        bool used_new_gnss = false;
        bool has_measurement = false;
        bool calibrated = false;
        int calib_samples = 0;
        int reject_count = 0;
        double yaw_offset_rad = 0.0;
    };

    StereoYawBridge();
    explicit StereoYawBridge(const Config &cfg);

    void set_origin(const Eigen::Vector3d &origin_ecef, const Eigen::Matrix3d &R_enu2ecef);
    void set_latest_raw_ecef(const Eigen::Vector3d &raw_ecef, double stamp_sec);
    void clear_raw_gnss();
    void reset();

    UpdateResult update(const Eigen::Vector3d &enu_pos_raw, double now_sec);
    Eigen::Vector3d rotate_raw_to_enu(const Eigen::Vector3d &enu_pos_raw) const;

private:
    double normalize_angle(double rad) const;
    bool compute_yaw_measurement(const Eigen::Vector3d &enu_pos_raw, double *yaw_meas);
    void update_yaw_offset_with_measurement(double yaw_meas);

private:
    Config cfg_;

    bool has_origin_ = false;
    bool has_latest_raw_ecef_ = false;
    bool has_new_raw_ecef_for_update_ = false;
    bool is_yaw_calibrated_ = false;

    int yaw_calib_samples_ = 0;
    int yaw_update_reject_count_ = 0;

    double yaw_offset_rad_ = 0.0;
    double yaw_sin_sum_ = 0.0;
    double yaw_cos_sum_ = 0.0;
    double latest_raw_stamp_sec_ = 0.0;
    bool has_motion_anchor_ = false;
    Eigen::Vector2d anchor_odom_2d_{Eigen::Vector2d::Zero()};
    Eigen::Vector2d anchor_gnss_enu_2d_{Eigen::Vector2d::Zero()};

    Eigen::Vector3d latest_raw_ecef_{0.0, 0.0, 0.0};
    Eigen::Vector3d origin_ecef_{0.0, 0.0, 0.0};
    Eigen::Matrix3d R_enu2ecef_{Eigen::Matrix3d::Identity()};
};

} // namespace gnss_comm_impl
