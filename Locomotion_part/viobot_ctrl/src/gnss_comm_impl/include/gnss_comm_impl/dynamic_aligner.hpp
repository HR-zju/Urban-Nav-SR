#pragma once

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <deque>
#include <vector>
#include <random>
#include <cmath>

namespace gnss_comm_impl {

class DynamicAligner {
public:
    struct Config {
        size_t window_size = 100;                 // Max number of pairs in sliding window
        double min_dist_between_pairs = 1.0;      // Minimum distance (m) to add a new point
        int ransac_iterations = 100;              // Number of RANSAC iterations
        double ransac_inlier_thresh = 2.0;        // Inlier threshold (m)
        size_t min_inliers_required = 10;         // Minimum inliers to consider alignment successful
        double filter_gain = 0.05;                // Low-pass filter gain for exponential smoothing
        double max_yaw_jump_rad = 0.5;            // Max allowed yaw jump (rad) in a single step (approx 28 deg)
        double max_trans_jump_m = 10.0;           // Max allowed translation jump (m) in a single step
        size_t anchor_window_size = 50;           // Max number of pairs in sparse anchor buffer
        double anchor_min_dist = 10.0;            // Minimum distance (m) to add a new anchor point
    };

    struct AlignPair {
        Eigen::Vector2d odom;
        Eigen::Vector2d gnss;
    };

    struct AlignmentResult {
        bool success = false;
        int inlier_count = 0;
        double yaw = 0.0;
        Eigen::Vector2d translation = Eigen::Vector2d::Zero();
    };

    DynamicAligner() = default;
    explicit DynamicAligner(const Config& cfg);

    void addMeasurement(const Eigen::Vector2d& odom, const Eigen::Vector2d& gnss);

    AlignmentResult align();

    bool isAligned() const;

    Eigen::Vector3d transform(const Eigen::Vector3d& odom_enu) const;

    void reset();
    
    double getCurrentYaw() const;
    Eigen::Vector2d getCurrentTranslation() const;
    size_t getBufferSize() const;

private:
    double normalize_angle(double rad) const;

    Config cfg_;
    bool is_aligned_ = false;
    double current_yaw_ = 0.0;
    Eigen::Vector2d current_translation_ = Eigen::Vector2d::Zero();

    std::deque<AlignPair> buffer_;
    std::deque<AlignPair> anchor_buffer_;
    std::mt19937 rng_;
};

} // namespace gnss_comm_impl