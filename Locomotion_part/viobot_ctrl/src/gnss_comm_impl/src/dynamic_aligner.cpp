#include "gnss_comm_impl/dynamic_aligner.hpp"

namespace gnss_comm_impl {

DynamicAligner::DynamicAligner(const Config& cfg) : cfg_(cfg) {
    rng_.seed(std::random_device{}());
}

void DynamicAligner::addMeasurement(const Eigen::Vector2d& odom, const Eigen::Vector2d& gnss) {
    // Add to dense recent buffer
    if (buffer_.empty() || (odom - buffer_.back().odom).norm() >= cfg_.min_dist_between_pairs) {
        buffer_.push_back({odom, gnss});
        if (buffer_.size() > cfg_.window_size) {
            buffer_.pop_front();
        }
    }

    // Add to sparse anchor buffer
    if (anchor_buffer_.empty() || (odom - anchor_buffer_.back().odom).norm() >= cfg_.anchor_min_dist) {
        anchor_buffer_.push_back({odom, gnss});
        if (anchor_buffer_.size() > cfg_.anchor_window_size) {
            anchor_buffer_.pop_front();
        }
    }
}

DynamicAligner::AlignmentResult DynamicAligner::align() {
    AlignmentResult res;
    
    std::vector<AlignPair> all_points;
    all_points.insert(all_points.end(), buffer_.begin(), buffer_.end());
    all_points.insert(all_points.end(), anchor_buffer_.begin(), anchor_buffer_.end());

    if (all_points.size() < std::max<size_t>(3, cfg_.min_inliers_required)) {
        return res;
    }

    Eigen::Matrix2d best_R;
    Eigen::Vector2d best_t;
    double max_inlier_score = -1.0;
    std::vector<int> best_inliers;
    
    double anchor_weight = cfg_.anchor_min_dist / std::max(0.1, cfg_.min_dist_between_pairs);

    for (int iter = 0; iter < cfg_.ransac_iterations; ++iter) {
        std::uniform_int_distribution<size_t> dist(0, all_points.size() - 1);
        size_t idx1 = dist(rng_);
        size_t idx2 = dist(rng_);

        if (idx1 == idx2) continue;

        const auto& p1 = all_points[idx1];
        const auto& p2 = all_points[idx2];
        
        // Calculate transformation from p1, p2
        Eigen::Vector2d odom_diff = p2.odom - p1.odom;
        Eigen::Vector2d gnss_diff = p2.gnss - p1.gnss;

        if (odom_diff.norm() < 1e-6) continue;

        double angle_odom = std::atan2(odom_diff.y(), odom_diff.x());
        double angle_gnss = std::atan2(gnss_diff.y(), gnss_diff.x());
        double angle_diff = angle_gnss - angle_odom;

        Eigen::Matrix2d R;
        R << std::cos(angle_diff), -std::sin(angle_diff),
             std::sin(angle_diff),  std::cos(angle_diff);

        Eigen::Vector2d t = p1.gnss - R * p1.odom;

        double current_score = 0.0;
        std::vector<int> current_inliers;

        for (size_t i = 0; i < all_points.size(); ++i) {
            Eigen::Vector2d gnss_pred = R * all_points[i].odom + t;
            if ((gnss_pred - all_points[i].gnss).norm() < cfg_.ransac_inlier_thresh) {
                current_inliers.push_back(i);
                if (i < buffer_.size()) {
                    current_score += 1.0;
                } else {
                    current_score += anchor_weight;
                }
            }
        }

        if (current_score > max_inlier_score) {
            max_inlier_score = current_score;
            best_inliers = current_inliers;
            best_R = R;
            best_t = t;
        }
    }

    // Must have at least required inliers (based on raw count, not score, for numerical stability in Umeyama)
    if (best_inliers.size() < cfg_.min_inliers_required) {
        return res;
    }

    // Re-estimate using all inliers
    int num_inliers = best_inliers.size();
    Eigen::MatrixXd odom_inliers(2, num_inliers);
    Eigen::MatrixXd gnss_inliers(2, num_inliers);

    for (int i = 0; i < num_inliers; ++i) {
        int idx = best_inliers[i];
        odom_inliers.col(i) = all_points[idx].odom;
        gnss_inliers.col(i) = all_points[idx].gnss;
    }

    Eigen::MatrixXd T_opt = Eigen::umeyama(odom_inliers, gnss_inliers, false);
    if (T_opt.allFinite() == false || T_opt.rows() != 3) {
        return res;
    }

    Eigen::Matrix2d R_opt = T_opt.block<2, 2>(0, 0);
    Eigen::Vector2d t_opt = T_opt.block<2, 1>(0, 2);

    double meas_yaw = std::atan2(R_opt(1, 0), R_opt(0, 0));

    // Smooth update or initialize
    if (!is_aligned_) {
        current_yaw_ = meas_yaw;
        current_translation_ = t_opt;
        is_aligned_ = true;
    } else {
        // Check for jumps
        double yaw_diff = normalize_angle(meas_yaw - current_yaw_);
        double trans_diff = (t_opt - current_translation_).norm();

        if (std::abs(yaw_diff) < cfg_.max_yaw_jump_rad && trans_diff < cfg_.max_trans_jump_m) {
            // Apply EMA filter
            current_yaw_ = normalize_angle(current_yaw_ + cfg_.filter_gain * yaw_diff);
            current_translation_ += cfg_.filter_gain * (t_opt - current_translation_);
        }
        // If jump is too large, we ignore it to prevent trajectory discontinuity
        // But if it persists, one might add logic to re-initialize (omitted for smoothness priority)
    }

    res.success = true;
    res.inlier_count = best_inliers.size();
    res.yaw = current_yaw_;
    res.translation = current_translation_;

    return res;
}

bool DynamicAligner::isAligned() const { return is_aligned_; }

Eigen::Vector3d DynamicAligner::transform(const Eigen::Vector3d& odom_enu) const {
    if (!is_aligned_) return odom_enu;
    
    Eigen::Vector2d odom_xy(odom_enu.x(), odom_enu.y());
    
    double c = std::cos(current_yaw_);
    double s = std::sin(current_yaw_);
    Eigen::Matrix2d R;
    R << c, -s,
         s,  c;
         
    Eigen::Vector2d gnss_xy = R * odom_xy + current_translation_;
    
    return Eigen::Vector3d(gnss_xy.x(), gnss_xy.y(), odom_enu.z()); // keep Z unchanged, or it can be fused separately
}

void DynamicAligner::reset() {
    is_aligned_ = false;
    buffer_.clear();
    anchor_buffer_.clear();
    current_yaw_ = 0.0;
    current_translation_.setZero();
}

double DynamicAligner::getCurrentYaw() const { return current_yaw_; }
Eigen::Vector2d DynamicAligner::getCurrentTranslation() const { return current_translation_; }
size_t DynamicAligner::getBufferSize() const { return buffer_.size(); }

double DynamicAligner::normalize_angle(double rad) const {
    while (rad > M_PI) rad -= 2.0 * M_PI;
    while (rad < -M_PI) rad += 2.0 * M_PI;
    return rad;
}

} // namespace gnss_comm_impl
