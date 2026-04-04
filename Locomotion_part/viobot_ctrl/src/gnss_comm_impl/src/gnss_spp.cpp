/**
* This file is part of gnss_comm.
*
* Copyright (C) 2021 Aerial Robotics Group, Hong Kong University of Science and Technology
* Author: CAO Shaozu (shaozu.cao@gmail.com)
*
* gnss_comm is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* gnss_comm is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with gnss_comm. If not, see <http://www.gnu.org/licenses/>.
*/

#include "gnss_comm_impl/gnss_spp.hpp"
#include "gnss_comm_impl/gnss_utility.hpp"
#include <glog/logging.h>

#define CUT_OFF_DEGREE 15.0

namespace gnss_comm
{
    void filter_L1(const std::vector<ObsPtr> &obs, const std::vector<EphemBasePtr> &ephems, 
        std::vector<ObsPtr> &L1_obs, std::vector<EphemBasePtr> &L1_ephems)
    {
        L1_obs.clear();
        L1_ephems.clear();
        for (size_t i = 0; i < obs.size(); ++i)
        {
            const ObsPtr &this_obs = obs[i];
            // check system
            const uint32_t sys = satsys(this_obs->sat, NULL);
            if (sys != SYS_GPS && sys != SYS_GLO && sys != SYS_BDS && sys != SYS_GAL)
                continue;
            // check signal frequency
            const double obs_freq = L1_freq(this_obs, NULL);
            if (obs_freq < 0)   continue;

            L1_obs.push_back(this_obs);
            L1_ephems.push_back(ephems[i]);
        }
    }

    std::vector<SatStatePtr> sat_states(const std::vector<ObsPtr> &obs, 
        const std::vector<EphemBasePtr> &ephems)
    {
        std::vector<SatStatePtr> all_sv_states;
        const uint32_t num_obs = obs.size();
        for (size_t i = 0; i < num_obs; ++i)
        {
            SatStatePtr sat_state(new SatState());
            all_sv_states.push_back(sat_state);

            ObsPtr this_obs = obs[i];
            const uint32_t sat = this_obs->sat;
            const uint32_t sys = satsys(sat, NULL);
            int l1_idx = -1;
            L1_freq(this_obs, &l1_idx);
            if (l1_idx < 0)   continue;
            double tof = this_obs->psr[l1_idx] / LIGHT_SPEED;

            gtime_t sv_tx = time_add(this_obs->time, -tof);
            double svdt = 0, svddt = 0;
            Eigen::Vector3d sv_pos = Eigen::Vector3d::Zero();
            Eigen::Vector3d sv_vel = Eigen::Vector3d::Zero();
            if (sys == SYS_GLO)
            {
                GloEphemPtr glo_ephem = std::dynamic_pointer_cast<GloEphem>(ephems[i]);
                svdt = geph2svdt(sv_tx, glo_ephem);
                sv_tx = time_add(sv_tx, -svdt);
                sv_pos = geph2pos(sv_tx, glo_ephem, &svdt);
                sv_vel = geph2vel(sv_tx, glo_ephem, &svddt);
            }
            else
            {
                EphemPtr ephem = std::dynamic_pointer_cast<Ephem>(ephems[i]);
                svdt = eph2svdt(sv_tx, ephem);
                sv_tx = time_add(sv_tx, -svdt);
                sv_pos = eph2pos(sv_tx, ephem, &svdt);
                sv_vel = eph2vel(sv_tx, ephem, &svddt);
                sat_state->tgd = ephem->tgd[0];
            }
            sat_state->sat_id = sat;
            sat_state->ttx    = sv_tx;
            sat_state->pos    = sv_pos;
            sat_state->vel    = sv_vel;
            sat_state->dt     = svdt;
            sat_state->ddt    = svddt;
        }
        return all_sv_states;
    }

    void psr_res(const Eigen::Matrix<double, 7, 1> &rcv_state, const std::vector<ObsPtr> &obs, 
        const std::vector<SatStatePtr> &all_sv_states, const std::vector<double> &iono_params, 
        Eigen::VectorXd &res, Eigen::MatrixXd &J, std::vector<Eigen::Vector2d> &atmos_delay, 
        std::vector<Eigen::Vector2d> &all_sv_azel)
    {
        const uint32_t num_sv = all_sv_states.size();
        // clear output
        res = Eigen::VectorXd::Zero(num_sv);
        J = Eigen::MatrixXd::Zero(num_sv, 7);
        atmos_delay.resize(num_sv);
        all_sv_azel.resize(num_sv);

        for (uint32_t i = 0; i < num_sv; ++i)
        {
            int l1_idx = -1;
            L1_freq(obs[i], &l1_idx);
            if (l1_idx < 0)   continue;

            const SatStatePtr &sat_state = all_sv_states[i];
            uint32_t this_sys = satsys(sat_state->sat_id, NULL);
            Eigen::Vector3d sv_pos = sat_state->pos;

            double ion_delay=0, tro_delay=0;
            double azel[2] = {0, M_PI/2.0};
            if (rcv_state.topLeftCorner<3,1>().norm() > 0)
            {
                sat_azel(rcv_state.topLeftCorner<3,1>(), sv_pos, azel);
                Eigen::Vector3d rcv_lla = ecef2geo(rcv_state.head<3>());
                // use satellite signal transmit time instead
                tro_delay = calculate_trop_delay(sat_state->ttx, rcv_lla, azel);
                ion_delay = calculate_ion_delay(sat_state->ttx, iono_params, rcv_lla, azel);
            }

            Eigen::Vector3d rv2sv = sv_pos - rcv_state.topLeftCorner<3,1>();
            Eigen::Vector3d unit_rv2sv = rv2sv.normalized();
            double sagnac_term = EARTH_OMG_GPS*(sv_pos(0)*rcv_state(1,0)-
                sv_pos(1)*rcv_state(0,0))/LIGHT_SPEED;
            double psr_estimated = rv2sv.norm() + sagnac_term + rcv_state(3+sys2idx.at(this_sys)) -
             sat_state->dt*LIGHT_SPEED + tro_delay + ion_delay + sat_state->tgd*LIGHT_SPEED;

            J.block(i, 0, 1, 3) = -unit_rv2sv.transpose();
            J(i, 3+sys2idx.at(this_sys)) = 1.0;
            res(i) = psr_estimated - obs[i]->psr[l1_idx];

            atmos_delay[i] = Eigen::Vector2d(ion_delay, tro_delay);
            all_sv_azel[i] = Eigen::Vector2d(azel[0], azel[1]);
        }
    }

    
    Eigen::Matrix<double, 7, 1> psr_pos(const std::vector<ObsPtr> &obs, 
        const std::vector<EphemBasePtr> &ephems, const std::vector<double> &iono_params)
    {
        Eigen::Matrix<double, 7, 1> result;
        result.setZero();

        std::vector<ObsPtr> valid_obs;
        std::vector<EphemBasePtr> valid_ephems;
        filter_L1(obs, ephems, valid_obs, valid_ephems);

        // ================== RAIM 配置参数 ==================
        const double RAIM_RESIDUAL_THRESH = 15.0; // 残差剔除阈值（m）
        const int MAX_RAIM_EXCLUSIONS = 4;        // 单历元最大允许剔除的卫星数，防止误杀导致解算不可用
        int num_exclusions = 0;                   // 已剔除计数器
        // =================================================

        // 将位置坐标系初始化放在 RAIM 循环外，这样剔除卫星后可以利用上一次计算的坐标进行热启动，加速下一次收敛
        Eigen::Matrix<double, 7, 1> xyzt;
        xyzt.setZero(); 

        // RAIM 外部循环
        while (true)
        {
            if (valid_obs.size() < 4)
            {
                LOG(ERROR) << "[gnss_comm::psr_pos] GNSS observation not enough (after exclusions).\n";
                result.setZero();
                return result;
            }

            std::vector<SatStatePtr> all_sat_states = sat_states(valid_obs, valid_ephems);

            double dx_norm = 1.0;
            uint32_t num_iter = 0;

            // 加权最小二乘 (WLS) 内部循环
            while(num_iter < MAX_ITER_PVT && dx_norm > EPSILON_PVT)
            {
                Eigen::VectorXd b;
                Eigen::MatrixXd G;
                std::vector<Eigen::Vector2d> atmos_delay;
                std::vector<Eigen::Vector2d> all_sv_azel;
                psr_res(xyzt, valid_obs, all_sat_states, iono_params, b, G, atmos_delay, all_sv_azel);

                std::vector<uint32_t> good_idx;
                for (uint32_t i = 0; i < valid_obs.size(); ++i)
                {
                    if (G.row(i).norm() <= 0)   continue;       // res not computed
                    if (all_sv_azel[i].y() > CUT_OFF_DEGREE/180.0*M_PI)  
                        good_idx.push_back(i);
                }
                int sys_mask[4] = {0, 0, 0, 0};
                for (uint32_t i = 0; i < valid_obs.size(); ++i)
                {
                    const uint32_t obs_sys = satsys(valid_obs[i]->sat, NULL);
                    sys_mask[sys2idx.at(obs_sys)] = 1;
                }
                uint32_t num_extra_constraint = 4;
                for (uint32_t k = 0; k < 4; ++k)    num_extra_constraint -= sys_mask[k];
                LOG_IF(FATAL, num_extra_constraint >= 4) << "[gnss_comm::psr_pos] too many extra-clock constraints.\n";

                const uint32_t good_num = good_idx.size();
                // 在 RAIM 迭代中，如果有太多的低高度角卫星被砍掉导致好星不足 4 颗，应当直接退出
                if (good_num < 4) {
                    LOG(ERROR) << "[gnss_comm::psr_pos] too few good obs: " << good_num;
                    result.setZero();
                    return result;
                }

                Eigen::MatrixXd good_G(good_num+num_extra_constraint, 7);
                Eigen::VectorXd good_b(good_num+num_extra_constraint);
                Eigen::MatrixXd good_W(good_num+num_extra_constraint, good_num+num_extra_constraint);
                good_W.setZero();
                for (uint32_t i = 0; i < good_num; ++i)
                {
                    const uint32_t origin_idx = good_idx[i];
                    const ObsPtr &this_obs = valid_obs[origin_idx];
                    good_G.row(i) = G.row(origin_idx);
                    good_b(i) = b(origin_idx);
                    // compose weight
                    // 高度角 + 观测噪声 联合加权
                    const double el = all_sv_azel[origin_idx].y();
                    double sin_el = std::sin(el);

                    int l1_idx = -1;
                    L1_freq(this_obs, &l1_idx);
                    LOG_IF(FATAL, l1_idx < 0) << "[gnss_comm::psr_pos] no L1 observation found.\n";

                    // 高度角方差模型
                    // 采用经典模型: a^2 / sin^2(el)。低高度角赋予更大的方差
                    double el_var = (0.5 * 0.5) / (sin_el * sin_el + 0.05);

                    // 观测噪声方差
                    double std_obs = this_obs->psr_std[l1_idx];
                    if (std_obs <= 0) std_obs = 1.0; // 默认 1m 的先验测距白噪声
                    double obs_var = std_obs * std_obs;

                    // 广播星历误差方差 (URA)
                    double raw_ura = valid_ephems[origin_idx]->ura;
                    double ura_std = (raw_ura >= 0.5 && raw_ura < 50.0) ? raw_ura : 2.0;
                    const uint32_t obs_sys = satsys(this_obs->sat, NULL);
                    if (obs_sys == SYS_GAL) ura_std = std::max(ura_std - 1.0, 1.0);
                    if (obs_sys == SYS_GLO) ura_std = std::max(ura_std, 3.0);
                    double ura_var = ura_std * ura_std;

                    // 总方差叠加与定权
                    double total_var = obs_var + el_var + ura_var;
                    total_var = std::max(total_var, 1e-4); // 防止数值异常下溢

                    good_W(i, i) = 1.0 / total_var;
                }
                uint32_t tmp_count = good_num;
                // add extra pseudo measurement to contraint unobservable clock bias
                for (size_t k = 0; k < 4; ++k)
                {
                    if (!sys_mask[k])
                    {
                        good_G.row(tmp_count).setZero();
                        good_G(tmp_count, k+3) = 1.0;
                        good_b(tmp_count) = 0;
                        good_W(tmp_count, tmp_count) = 1000;       // large weight
                        ++tmp_count;
                    }
                }

                // ready for solving
                // Eigen::VectorXd dx = -(good_G.transpose()*good_W*good_G).inverse() * good_G.transpose() * good_W * good_b;
                Eigen::MatrixXd H = good_G.transpose() * good_W * good_G;
                Eigen::VectorXd g = good_G.transpose() * good_W * good_b;
                Eigen::VectorXd dx = -H.colPivHouseholderQr().solve(g);
                dx_norm = dx.norm();
                xyzt += dx;
                ++num_iter;
            }

            if (num_iter == MAX_ITER_PVT)
            {
                LOG(WARNING) << "[gnss_comm::psr_pos] XYZT solver reached maximum iterations.\n";
                // 迭代不收敛，直接返回错误，不进行后续完好性检测
                result.setZero();
                return result;
            }

            // =========================================================================
            //  WLS 收敛后执行 RAIM 后验残差检验
            // =========================================================================

            // 必须要有冗余观测数据( >= 5 颗星)才能进行一致性检验
            if (valid_obs.size() < 5 || num_exclusions >= MAX_RAIM_EXCLUSIONS) 
            {
                result = xyzt;
                return result;
            }

            // 基于当前已收敛的 xyzt 获取最终的精确后验残差
            Eigen::VectorXd b_post;
            Eigen::MatrixXd G_post;
            std::vector<Eigen::Vector2d> atmos_delay_post;
            std::vector<Eigen::Vector2d> all_sv_azel_post;
            psr_res(xyzt, valid_obs, all_sat_states, iono_params, b_post, G_post, atmos_delay_post, all_sv_azel_post);

            double max_residual = -1.0;
            int max_res_idx = -1;

            // 查找产生最大残差的卫星
            for (uint32_t i = 0; i < valid_obs.size(); ++i)
            {
                // 忽略未通过几何检验的卫星（防越界或除 0）
                if (G_post.row(i).norm() <= 0) continue;
                if (all_sv_azel_post[i].y() <= CUT_OFF_DEGREE/180.0*M_PI) continue;

                // 由于该源码实现中，b 实际上表示： Calc_Range - Obs_Range
                // 所以绝对值即代表当前的伪距拟合残差
                double current_res = std::abs(b_post(i));
                
                if (current_res > max_residual) {
                    max_residual = current_res;
                    max_res_idx = i;
                }
            }

            // 判断是否触发剔除机制
            if (max_res_idx != -1 && max_residual > RAIM_RESIDUAL_THRESH)
            {
                uint32_t faulty_sat = valid_obs[max_res_idx]->sat;
                uint32_t faulty_sys = satsys(faulty_sat, NULL);
                
                LOG(INFO) << "[gnss_comm::psr_pos] RAIM FAULT DETECTED: Excluded Sat " 
                        << faulty_sat << " (Sys " << faulty_sys << ") "
                        << "| Post-fit Residual: " << std::fixed << std::setprecision(2) 
                        << max_residual << " m.";

                // 剔除异常卫星，并进入下一轮 while 循环重新解算
                valid_obs.erase(valid_obs.begin() + max_res_idx);
                valid_ephems.erase(valid_ephems.begin() + max_res_idx);
                
                num_exclusions++;
                
                // 注意：我们这里不重置 xyzt，因为直接使用上一次包含微小偏差的坐标作为热启动，
                // 重新剔除残差后，二次求解将会在极少迭代次数内收敛。
            }
            else
            {
                // 所有可用卫星的残差均低于阈值，RAIM 检验安全通过！
                result = xyzt;
                break;
            }
        }

        return result;
    }
    void dopp_res(const Eigen::Matrix<double, 4, 1> &rcv_state, const Eigen::Vector3d &rcv_ecef,
                  const std::vector<ObsPtr> &obs, const std::vector<SatStatePtr> &all_sv_states, 
                  Eigen::VectorXd &res, Eigen::MatrixXd &J)
    {
        const uint32_t num_sv = all_sv_states.size();
        //clear output
        res = Eigen::VectorXd::Zero(num_sv);
        J = Eigen::MatrixXd::Zero(num_sv, 4);

        for (uint32_t i = 0; i < num_sv; ++i)
        {
            const SatStatePtr &sat_state = all_sv_states[i];
            Eigen::Vector3d unit_rv2sv = (sat_state->pos - rcv_ecef).normalized();
            double sagnac_term = EARTH_OMG_GPS/LIGHT_SPEED*(
                sat_state->vel(0)*rcv_ecef(1)+ sat_state->pos(0)*rcv_state(1,0) - 
                sat_state->vel(1)*rcv_ecef(0) - sat_state->pos(1)*rcv_state(0,0));
            double dopp_estimated = (sat_state->vel - rcv_state.topLeftCorner<3, 1>()).dot(unit_rv2sv) + 
                    rcv_state(3, 0) + sagnac_term - sat_state->ddt*LIGHT_SPEED;
            int l1_idx = -1;
            const double obs_freq = L1_freq(obs[i], &l1_idx);
            if (obs_freq < 0)   continue;
            const double wavelength = LIGHT_SPEED / obs_freq;
            res(i) = dopp_estimated + obs[i]->dopp[l1_idx]*wavelength;
            J.block(i, 0, 1, 3) = -1.0 * unit_rv2sv.transpose();
            J(i, 3) = 1.0;
        }
    }

    Eigen::Matrix<double, 4, 1> dopp_vel(const std::vector<ObsPtr> &obs, 
        const std::vector<EphemBasePtr> &ephems, Eigen::Vector3d &ref_ecef)
    {
        LOG(INFO) << "try to solve Doppler velocity.";
        Eigen::Matrix<double, 4, 1> result;
        result.setZero();

        if (ref_ecef.norm() == 0)
        {
            // reference point not given, try calculate using pseudorange
            std::vector<double> zero_iono_params(8, 0.0);
            Eigen::Matrix<double, 7, 1> psr_result = psr_pos(obs, ephems, zero_iono_params);
            if (psr_result.head<3>().norm() != 0)
            {
                ref_ecef = psr_result.head<3>();
            }
            else
            {
                LOG(ERROR) << "[gnss_comm::dopp_vel] Unable to initialize reference position for Doppler calculation.\n";
                return result;
            }
        }

        std::vector<ObsPtr> valid_obs;
        std::vector<EphemBasePtr> valid_ephems;
        filter_L1(obs, ephems, valid_obs, valid_ephems);

        if (valid_obs.size() < 4)
        {
            LOG(ERROR) << "[gnss_comm::dopp_vel] GNSS observation not enough for velocity calculation.\n";
            return result;
        }

        std::vector<SatStatePtr> all_sat_states = sat_states(valid_obs, valid_ephems);
        // compute azel for all satellite
        std::vector<Eigen::Vector2d> all_sv_azel;
        for (uint32_t i = 0; i < all_sat_states.size(); ++i)
        {
            double azel[2] = {0, 0};
            sat_azel(ref_ecef, all_sat_states[i]->pos, azel);
            all_sv_azel.emplace_back(azel[0], azel[1]);
        }

        Eigen::Matrix<double, 4, 1> xyzt_dot;
        xyzt_dot.setZero();
        double dx_norm = 1.0;
        uint32_t num_iter = 0;
        while(num_iter < MAX_ITER_PVT && dx_norm > EPSILON_PVT)
        {
            Eigen::MatrixXd G;
            Eigen::VectorXd b;
            dopp_res(xyzt_dot, ref_ecef, valid_obs, all_sat_states, b, G);
            
            std::vector<uint32_t> good_idx;
            for (uint32_t i = 0; i < valid_obs.size(); ++i)
            {
                if (G.row(i).norm() <= 0)   continue;       // res not computed
                if (all_sv_azel[i].y() > CUT_OFF_DEGREE/180.0*M_PI)  
                    good_idx.push_back(i);
            }
            const uint32_t good_num = good_idx.size();
            Eigen::MatrixXd good_G = Eigen::MatrixXd::Zero(good_num, 4);
            Eigen::VectorXd good_b = Eigen::VectorXd::Zero(good_num);
            Eigen::MatrixXd good_W = Eigen::MatrixXd::Zero(good_num, good_num);
            for (uint32_t i = 0; i < good_num; ++i)
            {
                const uint32_t origin_idx = good_idx[i];
                good_G.row(i) = G.row(origin_idx);
                good_b(i) = b(origin_idx);
                const double sin_el = sin(all_sv_azel[origin_idx].y());
                double weight = sin_el*sin_el;
                const ObsPtr &this_obs = valid_obs[origin_idx];
                int l1_idx = -1;
                L1_freq(this_obs, &l1_idx);
                LOG_IF(FATAL, l1_idx < 0) << "[gnss_comm::dopp_vel] no L1 observation found.\n";
                if (this_obs->dopp_std[l1_idx] > 0)
                    weight /= (this_obs->dopp_std[l1_idx]/0.256);

                const uint32_t obs_sys = satsys(this_obs->sat, NULL);
                if (obs_sys == SYS_GPS || obs_sys == SYS_BDS)
                    weight /= valid_ephems[origin_idx]->ura-1;
                else if (obs_sys == SYS_GAL)
                    weight /= valid_ephems[origin_idx]->ura-2;
                else if (obs_sys == SYS_GLO)
                    weight /= 2;
                good_W(i, i) = weight;
            }

            // solve
            // Eigen::VectorXd dx = -(good_G.transpose()*good_W*good_G).inverse() * good_G.transpose() * good_W * good_b;
            Eigen::MatrixXd H = good_G.transpose() * good_W * good_G;
            Eigen::VectorXd g = good_G.transpose() * good_W * good_b;
            Eigen::VectorXd dx = -H.colPivHouseholderQr().solve(g);
            dx_norm = dx.norm();
            xyzt_dot += dx;

            // dx_norm = 0;
            // LOG(INFO) << "cov is \n" << (G.transpose()*W*G).inverse();
            ++num_iter;
        }

        if (num_iter == MAX_ITER_PVT)
            LOG(WARNING) << "[gnss_comm::dopp_vel] XYZT solver reached maximum iterations.\n";
        
        result = xyzt_dot;
        return result;
    }
}