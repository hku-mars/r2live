#pragma once
#include <ros/assert.h>
#include <iostream>
#include <eigen3/Eigen/Dense>

#include "../utility/utility.h"
#include "../parameters.h"
#include "integration_base.h"

#include <ceres/ceres.h>
#include "common_lib.h"


// ANCHOR:  LiDAR prior factor here.
class LiDAR_prior_factor : public ceres::SizedCostFunction<6, 7>
{
  public:
    eigen_q m_prior_q=  eigen_q::Identity();
    vec_3 m_prior_t=  vec_3::Zero();
    LiDAR_prior_factor() = delete;
    StatesGroup* m_lio_prior_state;
    
    LiDAR_prior_factor(StatesGroup* lio_prior_state):m_lio_prior_state(lio_prior_state)
    {
        m_prior_q = eigen_q(m_lio_prior_state->rot_end);
        m_prior_t = m_lio_prior_state->pos_end;
    }
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {

        Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
        Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

        double w_s = 0.1;

        Eigen::Map<Eigen::Matrix<double, 6, 1>> residual(residuals);
        Eigen::Matrix<double, 6, 6> cov_mat_temp =  m_lio_prior_state->cov.block(0,0,6,6);
        cov_mat_temp.block(0,0,3,6).swap(cov_mat_temp.block(3,0,3,6));
        cov_mat_temp.block(0,0,6,3).swap(cov_mat_temp.block(0,3,6,3));
    
        Eigen::Matrix<double, 6, 6> sqrt_info = Eigen::LLT<Eigen::Matrix<double, 6, 6>>(cov_mat_temp.inverse()).matrixL().transpose();
       
        residual.block(0, 0, 3, 1) = (Pi - m_prior_t) * w_s;
        residual.block(3, 0, 3, 1) = Sophus::SO3d( (m_prior_q.inverse() * (Qi)) ).log() * w_s;
        
        residual = sqrt_info * residual;

        if (jacobians)
        {
            if (jacobians[0])
            {
                Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor>> jacobian_pose_i(jacobians[0]);
                jacobian_pose_i.setZero();

                jacobian_pose_i.block<3, 3>(0, 0) =  mat_3_3::Identity();
                jacobian_pose_i.block<3, 3>(O_R, O_R) = inverse_right_jacobian_of_rotion_matrix (Sophus::SO3d( (m_prior_q.inverse() * (Qi)) ).log() );
                jacobian_pose_i = w_s  * sqrt_info * jacobian_pose_i;

                if (jacobian_pose_i.maxCoeff() > 1e8 || jacobian_pose_i.minCoeff() < -1e8)
                {
                    ROS_WARN("numerical unstable in preintegration");
                }
            }
        }
        return true;
    }
};
#if 1
// ANCHOR:  LiDAR prior factor here.
class LiDAR_prior_factor_15 : public ceres::SizedCostFunction<15, 7, 9>
{
  public:
    eigen_q m_prior_q=  eigen_q::Identity();
    vec_3 m_prior_t=  vec_3::Zero();
    Eigen::Matrix<double, 9, 1> m_prior_speed_bias ;
    LiDAR_prior_factor_15() = delete;
    StatesGroup* m_lio_prior_state;
    
    LiDAR_prior_factor_15(StatesGroup* lio_prior_state): m_lio_prior_state(lio_prior_state)
    {
        m_prior_q = eigen_q(m_lio_prior_state->rot_end);
        m_prior_t = m_lio_prior_state->pos_end;
        m_prior_speed_bias.block(0,0,3,1) = m_lio_prior_state->vel_end;
        m_prior_speed_bias.block(3,0,3,1) = m_lio_prior_state->bias_a;
        m_prior_speed_bias.block(6,0,3,1) = m_lio_prior_state->bias_g;
    }

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {

        Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
        Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);
        Eigen::Matrix<double, 9 , 1> speed_bias;
        speed_bias << parameters[1][0], parameters[1][1], parameters[1][2],
            parameters[1][3], parameters[1][4], parameters[1][5],
            parameters[1][6], parameters[1][7], parameters[1][8];
        double w_s = 0.1;

        Eigen::Map<Eigen::Matrix<double, 15, 1>> residual(residuals);
        Eigen::Matrix<double, 15, 15> cov_mat_temp;

        cov_mat_temp = m_lio_prior_state->cov.block(0, 0, 15, 15);

        cov_mat_temp.block(0, 0, 3, 15).swap(cov_mat_temp.block(3, 0, 3, 15));
        cov_mat_temp.block(9, 0, 3, 15).swap(cov_mat_temp.block(12, 0, 3, 15));

        cov_mat_temp.block(0, 0, 15, 3).swap(cov_mat_temp.block(0, 3, 15, 3));
        cov_mat_temp.block(0, 9, 15, 3).swap(cov_mat_temp.block(0, 12, 15, 3));

        Eigen::Matrix<double, 15, 15> sqrt_info = Eigen::LLT<Eigen::Matrix<double, 15, 15>>(cov_mat_temp.inverse()).matrixL().transpose();

        residual.block(0, 0, 3, 1) = (Pi - m_prior_t) * w_s;
        residual.block(3, 0, 3, 1) = Sophus::SO3d( (m_prior_q.inverse() * (Qi)) ).log() * w_s;
        residual.block(6, 0, 9, 1) = (speed_bias -  m_prior_speed_bias ) * w_s;

        residual = sqrt_info * residual;

        if (jacobians)
        {
            if (jacobians[0])
            {
                Eigen::Map<Eigen::Matrix<double, 15, 7, Eigen::RowMajor>> jacobian_pose_i(jacobians[0]);
                jacobian_pose_i.setZero();

                jacobian_pose_i.block<3, 3>(0, 0) =  mat_3_3::Identity();
                jacobian_pose_i.block<3, 3>(3, 3) = inverse_right_jacobian_of_rotion_matrix (Sophus::SO3d( (m_prior_q.inverse() * (Qi)) ).log() );
                
                jacobian_pose_i = w_s  * sqrt_info * jacobian_pose_i;
                
                if (jacobian_pose_i.maxCoeff() > 1e8 || jacobian_pose_i.minCoeff() < -1e8)
                {
                    ROS_WARN("numerical unstable in preintegration");
                }
            }

            if (jacobians[1])
            {

                Eigen::Map<Eigen::Matrix<double, 15, 9, Eigen::RowMajor>> jacobian_pose_i(jacobians[1]);
                jacobian_pose_i.setZero();
                jacobian_pose_i.block(6, 0, 9, 9).setIdentity();
                jacobian_pose_i = w_s * sqrt_info * jacobian_pose_i;
                if (jacobian_pose_i.maxCoeff() > 1e8 || jacobian_pose_i.minCoeff() < -1e8)
                {
                    ROS_WARN("numerical unstable in preintegration");
                }

            }
        }
        return true;
    }
};
#endif;