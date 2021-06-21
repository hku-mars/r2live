#include "estimator.h"
#include <Eigen/Cholesky>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include "tools_ceres.hpp"
#include "LM_Strategy.hpp"

#define USING_CERES_HUBER 0
extern Camera_Lidar_queue g_camera_lidar_queue;
extern MeasureGroup Measures;
extern StatesGroup g_lio_state;
Estimator *g_estimator;
ceres::LossFunction *g_loss_function;
int g_extra_iterations = 0;

// ANCHOR - huber_loss
void huber_loss(Eigen::Matrix<double, -1, 1> residual, double  & residual_scale, double & jacobi_scale,
                 double outlier_threshold = 1.0  )
{
    // http://ceres-solver.org/nnls_modeling.html#lossfunction
    double res_norm = residual.norm()  ;
    if( res_norm / outlier_threshold < 1.0 )
    {
        residual_scale = 1.0;
        jacobi_scale = 1.0;
    }
    else
    {
        residual_scale =  (2 * sqrt(res_norm) / sqrt(outlier_threshold) - 1.0) / res_norm ;
        jacobi_scale  = residual_scale;
    }
}

struct IMU_factor_res
{
    Eigen::Matrix<double, 15, 1> m_residual;
    double data_buffer[4][150];
    std::vector<Eigen::Matrix<double, -1, -1, Eigen::RowMajor|Eigen::DontAlign>> m_jacobian_mat_vec;
    double *m_jacobian_addr_vector[4];
    double *m_residual_addr;
    double *m_parameters_vector[4];
    int m_index_i;
    int m_index_j;
    IMUFactor *m_imu_factor;
    Estimator *m_estimator;
    void init()
    {
        if (m_jacobian_mat_vec.size() == 0)
        {
            m_jacobian_mat_vec.resize(4);
            m_jacobian_mat_vec[0].resize(15, 7);
            m_jacobian_mat_vec[1].resize(15, 9);
            m_jacobian_mat_vec[2].resize(15, 7);
            m_jacobian_mat_vec[3].resize(15, 9);
        }
        m_residual.setZero();
        m_residual_addr = m_residual.data();
        for (int i = 0; i < m_jacobian_mat_vec.size(); i++)
        {
            m_jacobian_mat_vec[i].setZero();
            m_jacobian_addr_vector[i] = m_jacobian_mat_vec[i].data();
        }
    };

    IMU_factor_res()
    {
        init();
    };

    void add_keyframe_to_keyframe_factor(Estimator *estimator, IMUFactor *imu_factor, const int &index_i,
                                         const int &index_j)
    {
        m_index_i = index_i;
        m_index_j = index_j;
        m_estimator = estimator;
       
        m_parameters_vector[0] = m_estimator->m_para_Pose[m_index_i];
        m_parameters_vector[1] = m_estimator->m_para_SpeedBias[m_index_i];
        m_parameters_vector[2] = m_estimator->m_para_Pose[m_index_j];
        m_parameters_vector[3] = m_estimator->m_para_SpeedBias[m_index_j];

        m_imu_factor = imu_factor;
        init();
    }

    void Evaluate()
    {
        init();
        m_imu_factor->Evaluate(m_parameters_vector, m_residual_addr, m_jacobian_addr_vector);
    }
};

// ANCHOR - LiDAR prior factor
struct L_prior_factor
{
    Eigen::Matrix<double, -1, 1> m_residual;
    std::vector<Eigen::Matrix<double, -1, -1, Eigen::RowMajor>> m_jacobian_mat_vec;
    int m_index_i;
    double *m_jacobian_addr_vector[5];
    double *m_residual_addr;
    double *m_parameters_vector[5];
    // LiDAR_prior_factor * m_lidar_prior_factor;
    LiDAR_prior_factor_15 * m_lidar_prior_factor;
    Estimator *m_estimator;
    void init()
    {
        if (m_jacobian_mat_vec.size() == 0)
        {
            m_jacobian_mat_vec.resize(2);
            m_jacobian_mat_vec[0].resize(15, 7);
            m_jacobian_mat_vec[1].resize(15, 9);
            m_residual.resize(15);
        }
        m_residual.setZero();
        m_residual_addr = m_residual.data();
        for(int i =0 ; i  < m_jacobian_mat_vec.size(); i++)
        {
            m_jacobian_mat_vec[i].setZero();
            m_jacobian_addr_vector[i] = m_jacobian_mat_vec[i].data();
        }
    }

    L_prior_factor() =default;

    void add_lidar_prior_factor(Estimator *estimator, LiDAR_prior_factor_15 *lidar_prior_factor, const int &index_i)
    {
        m_estimator = estimator;
        m_lidar_prior_factor = lidar_prior_factor;
        m_index_i = index_i;
        m_parameters_vector[0] = m_estimator->m_para_Pose[m_index_i];
        m_parameters_vector[1] = m_estimator->m_para_SpeedBias[m_index_i];
    }

    void Evaluate()
    {
        init();
        m_lidar_prior_factor->Evaluate(m_parameters_vector, m_residual_addr, m_jacobian_addr_vector);
    }
};

// ANCHOR - Keypoint_projection_factor
struct Keypoint_projection_factor
{
    Eigen::Matrix<double, -1, 1> m_residual;
    std::vector<Eigen::Matrix<double, -1, -1, Eigen::RowMajor>> m_jacobian_mat_vec;
    int m_index_i;
    int m_index_j;
    int m_feature_index;
    double *m_jacobian_addr_vector[5];
    double *m_residual_addr;
    double *m_parameters_vector[5];
    ProjectionTdFactor *m_projection_factor;
    Estimator *m_estimator;
    void init()
    {
        if (m_jacobian_mat_vec.size() == 0)
        {
            m_jacobian_mat_vec.resize(5);
            m_jacobian_mat_vec[0].resize(2, 7);
            m_jacobian_mat_vec[1].resize(2, 7);
            m_jacobian_mat_vec[2].resize(2, 7);
            m_jacobian_mat_vec[3].resize(2, 1);
            m_jacobian_mat_vec[4].resize(2, 1);
        }
        m_residual.resize(2);
        m_residual.setZero();
        m_residual_addr = m_residual.data();
        for(int i =0 ; i  < m_jacobian_mat_vec.size(); i++)
        {
            m_jacobian_mat_vec[i].setZero();
            m_jacobian_addr_vector[i] = m_jacobian_mat_vec[i].data();
        }
    };

    Keypoint_projection_factor() = default;
    ~Keypoint_projection_factor() = default;

    void add_projection_factor(Estimator *estimator, ProjectionTdFactor *projection_factor, const int &index_i,
                               const int &index_j, const int &feature_idx)
    {
        m_estimator = estimator;
        m_index_i = index_i;
        m_index_j = index_j;
        m_feature_index = feature_idx;
        m_projection_factor = projection_factor;
        m_parameters_vector[0] = m_estimator->m_para_Pose[m_index_i];
        m_parameters_vector[1] = m_estimator->m_para_Pose[m_index_j];
        m_parameters_vector[2] = m_estimator->m_para_Ex_Pose[0];
        m_parameters_vector[3] = m_estimator->m_para_Feature[m_feature_index];
        m_parameters_vector[4] = m_estimator->m_para_Td[0];
        init();
    }

    void Evaluate()
    {
        init();
        m_projection_factor->Evaluate(m_parameters_vector, m_residual_addr, m_jacobian_addr_vector);
        
        if (USING_CERES_HUBER)
        {
            Common_tools::apply_ceres_loss_fun(g_loss_function, m_residual, m_jacobian_mat_vec);
        }
        else
        {
            double res_scale, jacobi_scale;
            huber_loss(m_residual, res_scale, jacobi_scale, 0.5);
            if (res_scale != 1.0)
            {
                m_residual *= res_scale;
                for (int i = 0; i < m_jacobian_mat_vec.size(); i++)
                {
                    m_jacobian_mat_vec[i] *= jacobi_scale;
                }
            }
        }
        
    }
};

// ANCHOR -  Marginalization_factor
struct Marginalization_factor
{
    Eigen::Matrix<double, -1, 1> m_residual, m_residual_new;
    std::vector<Eigen::Matrix<double, -1, -1, Eigen::RowMajor>> m_jacobian_mat_vec;
    int m_index_i;
    int m_index_j;
    int m_feature_index;
    double *m_jacobian_addr_vector[100];
    double *m_residual_addr;
    double *m_parameters_vector[100];
    int m_residual_block_size;
    std::vector<int> m_jacobian_pos;

    int m_margin_res_size;
    std::vector<int> m_margin_res_pos_vector;
    std::vector<int> m_margin_res_pos_parameters_size;
    vio_marginalization * m_vio_margin_ptr = nullptr;
    Marginalization_factor()
    {
        m_margin_res_size = 0;
    }
    
    ~Marginalization_factor() = default;
    
    void Evaluate_mine(Eigen::VectorXd & residual_vec,  Eigen::MatrixXd & jacobian_matrix)
    {
         scope_color(ANSI_COLOR_CYAN_BOLD);
        int margin_residual_size = m_vio_margin_ptr->m_linearized_jacobians.rows();
        Eigen::VectorXd diff_x(margin_residual_size);
        Eigen::Quaterniond temp_Q;
        Eigen::Vector3d temp_t;

        m_residual_new.resize(margin_residual_size);
        m_residual_new.setZero();
        diff_x.setZero();

        temp_Q = Eigen::Quaterniond(g_estimator->m_para_Pose[0][6], g_estimator->m_para_Pose[0][3], g_estimator->m_para_Pose[0][4], g_estimator->m_para_Pose[0][5]).normalized();
        int pos = 15;
        if (m_vio_margin_ptr->m_margin_flag == 0) // mar oldest
        {
            // pose[0] speed_bias[0]
            diff_x.block(0, 0, 3, 1) = Eigen::Vector3d(g_estimator->m_para_Pose[0][0], g_estimator->m_para_Pose[0][1], g_estimator->m_para_Pose[0][2]) - m_vio_margin_ptr->m_Ps[1];
            diff_x.block(3, 0, 3, 1) = Sophus::SO3d(m_vio_margin_ptr->m_Rs[1].transpose() * temp_Q.toRotationMatrix()).log();
            diff_x.block(6, 0, 3, 1) = Eigen::Vector3d(g_estimator->m_para_SpeedBias[0][0], g_estimator->m_para_SpeedBias[0][1], g_estimator->m_para_SpeedBias[0][2]) - m_vio_margin_ptr->m_Vs[1];
            diff_x.block(9, 0, 3, 1) = Eigen::Vector3d(g_estimator->m_para_SpeedBias[0][3], g_estimator->m_para_SpeedBias[0][4], g_estimator->m_para_SpeedBias[0][5]) - m_vio_margin_ptr->m_Bas[1];
            diff_x.block(12, 0, 3, 1) = Eigen::Vector3d(g_estimator->m_para_SpeedBias[0][6], g_estimator->m_para_SpeedBias[0][7], g_estimator->m_para_SpeedBias[0][8]) - m_vio_margin_ptr->m_Bgs[1];
            jacobian_matrix.block(0, 0, margin_residual_size, 15) = m_vio_margin_ptr->m_linearized_jacobians.block(0, 0, margin_residual_size, 15);

            for (int i = 1; i < WINDOW_SIZE; i++)
            {
                temp_t = Eigen::Vector3d(g_estimator->m_para_Pose[i][0], g_estimator->m_para_Pose[i][1], g_estimator->m_para_Pose[i][2]);
                temp_Q = Eigen::Quaterniond(g_estimator->m_para_Pose[i][6], g_estimator->m_para_Pose[i][3], g_estimator->m_para_Pose[i][4], g_estimator->m_para_Pose[i][5]).normalized();
                diff_x.block(pos, 0, 3, 1) = temp_t - Eigen::Vector3d(m_vio_margin_ptr->m_Ps[i + 1]);
                diff_x.block(pos + 3, 0, 3, 1) = Eigen::Vector3d(Sophus::SO3d(m_vio_margin_ptr->m_Rs[i + 1].transpose() * temp_Q.toRotationMatrix()).log());
                jacobian_matrix.block(0, i * 15, margin_residual_size, 6) = m_vio_margin_ptr->m_linearized_jacobians.block(0, pos, margin_residual_size, 6);
                pos += 6;
            }
        }
        else if (m_vio_margin_ptr->m_margin_flag == 1) // mar second new
        {
            // pose[0] speed_bias[0]
            diff_x.block(0, 0, 3, 1) = Eigen::Vector3d(g_estimator->m_para_Pose[0][0], g_estimator->m_para_Pose[0][1], g_estimator->m_para_Pose[0][2]) - m_vio_margin_ptr->m_Ps[0];
            diff_x.block(3, 0, 3, 1) = Sophus::SO3d(m_vio_margin_ptr->m_Rs[0].transpose() * temp_Q.toRotationMatrix()).log();
            diff_x.block(6, 0, 3, 1) = Eigen::Vector3d(g_estimator->m_para_SpeedBias[0][0], g_estimator->m_para_SpeedBias[0][1], g_estimator->m_para_SpeedBias[0][2]) - m_vio_margin_ptr->m_Vs[0];
            diff_x.block(9, 0, 3, 1) = Eigen::Vector3d(g_estimator->m_para_SpeedBias[0][3], g_estimator->m_para_SpeedBias[0][4], g_estimator->m_para_SpeedBias[0][5]) - m_vio_margin_ptr->m_Bas[0];
            diff_x.block(12, 0, 3, 1) = Eigen::Vector3d(g_estimator->m_para_SpeedBias[0][6], g_estimator->m_para_SpeedBias[0][7], g_estimator->m_para_SpeedBias[0][8]) - m_vio_margin_ptr->m_Bgs[0];
            jacobian_matrix.block(0, 0, margin_residual_size, 15) = m_vio_margin_ptr->m_linearized_jacobians.block(0, 0, margin_residual_size, 15);
            for (int i = 1; i < WINDOW_SIZE - 1; i++)
            {
                temp_t = Eigen::Vector3d(g_estimator->m_para_Pose[i][0], g_estimator->m_para_Pose[i][1], g_estimator->m_para_Pose[i][2]);
                temp_Q = Eigen::Quaterniond(g_estimator->m_para_Pose[i][6], g_estimator->m_para_Pose[i][3], g_estimator->m_para_Pose[i][4], g_estimator->m_para_Pose[i][5]).normalized();
                diff_x.block(pos, 0, 3, 1) = temp_t - Eigen::Vector3d(m_vio_margin_ptr->m_Ps[i]);
                diff_x.block(pos + 3, 0, 3, 1) = Eigen::Vector3d(Sophus::SO3d(m_vio_margin_ptr->m_Rs[i].transpose() * temp_Q.toRotationMatrix()).log());
                jacobian_matrix.block(0, i * 15, margin_residual_size, 6) = m_vio_margin_ptr->m_linearized_jacobians.block(0, pos, margin_residual_size, 6);
                pos += 6;
            }
        }

        temp_t = Eigen::Vector3d(g_estimator->m_para_Ex_Pose[0][0], g_estimator->m_para_Ex_Pose[0][1], g_estimator->m_para_Ex_Pose[0][2]);
        temp_Q = Eigen::Quaterniond(g_estimator->m_para_Ex_Pose[0][6], g_estimator->m_para_Ex_Pose[0][3], g_estimator->m_para_Ex_Pose[0][4], g_estimator->m_para_Ex_Pose[0][5]);
        diff_x.block(pos, 0, 3, 1) = temp_t - m_vio_margin_ptr->m_tic[0];
        diff_x.block(pos + 3, 0, 3, 1) = Sophus::SO3d(m_vio_margin_ptr->m_ric[0].transpose() * temp_Q.toRotationMatrix()).log();
        diff_x(pos + 6, 0) = g_estimator->m_para_Td[0][0] - m_vio_margin_ptr->m_td;
        jacobian_matrix.block(0, (WINDOW_SIZE + 1) * 15, margin_residual_size, 7) = m_vio_margin_ptr->m_linearized_jacobians.block(0, pos, margin_residual_size, 7);
        m_residual_new = m_vio_margin_ptr->m_linearized_residuals + (m_vio_margin_ptr->m_linearized_jacobians * diff_x);
        residual_vec.block(0, 0, margin_residual_size, 1) = m_residual_new;
        if (m_vio_margin_ptr->m_if_enable_debug == 1)
        {
            Common_tools::save_matrix_to_txt("/home/ziv/temp/mar_linearized_res_new.txt", m_vio_margin_ptr->m_linearized_residuals);
            Common_tools::save_matrix_to_txt("/home/ziv/temp/mar_linearized_jac_new.txt", m_vio_margin_ptr->m_linearized_jacobians);
            Common_tools::save_matrix_to_txt("/home/ziv/temp/mar_residual_new.txt", m_residual_new);
            Common_tools::save_matrix_to_txt("/home/ziv/temp/mar_dx.txt", diff_x);
        }
    }
};



Common_tools::Timer LM_timer_tictoc;
double total_visual_res = 0;
void update_delta_vector(Estimator *estimator, Eigen::Matrix<double, -1, 1> &delta_vector)
{
    if (std::isnan(delta_vector.sum()))
    {
        return;
    }
    int feature_residual_size = delta_vector.rows() - (WINDOW_SIZE + 1) * 15 - 1 - 6;
    int if_update_para = 1;
    for (int idx = 0; idx < WINDOW_SIZE + 1; idx++)
    {
        Eigen::Quaterniond q_ori = Eigen::Quaterniond(estimator->m_para_Pose[idx][6], estimator->m_para_Pose[idx][3], estimator->m_para_Pose[idx][4], estimator->m_para_Pose[idx][5]);
        Eigen::Quaterniond q_delta = Sophus::SO3d::exp(delta_vector.block(idx * 15 + 3, 0, 3, 1)).unit_quaternion();
        Eigen::Quaterniond q_res = (q_ori * q_delta).normalized();
        for (int element = 0; element < 3; element++)
        {
            estimator->m_para_Pose[idx][element] += delta_vector(idx * 15 + element);
        }
        estimator->m_para_Pose[idx][6] = q_res.w();
        estimator->m_para_Pose[idx][3] = q_res.x();
        estimator->m_para_Pose[idx][4] = q_res.y();
        estimator->m_para_Pose[idx][5] = q_res.z();
        for (int element = 0; element < 9; element++)
        {
            estimator->m_para_SpeedBias[idx][element] += delta_vector(idx * 15 + 6 + element);
        }
    }

    if (ESTIMATE_EXTRINSIC)
    {
        Eigen::Quaterniond q_ori = Eigen::Quaterniond(estimator->m_para_Ex_Pose[0][6], estimator->m_para_Ex_Pose[0][3], estimator->m_para_Ex_Pose[0][4], estimator->m_para_Ex_Pose[0][5]);
        Eigen::Quaterniond q_delta = Sophus::SO3d::exp(delta_vector.block((WINDOW_SIZE + 1) * 15 + 3, 0, 3, 1)).unit_quaternion();
        Eigen::Quaterniond q_res = (q_ori * q_delta).normalized();
        for (int element = 0; element < 3; element++)
        {
            estimator->m_para_Ex_Pose[0][element] += delta_vector[(WINDOW_SIZE + 1) * 15 + element];
        }
        estimator->m_para_Ex_Pose[0][6] = q_res.w();
        estimator->m_para_Ex_Pose[0][3] = q_res.x();
        estimator->m_para_Ex_Pose[0][4] = q_res.y();
        estimator->m_para_Ex_Pose[0][5] = q_res.z();
    }
    
    estimator->m_para_Td[0][0] += delta_vector[(WINDOW_SIZE + 1) * 15 + 6];
    for (int element = 0; element < feature_residual_size; element++)
    {
        estimator->m_para_Feature[element][0] += delta_vector[(WINDOW_SIZE + 1) * 15 + 6 + 1 + element];
    }
}

void Evaluate(Estimator *estimator, std::vector<IMU_factor_res> &imu_factor_res_vec,
              std::vector<Keypoint_projection_factor> &projection_factor_res_vec,
              std::vector<L_prior_factor> & lidar_prior_factor_vec,
              Marginalization_factor &margin_factor,
              const int &feature_residual_size,
              Eigen::SparseMatrix<double> & jacobian_mat_sparse,
              Eigen::SparseMatrix<double> & residual_sparse,
              int marginalization_flag  = -1)   // Flag = 0, evaluate all, flag = 1, marginalize old, flag = 2, marginalize last.
{
    int number_of_imu_res = imu_factor_res_vec.size();
    int number_of_projection_res = projection_factor_res_vec.size();
    // pose[0], speed_bias[0],..., pose[Win+1], speed_bias[Win+1], I_CAM_E, Td, Feature_size
    int parameter_size = (WINDOW_SIZE + 1) * 15 + 6 + 1 + feature_residual_size + 1 ;
    int margin_res_size = 0;
    int lidar_prior_res_size = lidar_prior_factor_vec.size();
    int res_size = number_of_imu_res * 15 + number_of_projection_res * 2 + lidar_prior_res_size * 15 ;
    Eigen::Matrix<double, -1, 1> residual;
    Eigen::Matrix<double, -1, -1> jacobian_mat;
    Eigen::Matrix<double, -1, -1> hessian_mat, mat_I;
    
    Eigen::SparseMatrix<double> mat_I_sparse;

    if ( margin_factor.m_vio_margin_ptr != nullptr)
    {
        // margin_factor.Evaluate();
        // margin_res_size = margin_factor.m_margin_res_size;
        margin_res_size = margin_factor.m_vio_margin_ptr->m_linearized_jacobians.rows();
        res_size += margin_res_size;
    }

    residual.resize(res_size);
    jacobian_mat.resize(res_size, parameter_size);
    hessian_mat.resize(parameter_size, parameter_size);
    mat_I.resize(hessian_mat.rows(), hessian_mat.cols());
    mat_I.setIdentity();
    mat_I_sparse = mat_I.sparseView();
    residual.setZero();
    jacobian_mat.setZero();
    int jacobian_pos_col;
    int res_pos_ros = 0;

    if (margin_factor.m_vio_margin_ptr != nullptr)
    {
        margin_factor.Evaluate_mine(residual, jacobian_mat);
    }
    
    // Add IMU constrain factor
    for (int idx = 0; idx < number_of_imu_res; idx++)
    {
        if(marginalization_flag == Estimator::MARGIN_OLD && idx >=1) // Margin old
        {
            continue;
        }
        if(marginalization_flag == Estimator::MARGIN_SECOND_NEW )   // Margin second new
        {
            continue;
        }
        imu_factor_res_vec[idx].Evaluate();
        res_pos_ros = margin_res_size + 15 * idx;
        residual.block(res_pos_ros, 0, 15, 1) = imu_factor_res_vec[idx].m_residual;
        jacobian_pos_col = imu_factor_res_vec[idx].m_index_i * 15; // Pos[i]
        jacobian_mat.block(res_pos_ros, jacobian_pos_col, 15, 6) = imu_factor_res_vec[idx].m_jacobian_mat_vec[0].block(0, 0, 15, 6) ;

        jacobian_pos_col = imu_factor_res_vec[idx].m_index_i * 15 + 6; // speed_bias[i]
        jacobian_mat.block(res_pos_ros, jacobian_pos_col, 15, 9) = imu_factor_res_vec[idx].m_jacobian_mat_vec[1] ;

        jacobian_pos_col = imu_factor_res_vec[idx].m_index_j * 15; // Pos[j]
        jacobian_mat.block(res_pos_ros, jacobian_pos_col, 15, 6) = imu_factor_res_vec[idx].m_jacobian_mat_vec[2].block(0, 0, 15, 6) ;

        jacobian_pos_col = imu_factor_res_vec[idx].m_index_j * 15 + 6; // speed_bias[j]
        jacobian_mat.block(res_pos_ros, jacobian_pos_col, 15, 9) = imu_factor_res_vec[idx].m_jacobian_mat_vec[3] ;
    }
    

    // Add LiDAR prior residual
    for(int idx = 0 ; idx < lidar_prior_factor_vec.size(); idx++)
    {
        lidar_prior_factor_vec[idx].Evaluate();
        res_pos_ros = margin_res_size + 15 * (number_of_imu_res) + idx * 15;
        residual.block(res_pos_ros, 0, 15, 1) = lidar_prior_factor_vec[idx].m_residual  ;

        jacobian_pos_col = lidar_prior_factor_vec[idx].m_index_i * 15;     // Pos[i]
        jacobian_mat.block(res_pos_ros, jacobian_pos_col, 6, 6) = lidar_prior_factor_vec[idx].m_jacobian_mat_vec[0].block(0, 0, 6, 6);

        jacobian_pos_col = lidar_prior_factor_vec[idx].m_index_i * 15 + 6; // speed_bias[i]
        jacobian_mat.block(res_pos_ros, jacobian_pos_col, 15, 9) = lidar_prior_factor_vec[idx].m_jacobian_mat_vec[1].block(0, 0, 15, 9);
    }
    
    // Add projection factor
    for (int idx = 0; idx < number_of_projection_res; idx++)
    {
        if (marginalization_flag == Estimator::MARGIN_OLD && projection_factor_res_vec[idx].m_index_i != 0)
        {
            continue;
        }
        if (marginalization_flag == Estimator::MARGIN_SECOND_NEW)
        {
            continue;
        }
        projection_factor_res_vec[idx].Evaluate();
        if( fabs((projection_factor_res_vec[idx].m_jacobian_mat_vec[3].transpose() * projection_factor_res_vec[idx].m_jacobian_mat_vec[3]).coeff(0,0)) <= MIMIMUM_DELTA )
        {
            cout << "Visual [" << idx << "] unavailable!" << endl;
            continue;
        }
        res_pos_ros = margin_res_size + 15 * (number_of_imu_res) + idx * 2 + lidar_prior_res_size * 15;
        residual.block(res_pos_ros, 0, 2, 1) = projection_factor_res_vec[idx].m_residual;

        jacobian_pos_col = projection_factor_res_vec[idx].m_index_i * 15; // Pos[i]
        jacobian_mat.block(res_pos_ros, jacobian_pos_col, 2, 6) = projection_factor_res_vec[idx].m_jacobian_mat_vec[0].block(0, 0, 2, 6);

        jacobian_pos_col = projection_factor_res_vec[idx].m_index_j * 15; // Pos[j]
        jacobian_mat.block(res_pos_ros, jacobian_pos_col, 2, 6) = projection_factor_res_vec[idx].m_jacobian_mat_vec[1].block(0, 0, 2, 6);
        // Cam_IMU_extrinsic
        if (ESTIMATE_EXTRINSIC)
        {
            jacobian_pos_col = 15 * (WINDOW_SIZE + 1);
            jacobian_mat.block(res_pos_ros, jacobian_pos_col, 2, 6) = projection_factor_res_vec[idx].m_jacobian_mat_vec[2].block(0, 0, 2, 6);
        }

        jacobian_pos_col = 15 * (WINDOW_SIZE + 1) + 6; // Time offset
        jacobian_mat.block(res_pos_ros, jacobian_pos_col, 2, 1) = projection_factor_res_vec[idx].m_jacobian_mat_vec[4].block(0, 0, 2, 1);

        jacobian_pos_col = 15 * (WINDOW_SIZE + 1) + 6 + 1 + projection_factor_res_vec[idx].m_feature_index; // Keypoint res
        jacobian_mat.block(res_pos_ros, jacobian_pos_col, 2, 1) = projection_factor_res_vec[idx].m_jacobian_mat_vec[3].block(0, 0, 2, 1);
    }
    
    jacobian_mat_sparse = jacobian_mat.sparseView();
    residual_sparse = residual.sparseView();

    double current_cost = residual.array().abs().sum();

    return;
}

Eigen::Matrix<double, -1, 1> solve_LM(
              Eigen::SparseMatrix<double> & jacobian_mat_sparse,
              Eigen::SparseMatrix<double> & residual_sparse)
{
    int res_size = jacobian_mat_sparse.cols();
    int feature_residual_size = res_size - ((WINDOW_SIZE + 1) * 15 + 6 + 1);
    Eigen::Matrix<double, -1, 1> delta_vector;
    Eigen::SparseMatrix<double> hessian_mat_sparse, delta_vector_sparse, gradient_sparse, hessian_inv_sparse, hessian_temp_sparse;
    Eigen::Matrix<double, -1, -1> hessian_inv, gradient_dense, hessian_temp_dense;
    delta_vector.resize(res_size);
    delta_vector.setZero();
    jacobian_mat_sparse.makeCompressed();
    residual_sparse.makeCompressed();
    hessian_mat_sparse = jacobian_mat_sparse.transpose() * jacobian_mat_sparse;
    gradient_sparse = -jacobian_mat_sparse.transpose() * residual_sparse;

    LM_timer_tictoc.tic();
    hessian_temp_sparse = (hessian_mat_sparse);
    int solver_status = 0;

    delta_vector = sparse_schur_solver(hessian_temp_sparse * 1000.0, gradient_sparse * 1000.0, (WINDOW_SIZE + 1) * 15 + 6 + 1).toDense();

    double delta_vector_norm = delta_vector.block(0, 0, (WINDOW_SIZE + 1) * 15 + 6 + 1, 1).norm();
    
    if (delta_vector_norm > 1.0)
    {
        g_extra_iterations = 1;
    }

    return delta_vector;
}

void Estimator::optimization_LM()
{
    vector2double();
    double t_LM_cost = 0;
    double t_build_cost = 0;
    g_estimator = this;
    g_extra_iterations = 0;
    Common_tools::Timer timer_tictoc;
    timer_tictoc.tic();
    std::vector<IMU_factor_res> imu_factor_res_vec;
    std::vector<Keypoint_projection_factor> projection_factor_res_vec;
    std::vector<L_prior_factor> lidar_prior_factor_vec;
    Marginalization_factor margin_factor;
    imu_factor_res_vec.clear();
    projection_factor_res_vec.clear();
    lidar_prior_factor_vec.clear();
  
    g_loss_function = new ceres::HuberLoss(0.5);
    TicToc t_whole, t_prepare, t_solver;

    if (m_vio_margin_ptr)
    {
        margin_factor.m_vio_margin_ptr = m_vio_margin_ptr;
    }

    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        int j = i + 1;
        if (pre_integrations[j]->sum_dt > 10.0)
            continue;
        // IMUFactor_no_ceres *imu_factor = new IMUFactor_no_ceres(pre_integrations[j]);
        IMUFactor *imu_factor = new IMUFactor(pre_integrations[j]);
        IMU_factor_res imu_factor_res;
        imu_factor_res.add_keyframe_to_keyframe_factor(this, imu_factor, i, j);
        imu_factor_res_vec.push_back(imu_factor_res);        
    }

    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        LiDAR_prior_factor_15 *lidar_prior_factor = new LiDAR_prior_factor_15(&m_lio_state_prediction_vec[i]);
        L_prior_factor l_prior_factor;
        l_prior_factor.add_lidar_prior_factor(this, lidar_prior_factor, i);
        lidar_prior_factor_vec.push_back(l_prior_factor);
    }

    int f_m_cnt = 0;
    int feature_index = -1;
    for (auto &it_per_id : f_manager.feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;

        ++feature_index;

        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;

        Vector3d pts_i = it_per_id.feature_per_frame[0].point;

        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            if (imu_i == imu_j)
            {
                continue;
            }
            Vector3d pts_j = it_per_frame.point;

            ProjectionTdFactor *f_td = new ProjectionTdFactor(pts_i, pts_j, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
                                                              it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td,
                                                              it_per_id.feature_per_frame[0].uv.y(), it_per_frame.uv.y());
            Keypoint_projection_factor key_pt_projection_factor;
            key_pt_projection_factor.add_projection_factor(this, f_td, imu_i, imu_j, feature_index);
            projection_factor_res_vec.push_back(key_pt_projection_factor);
            f_m_cnt++;
        }
    }

    Eigen::SparseMatrix<double> residual_sparse, jacobian_sparse, hessian_sparse;
    LM_trust_region_strategy lm_trust_region;
    for (int iter_count = 0; iter_count < NUM_ITERATIONS + g_extra_iterations; iter_count++)
    {
        t_build_cost += timer_tictoc.toc();
        timer_tictoc.tic();
        Evaluate(this, imu_factor_res_vec, projection_factor_res_vec, lidar_prior_factor_vec, margin_factor, feature_index, jacobian_sparse, residual_sparse);
        Eigen::VectorXd delta_vector;
        //delta_vector = solve_LM(jacobian_sparse, residual_sparse);
        delta_vector = lm_trust_region.compute_step(jacobian_sparse, residual_sparse, (WINDOW_SIZE + 1) * 15 + 6 + 1).toDense();
        update_delta_vector(this, delta_vector);
        t_LM_cost += timer_tictoc.toc();
    }
    double2vector();
    Evaluate(this, imu_factor_res_vec, projection_factor_res_vec, lidar_prior_factor_vec, margin_factor, feature_index,
             jacobian_sparse, residual_sparse, marginalization_flag);
    // ANCHOR - VIO marginalization
    if (m_vio_margin_ptr)
    {
        delete m_vio_margin_ptr;
    }
    m_vio_margin_ptr = new vio_marginalization();    

    for(int i =0; i < WINDOW_SIZE+1; i++)
    {
        m_vio_margin_ptr->m_Ps[i] = Ps[i];
        m_vio_margin_ptr->m_Vs[i] = Vs[i];
        m_vio_margin_ptr->m_Rs[i] = Rs[i];
        m_vio_margin_ptr->m_Bas[i] = Bas[i];
        m_vio_margin_ptr->m_Bgs[i] = Bgs[i];
        m_vio_margin_ptr->m_ric[0] = ric[0];
        m_vio_margin_ptr->m_tic[0] = tic[0];
        m_vio_margin_ptr->m_td = td;
    }

    if (marginalization_flag == MARGIN_OLD)
    {
        int visual_size = jacobian_sparse.cols() - (15 * (WINDOW_SIZE + 1) + 6 + 1); // Extrinsic, Td
        hessian_sparse = jacobian_sparse.transpose() * jacobian_sparse;
        m_vio_margin_ptr->margin_oldest_frame(hessian_sparse.toDense(), (jacobian_sparse.transpose() * residual_sparse).toDense(), visual_size);
    }
    else if (marginalization_flag == MARGIN_SECOND_NEW)
    {
        int visual_size = jacobian_sparse.cols() - (15 * (WINDOW_SIZE + 1) + 6 + 1); // Extrinsic, Td
        hessian_sparse = jacobian_sparse.transpose() * jacobian_sparse;
        m_vio_margin_ptr->margin_second_new_frame(hessian_sparse.toDense(), (jacobian_sparse.transpose() * residual_sparse).toDense(), visual_size);
    }
}