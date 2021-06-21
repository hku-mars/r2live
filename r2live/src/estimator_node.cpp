#include <stdio.h>
#include <queue>
#include <map>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include "estimator.h"
#include "parameters.h"
#include "utility/visualization.h"
#include "./fast_lio/fast_lio.hpp"
#define CAM_MEASUREMENT_COV 1e-3
Camera_Lidar_queue g_camera_lidar_queue;
MeasureGroup Measures;
StatesGroup g_lio_state;

Estimator estimator;

std::condition_variable con;
double current_time = -1;
queue<sensor_msgs::ImuConstPtr> imu_buf;
queue<sensor_msgs::PointCloudConstPtr> feature_buf;
queue<sensor_msgs::PointCloudConstPtr> relo_buf;
int sum_of_wait = 0;

std::mutex m_buf;
std::mutex m_state;
std::mutex i_buf;
std::mutex m_estimator;

double latest_time;
Eigen::Vector3d tmp_P;
Eigen::Quaterniond tmp_Q;
Eigen::Vector3d tmp_V;
Eigen::Vector3d tmp_Ba;
Eigen::Vector3d tmp_Bg;
Eigen::Vector3d acc_0;
Eigen::Vector3d gyr_0;

eigen_q diff_vins_lio_q = eigen_q::Identity();
vec_3 diff_vins_lio_t = vec_3::Zero();

bool init_feature = 0;
bool init_imu = 1;
double last_imu_t = -1;

void predict(const sensor_msgs::ImuConstPtr &imu_msg)
{
    double t = imu_msg->header.stamp.toSec();
    if (init_imu)
    {
        latest_time = t;
        init_imu = 0;
        return;
    }
    double dt = t - latest_time;
    latest_time = t;

    double dx = imu_msg->linear_acceleration.x;
    double dy = imu_msg->linear_acceleration.y;
    double dz = imu_msg->linear_acceleration.z;
    Eigen::Vector3d linear_acceleration{dx, dy, dz};

    double rx = imu_msg->angular_velocity.x;
    double ry = imu_msg->angular_velocity.y;
    double rz = imu_msg->angular_velocity.z;
    Eigen::Vector3d angular_velocity{rx, ry, rz};

    Eigen::Vector3d un_acc_0 = tmp_Q * (acc_0 - tmp_Ba) - estimator.m_gravity;

    Eigen::Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - tmp_Bg;
    tmp_Q = tmp_Q * Utility::deltaQ(un_gyr * dt);

    Eigen::Vector3d un_acc_1 = tmp_Q * (linear_acceleration - tmp_Ba) - estimator.m_gravity;

    Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);

    tmp_P = tmp_P + dt * tmp_V + 0.5 * dt * dt * un_acc;
    tmp_V = tmp_V + dt * un_acc;

    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity;
}

void update()
{
    TicToc t_predict;
    latest_time = current_time;
    tmp_P = estimator.Ps[WINDOW_SIZE];
    tmp_Q = estimator.Rs[WINDOW_SIZE];
    tmp_V = estimator.Vs[WINDOW_SIZE];
    tmp_Ba = estimator.Bas[WINDOW_SIZE];
    tmp_Bg = estimator.Bgs[WINDOW_SIZE];
    acc_0 = estimator.acc_0;
    gyr_0 = estimator.gyr_0;

    queue<sensor_msgs::ImuConstPtr> tmp_imu_buf = imu_buf;
    for (sensor_msgs::ImuConstPtr tmp_imu_msg; !tmp_imu_buf.empty(); tmp_imu_buf.pop())
        predict(tmp_imu_buf.front());
}

std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>> getMeasurements()
{
    std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>> measurements;
    std::unique_lock<std::mutex> lk(m_buf);
    while (true)
    {
        if (imu_buf.empty() || feature_buf.empty())
            return measurements;

        if (!(imu_buf.back()->header.stamp.toSec() > feature_buf.front()->header.stamp.toSec() + estimator.td))
        {
            //ROS_WARN("wait for imu, only should happen at the beginning");
            sum_of_wait++;
            return measurements;
        }

        if (!(imu_buf.front()->header.stamp.toSec() < feature_buf.front()->header.stamp.toSec() + estimator.td))
        {
            ROS_WARN("throw img, only should happen at the beginning [%.5f | %.5f] ", imu_buf.front()->header.stamp.toSec(), feature_buf.front()->header.stamp.toSec());
            feature_buf.pop();
            continue;
        }
        sensor_msgs::PointCloudConstPtr img_msg = feature_buf.front();
        feature_buf.pop();

        std::vector<sensor_msgs::ImuConstPtr> IMUs;
        while (imu_buf.front()->header.stamp.toSec() < img_msg->header.stamp.toSec() + estimator.td)
        {
            IMUs.emplace_back(imu_buf.front());
            imu_buf.pop();
        }
        IMUs.emplace_back(imu_buf.front());
        if (IMUs.empty())
            ROS_WARN("no imu between two image");
        measurements.emplace_back(IMUs, img_msg);
    }
    return measurements;
}

void imu_callback(const sensor_msgs::ImuConstPtr &_imu_msg)
{
    sensor_msgs::ImuPtr imu_msg = boost::make_shared<sensor_msgs::Imu>();
    *imu_msg = *_imu_msg;

    // if (g_if_divide_g == 0)
    if (g_camera_lidar_queue.m_if_acc_mul_G) // For LiVOX Avia built-in IMU
    {
        imu_msg->linear_acceleration.x *= 9.805;
        imu_msg->linear_acceleration.y *= 9.805;
        imu_msg->linear_acceleration.z *= 9.805;
    }

    if (imu_msg->header.stamp.toSec() <= last_imu_t)
    {
        ROS_WARN("imu message in disorder!");
        return;
    }
    g_camera_lidar_queue.imu_in(imu_msg->header.stamp.toSec());

    last_imu_t = imu_msg->header.stamp.toSec();

    m_buf.lock();
    imu_buf.push(imu_msg);
    m_buf.unlock();
    con.notify_one();

    last_imu_t = imu_msg->header.stamp.toSec();
    // cout << "Imu_msg last time= " << last_imu_t << endl;
    {
        std::lock_guard<std::mutex> lg(m_state);
        predict(imu_msg);
        std_msgs::Header header = imu_msg->header;
        header.frame_id = "world";
        if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
            pubLatestOdometry(tmp_P, tmp_Q, tmp_V, header);
    }
}

void feature_callback(const sensor_msgs::PointCloudConstPtr &feature_msg)
{
    if (!init_feature)
    {
        //skip the first detected feature, which doesn't contain optical flow speed
        init_feature = 1;
        return;
    }

    m_buf.lock();
    feature_buf.push(feature_msg);
    m_buf.unlock();
    con.notify_one();
}

void restart_callback(const std_msgs::BoolConstPtr &restart_msg)
{
    if (restart_msg->data == true)
    {
        ROS_WARN("restart the estimator!");
        m_buf.lock();
        while (!feature_buf.empty())
            feature_buf.pop();
        while (!imu_buf.empty())
            imu_buf.pop();
        m_buf.unlock();
        m_estimator.lock();
        estimator.clearState();
        estimator.setParameter();
        m_estimator.unlock();
        current_time = -1;
        last_imu_t = 0;
    }
    return;
}

void relocalization_callback(const sensor_msgs::PointCloudConstPtr &points_msg)
{
    //printf("relocalization callback! \n");
    m_buf.lock();
    relo_buf.push(points_msg);
    m_buf.unlock();
}

// thread: visual-inertial odometry
void unlock_lio(Estimator &estimator)
{
    if (estimator.m_fast_lio_instance)
    {
        estimator.m_fast_lio_instance->m_mutex_lio_process.unlock();
    }
}

void lock_lio(Estimator &estimator)
{
    if (estimator.m_fast_lio_instance)
    {
        estimator.m_fast_lio_instance->m_mutex_lio_process.lock();
    }
}

// ANCHOR - sync lio to cam
void sync_lio_to_vio(Estimator &estimator)
{
    check_state(g_lio_state);
    int frame_idx = estimator.frame_count;
    frame_idx = WINDOW_SIZE;
    if (abs(g_camera_lidar_queue.m_last_visual_time - g_lio_state.last_update_time) < 1.0)
    {
        if (g_lio_state.bias_a.norm() < 0.5 && g_lio_state.bias_g.norm() < 1.0)
        {
            estimator.Bas[frame_idx] = g_lio_state.bias_a;
            estimator.Bgs[frame_idx] = g_lio_state.bias_g;
            estimator.Vs[frame_idx] = diff_vins_lio_q.toRotationMatrix().inverse() * g_lio_state.vel_end;
            estimator.m_gravity = g_lio_state.gravity;
            G_gravity = estimator.m_gravity;
            update();
        }
    }
}

void visual_imu_measure(const Eigen::Vector3d &pts_i, const Eigen::Vector3d &pts_j,
                        const Eigen::Vector3d &Pi, const Eigen::Quaterniond &Qi,
                        const Eigen::Vector3d &Pj, const Eigen::Quaterniond &Qj,
                        const Eigen::Vector3d &tic, const Eigen::Quaterniond &qic,
                        const double inv_dep_i,
                        Eigen::Vector2d & residual,
                        Eigen::Matrix<double, 2, 6, Eigen::RowMajor> & j_mat)

{
    Eigen::Matrix2d sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    // Eigen::Matrix2d sqrt_info = Matrix2d::Identity();
    Eigen::Vector3d pts_camera_i = pts_i / inv_dep_i;
    Eigen::Vector3d pts_imu_i = qic * pts_camera_i + tic;
    Eigen::Vector3d pts_w = Qi * pts_imu_i + Pi;
    Eigen::Vector3d pts_imu_j = Qj.inverse() * (pts_w - Pj);
    Eigen::Vector3d pts_camera_j = qic.inverse() * (pts_imu_j - tic);
    
    Eigen::Matrix3d Ri = Qi.toRotationMatrix();
    Eigen::Matrix3d Rj = Qj.toRotationMatrix();
    Eigen::Matrix3d ric = qic.toRotationMatrix();
    
    // Eigen::Vector2d residual;

    double dep_j = pts_camera_j.z();
    residual = (pts_camera_j / dep_j).head<2>() - pts_j.head<2>();
    residual = sqrt_info * residual;

    Eigen::Matrix<double, 2, 3> reduce(2, 3);
    reduce << 1. / dep_j, 0, -pts_camera_j(0) / (dep_j * dep_j),
        0, 1. / dep_j, -pts_camera_j(1) / (dep_j * dep_j);
    reduce = sqrt_info * reduce;

    // Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_pose_j(jacobians[1]);

    Eigen::Matrix<double, 3, 6> jaco_j;
    Eigen::Matrix<double, 2, 6, Eigen::RowMajor>  j_mat_temp;
    jaco_j.leftCols<3>() = ric.transpose() * -Rj.transpose();
    jaco_j.rightCols<3>() = ric.transpose() * Utility::skewSymmetric(pts_imu_j);
    j_mat_temp = reduce * jaco_j;
    j_mat.block(0, 3, 2, 3) = j_mat_temp.block(0, 0, 2, 3);
    j_mat.block(0, 0, 2, 3) = j_mat_temp.block(0, 3, 2, 3);

}

void construct_camera_measure(int frame_idx, Estimator &estimator,
                              std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> &reppro_err_vec,
                              std::vector<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>, Eigen::aligned_allocator<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>>> &J_mat_vec)
{
    J_mat_vec.clear();
    reppro_err_vec.clear();
    scope_color(ANSI_COLOR_GREEN_BOLD);
    int f_m_cnt = 0;
    int feature_index = -1;
    int min_frame = 3e8;
    int max_frame = -3e8;
    for (auto &it_per_id : estimator.f_manager.feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 1))
            continue;

        ++feature_index;

        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;

        Vector3d pts_i = it_per_id.feature_per_frame[0].point;

        Eigen::Vector2d residual_vec, residual_vec_old;
        std::vector<double *> parameters_vec;
        Eigen::Matrix<double, 2, 7, Eigen::RowMajor> j_mat_tq;
        Eigen::Matrix<double, 2, 6, Eigen::RowMajor> j_mat, j_mat_old;
        j_mat.setZero();
        j_mat_tq.setZero();
        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            //if (imu_i == imu_j)
            if(fabs(imu_i - imu_j) < std::max( double(WINDOW_SIZE / 3), 2.0 ))
            {
                continue;
            }
            min_frame = std::min(imu_j, min_frame);
            max_frame = std::max(imu_j, max_frame);
            min_frame = std::min(imu_i, min_frame);
            max_frame = std::max(imu_i, max_frame);

            if (imu_j == (frame_idx))
            {
                Vector3d pts_j = it_per_frame.point;
                double *jacobian_mat_vec[4];
                parameters_vec.push_back(estimator.m_para_Pose[imu_i]);
                parameters_vec.push_back(estimator.m_para_Pose[imu_j]);
                parameters_vec.push_back(estimator.m_para_Ex_Pose[0]);
                parameters_vec.push_back(estimator.m_para_Feature[feature_index]);

                Eigen::Vector3d Pi(parameters_vec[0][0], parameters_vec[0][1], parameters_vec[0][2]);
                Eigen::Quaterniond Qi(parameters_vec[0][6], parameters_vec[0][3], parameters_vec[0][4], parameters_vec[0][5]);

                Eigen::Vector3d Pj(parameters_vec[1][0], parameters_vec[1][1], parameters_vec[1][2]);
                Eigen::Quaterniond Qj(parameters_vec[1][6], parameters_vec[1][3], parameters_vec[1][4], parameters_vec[1][5]);

                Eigen::Vector3d tic(parameters_vec[2][0], parameters_vec[2][1], parameters_vec[2][2]);
                Eigen::Quaterniond qic(parameters_vec[2][6], parameters_vec[2][3], parameters_vec[2][4], parameters_vec[2][5]);

                double inverse_depth = parameters_vec[3][0];

                if(0)
                {
                    jacobian_mat_vec[0] = nullptr;
                    jacobian_mat_vec[1] = j_mat_tq.data();
                    jacobian_mat_vec[2] = nullptr;
                    jacobian_mat_vec[3] = nullptr;
                    ProjectionFactor *f = new ProjectionFactor(pts_i, pts_j);
                    f->Evaluate(parameters_vec.data(), residual_vec_old.data(), (double **)jacobian_mat_vec);
                    j_mat_old.block(0, 3, 2, 3) = j_mat_tq.block(0, 0, 2, 3);
                    j_mat_old.block(0, 0, 2, 3) = j_mat_tq.block(0, 3, 2, 3);
                }

                if(1)
                {
                    visual_imu_measure(pts_i, pts_j, Pi, Qi, Pj, Qj, tic, qic, inverse_depth, residual_vec, j_mat);
                }

                if (0 && reppro_err_vec.size() == 0)
                {
                    cout << "============================" << endl;
                    cout << "Old ESIKF first res_vec: " << residual_vec_old.transpose() << endl;
                    cout << "ESIKF first res_vec: " << residual_vec.transpose() << endl;
                    cout << "ESIKF first H_mat [2,7]:\r\n"
                         << j_mat_tq << endl;
                    cout << "Old ESIKF first H_mat [2,6]:\r\n"
                         << j_mat_old << endl;
                    cout << "ESIKF first H_mat [2,6]:\r\n"
                         << j_mat << endl;
                    
                }

                if (std::isnan(residual_vec.sum()) || std::isnan(j_mat.sum()))
                {
                    continue;
                }

                reppro_err_vec.push_back(residual_vec);
                J_mat_vec.push_back(j_mat);
            }
        }
        f_m_cnt++;
    }

    // cout << "Total measure size= " << reppro_err_vec.size() << endl;
    // cout << "Min frame = " << min_frame << ", max_frame = " << max_frame
    //      << ", Last track_num = " << estimator.f_manager.last_track_num
    //      << endl;
}

int need_refresh_extrinsic = 0;

void process()
{
    Eigen::Matrix<double, DIM_OF_STATES, DIM_OF_STATES> G, H_T_H, I_STATE;
    G.setZero();
    H_T_H.setZero();
    I_STATE.setIdentity();
    std::shared_ptr<ImuProcess> p_imu(new ImuProcess());

    g_camera_lidar_queue.m_if_lidar_can_start = g_camera_lidar_queue.m_if_lidar_start_first;
    std_msgs::Header header;
    while (true)
    {
        std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>> measurements;
        measurements = getMeasurements();
        if(measurements.size() == 0)
        {
            continue;
        }
        m_estimator.lock();

        g_camera_lidar_queue.m_last_visual_time = -3e8;
        TicToc t_s;
        for (auto &measurement : measurements)
        {
            auto img_msg = measurement.second;
            int if_camera_can_update = 1;
            double cam_update_tim = img_msg->header.stamp.toSec() + estimator.td;
            // ROS_INFO("Estimated td = %.5f" ,estimator.td );
            // ANCHOR - determine if update of not.
            if (estimator.m_fast_lio_instance != nullptr)
            {
                g_camera_lidar_queue.m_camera_imu_td = estimator.td;
                g_camera_lidar_queue.m_last_visual_time = img_msg->header.stamp.toSec();
                while (g_camera_lidar_queue.if_camera_can_process() == false)
                {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }

                lock_lio(estimator);
                t_s.tic();
                double camera_LiDAR_tim_diff = img_msg->header.stamp.toSec() + g_camera_lidar_queue.m_camera_imu_td - g_lio_state.last_update_time;
            *p_imu = *(estimator.m_fast_lio_instance->m_imu_process);
            }

            if ((g_camera_lidar_queue.m_if_lidar_can_start == true) && (g_camera_lidar_queue.m_lidar_drag_cam_tim >= 0))
            {
                m_state.lock();
                sync_lio_to_vio(estimator);
                m_state.unlock();
            }
            double dx = 0, dy = 0, dz = 0, rx = 0, ry = 0, rz = 0;
            int skip_imu = 0;
            for (auto &imu_msg : measurement.first)
            {
                double t = imu_msg->header.stamp.toSec();
                double img_t = img_msg->header.stamp.toSec() + estimator.td;

                if (t <= img_t)
                {
                    if (current_time < 0)
                        current_time = t;
                    double dt = t - current_time;
                    ROS_ASSERT(dt >= 0);
                    current_time = t;
                    dx = imu_msg->linear_acceleration.x;
                    dy = imu_msg->linear_acceleration.y;
                    dz = imu_msg->linear_acceleration.z;
                    rx = imu_msg->angular_velocity.x;
                    ry = imu_msg->angular_velocity.y;
                    rz = imu_msg->angular_velocity.z;
                    estimator.processIMU(dt, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz));
                }
                else
                {
                    double dt_1 = img_t - current_time;
                    double dt_2 = t - img_t;
                    current_time = img_t;
                    ROS_ASSERT(dt_1 >= 0);
                    ROS_ASSERT(dt_2 >= 0);
                    ROS_ASSERT(dt_1 + dt_2 > 0);
                    double w1 = dt_2 / (dt_1 + dt_2);
                    double w2 = dt_1 / (dt_1 + dt_2);
                    dx = w1 * dx + w2 * imu_msg->linear_acceleration.x;
                    dy = w1 * dy + w2 * imu_msg->linear_acceleration.y;
                    dz = w1 * dz + w2 * imu_msg->linear_acceleration.z;
                    rx = w1 * rx + w2 * imu_msg->angular_velocity.x;
                    ry = w1 * ry + w2 * imu_msg->angular_velocity.y;
                    rz = w1 * rz + w2 * imu_msg->angular_velocity.z;
                    estimator.processIMU(dt_1, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz));
                }
            }
            // set relocalization frame
            sensor_msgs::PointCloudConstPtr relo_msg = NULL;
            while (!relo_buf.empty())
            {
                relo_msg = relo_buf.front();
                relo_buf.pop();
            }

            if (relo_msg != NULL)
            {
                vector<Vector3d> match_points;
                double frame_stamp = relo_msg->header.stamp.toSec();
                for (unsigned int i = 0; i < relo_msg->points.size(); i++)
                {
                    Vector3d u_v_id;
                    u_v_id.x() = relo_msg->points[i].x;
                    u_v_id.y() = relo_msg->points[i].y;
                    u_v_id.z() = relo_msg->points[i].z;
                    match_points.push_back(u_v_id);
                }
                Vector3d relo_t(relo_msg->channels[0].values[0], relo_msg->channels[0].values[1], relo_msg->channels[0].values[2]);
                Quaterniond relo_q(relo_msg->channels[0].values[3], relo_msg->channels[0].values[4], relo_msg->channels[0].values[5], relo_msg->channels[0].values[6]);
                Matrix3d relo_r = relo_q.toRotationMatrix();
                int frame_index;
                frame_index = relo_msg->channels[0].values[7];
                estimator.setReloFrame(frame_stamp, frame_index, match_points, relo_t, relo_r);
            }

            ROS_DEBUG("processing vision data with stamp %f \n", img_msg->header.stamp.toSec());

            std::deque<sensor_msgs::Imu::ConstPtr> imu_queue;
            int total_IMU_cnt = 0;
            int acc_IMU_cnt = 0;
            for (auto &imu_msg : measurement.first)
            {
                total_IMU_cnt++;
                if(imu_msg->header.stamp.toSec() >  g_lio_state.last_update_time )
                {
                    acc_IMU_cnt++;
                    imu_queue.push_back(imu_msg);
                }
            }
            
            StatesGroup state_aft_integration = g_lio_state;
            int esikf_update_valid = false;
            if (imu_queue.size())
            {
                if (g_lio_state.last_update_time == 0)
                {
                    g_lio_state.last_update_time = imu_queue.front()->header.stamp.toSec();
                }
                double start_dt = g_lio_state.last_update_time - imu_queue.front()->header.stamp.toSec();
                double end_dt = cam_update_tim - imu_queue.back()->header.stamp.toSec();
                esikf_update_valid = true;
                if (g_camera_lidar_queue.m_if_have_lidar_data && (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR))
                {
                    *p_imu = *(estimator.m_fast_lio_instance->m_imu_process);
                    state_aft_integration = p_imu->imu_preintegration(g_lio_state, imu_queue, 0, cam_update_tim - imu_queue.back()->header.stamp.toSec());
                    estimator.m_lio_state_prediction_vec[WINDOW_SIZE] = state_aft_integration;
                    
                    diff_vins_lio_q = eigen_q(estimator.Rs[WINDOW_SIZE].transpose() * state_aft_integration.rot_end);
                    diff_vins_lio_t = state_aft_integration.pos_end - estimator.Ps[WINDOW_SIZE];
                    if (diff_vins_lio_t.norm() > 1.0)
                    {
                        // ROS_INFO("VIO subsystem restart ");
                        estimator.refine_vio_system(diff_vins_lio_q, diff_vins_lio_t);
                        diff_vins_lio_q.setIdentity();
                        diff_vins_lio_t.setZero();
                    }
                    if ((start_dt > -0.02) &&
                        (fabs(end_dt) < 0.02))
                    {
                        g_lio_state = state_aft_integration;
                        g_lio_state.last_update_time = cam_update_tim;
                    }
                    else
                    {
                        // esikf_update_valid = false;
                        scope_color(ANSI_COLOR_RED_BOLD);
                        cout << "Start time = " << std::setprecision(8) << imu_queue.front()->header.stamp.toSec() - estimator.m_fast_lio_instance->first_lidar_time << endl;
                        cout << "Final time = " << std::setprecision(8) << cam_update_tim - estimator.m_fast_lio_instance->first_lidar_time << endl;
                        cout << "Start dt = " << start_dt << std::setprecision(2) << endl;
                        cout << "Final dt = " << end_dt << std::setprecision(2) << endl;
                        cout << "LiDAR->Image preintegration: " << start_dt << " <--> " << end_dt << endl;
                    }
                }
            }

            std::map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> image;
            for (unsigned int i = 0; i < img_msg->points.size(); i++)
            {
                int v = img_msg->channels[0].values[i] + 0.5;
                int feature_id = v / NUM_OF_CAM;
                int camera_id = v % NUM_OF_CAM;
                double x = img_msg->points[i].x;
                double y = img_msg->points[i].y;
                double z = img_msg->points[i].z;
                double p_u = img_msg->channels[1].values[i];
                double p_v = img_msg->channels[2].values[i];
                double velocity_x = img_msg->channels[3].values[i];
                double velocity_y = img_msg->channels[4].values[i];
                ROS_ASSERT(z == 1);
                Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
                xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
                image[feature_id].emplace_back(camera_id, xyz_uv_velocity);
            }
            // t_s.tic();
            estimator.processImage(image, img_msg->header);
            // estimator.vector2double();
            //Step 1: IMU preintergration
            StatesGroup state_prediction = state_aft_integration;

            // //Step 3: ESIKF udpate.
            double mean_reprojection_error = 0.0;
            int minmum_number_of_camera_res = 10;
            
            StatesGroup state_before_esikf = g_lio_state;
            if (esikf_update_valid && (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR) 
                && (g_lio_state.last_update_time - g_camera_lidar_queue.m_visual_init_time > g_camera_lidar_queue.m_lidar_drag_cam_tim)
                )
            {
                estimator.vector2double();
                estimator.f_manager.triangulate(estimator.Ps, estimator.tic, estimator.ric);

                double deltaR = 0, deltaT = 0;
                int flg_EKF_converged = 0;
                Eigen::Matrix<double, DIM_OF_STATES, 1> solution;
                Eigen::Vector3d rot_add, t_add, v_add, bg_add, ba_add, g_add;

                std::vector<Eigen::Vector3d> pts_i, pts_j;
                std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> reppro_err_vec;
                std::vector<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>, Eigen::aligned_allocator<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>>> J_mat_vec;
                Eigen::Matrix<double, -1, -1> Hsub;
                Eigen::Matrix<double, -1, 1> meas_vec;
                int win_idx = WINDOW_SIZE;

                Eigen::Matrix<double, -1, -1> K;
                Eigen::Matrix<double, DIM_OF_STATES, DIM_OF_STATES> K_1;
                for (int iter_time = 0; iter_time < 2; iter_time++)
                {
                    Eigen::Quaterniond q_pose_last = Eigen::Quaterniond(state_aft_integration.rot_end * diff_vins_lio_q.inverse());
                    Eigen::Vector3d t_pose_last = state_aft_integration.pos_end - diff_vins_lio_t;
                    estimator.m_para_Pose[win_idx][0] = t_pose_last(0);
                    estimator.m_para_Pose[win_idx][1] = t_pose_last(1);
                    estimator.m_para_Pose[win_idx][2] = t_pose_last(2);
                    estimator.m_para_Pose[win_idx][3] = q_pose_last.x();
                    estimator.m_para_Pose[win_idx][4] = q_pose_last.y();
                    estimator.m_para_Pose[win_idx][5] = q_pose_last.z();
                    estimator.m_para_Pose[win_idx][6] = q_pose_last.w();
                    // estimator.f_manager.removeFailures();
                    construct_camera_measure(win_idx, estimator, reppro_err_vec, J_mat_vec);

                    if (reppro_err_vec.size() < minmum_number_of_camera_res)
                    {
                        cout << "Size of reppro_err_vec: " << reppro_err_vec.size() << endl;
                        break;
                    }
                    
                    // TODO: Add camera residual here
                    Hsub.resize(reppro_err_vec.size() * 2, 6);
                    meas_vec.resize(reppro_err_vec.size() * 2, 1);
                    K.resize(DIM_OF_STATES, reppro_err_vec.size());
                    int features_correspondences = reppro_err_vec.size();
                    
                    for (int residual_idx = 0; residual_idx < reppro_err_vec.size(); residual_idx++)
                    {
                        meas_vec.block(residual_idx * 2, 0, 2, 1) = -1 * reppro_err_vec[residual_idx];
                        // J_mat_vec[residual_idx].block(0,0,2,3) = J_mat_vec[residual_idx].block(0,0,2,3) * extrinsic_vins_lio_q.toRotationMatrix().transpose();
                        Hsub.block(residual_idx * 2, 0, 2, 6) = J_mat_vec[residual_idx];
                    }

                    K_1.setZero();
                    auto Hsub_T = Hsub.transpose();

                    H_T_H.setZero();
                    H_T_H.block<6, 6>(0, 0) = Hsub_T * Hsub;
                    K_1 = (H_T_H + (state_aft_integration.cov * CAM_MEASUREMENT_COV).inverse()).inverse();
                    K = K_1.block<DIM_OF_STATES, 6>(0, 0) * Hsub_T;
                    auto vec = state_prediction - state_aft_integration;
                    solution = K * (meas_vec - Hsub * vec.block<6, 1>(0, 0) );

                    mean_reprojection_error = abs(meas_vec.mean());

                    if (std::isnan(solution.sum()))
                    {
                        break;
                    }

                    state_aft_integration = state_prediction + solution;
                    solution = state_aft_integration - state_prediction;

                    rot_add = (solution).block<3, 1>(0, 0);
                    t_add = solution.block<3, 1>(3, 0);
                    flg_EKF_converged = false;
                    if (((rot_add.norm() * 57.3 - deltaR) < 0.01) && ((t_add.norm()  - deltaT) < 0.015))
                    {
                        flg_EKF_converged = true;
                    }
                    deltaR = rot_add.norm() * 57.3;
                    deltaT = t_add.norm() ;
                }
                if (reppro_err_vec.size() >= minmum_number_of_camera_res)
                {
                    G.setZero();
                    G.block<DIM_OF_STATES, 6>(0, 0) = K * Hsub;

                    if ((rot_add.norm() * 57.3 < 3) &&
                        (t_add.norm() < 0.5) &&
                        (mean_reprojection_error < 1.0))
                    {
                        g_lio_state = state_aft_integration;
                        eigen_q q_I = eigen_q(1.0, 0, 0, 0);
                        double angular_diff = eigen_q(g_lio_state.rot_end.transpose() * state_before_esikf.rot_end).angularDistance(q_I) * 57.3;
                        double t_diff = (g_lio_state.pos_end - state_before_esikf.pos_end).norm();
                        if ((t_diff > 0.2) || (angular_diff > 2.0))
                        {
                            g_lio_state = state_before_esikf;
                        }
                        // Unblock lio process, publish esikf state.
                        // TODO: publish esikf state.
                        // unlock_lio(estimator);
                        // g_camera_lidar_queue.m_last_visual_time  = g_lio_state.last_update_time + 0.02; // Unblock lio process, forward 80 ms
                    
                    }
                }
            }

            // Update state with pose graph optimization 
            g_lio_state = state_before_esikf;
            t_s.tic();
            estimator.solve_image_pose(img_msg->header);

            if (g_camera_lidar_queue.m_if_have_lidar_data)
            {
                if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR && esikf_update_valid)
                {
                    if (g_camera_lidar_queue.m_visual_init_time == 3e88)
                    {
                        scope_color(ANSI_COLOR_RED_BOLD);
                        printf("G_camera_lidar_queue.m_visual_init_time = %2.f \r\n", cam_update_tim);
                        g_camera_lidar_queue.m_visual_init_time = cam_update_tim;
                    }

                    if ((g_lio_state.last_update_time - g_camera_lidar_queue.m_visual_init_time > g_camera_lidar_queue.m_lidar_drag_cam_tim))
                    {
                        StatesGroup state_before_esikf = g_lio_state;
                        if(estimator.Bas[WINDOW_SIZE].norm() < 0.5)
                        {
                            g_lio_state.bias_a = estimator.Bas[WINDOW_SIZE];
                        }
                        g_lio_state.bias_g = estimator.Bgs[WINDOW_SIZE];
                        g_lio_state.vel_end = diff_vins_lio_q.toRotationMatrix() * estimator.Vs[WINDOW_SIZE];
                        g_lio_state.cov = state_aft_integration.cov;

                        Eigen::Matrix3d temp_R = estimator.Rs[WINDOW_SIZE] * diff_vins_lio_q.toRotationMatrix();
                        Eigen::Vector3d temp_T =  estimator.Ps[WINDOW_SIZE] + diff_vins_lio_t;
                        eigen_q q_I = eigen_q(1.0, 0, 0, 0);
                        double angular_diff = eigen_q(temp_R.transpose() * state_before_esikf.rot_end).angularDistance(q_I) * 57.3;
                        double t_diff = (temp_T - state_before_esikf.pos_end).norm();
                        if ((t_diff < 0.2) &&  (angular_diff < 2.0))
                        {
                            g_lio_state.cov = state_aft_integration.cov;
                            g_lio_state.last_update_time = cam_update_tim;
                            g_lio_state.rot_end = temp_R;
                            g_lio_state.pos_end = temp_T;
                        }
                        unlock_lio(estimator);
                    }
                    
                }
            }
            if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
            {
                m_state.lock();
                if ((diff_vins_lio_t.norm() > 0.1) &&
                    (diff_vins_lio_q.angularDistance(eigen_q::Identity()) * 57.3 > 0.1))
                {
                    estimator.refine_vio_system(diff_vins_lio_q, diff_vins_lio_t);
                    diff_vins_lio_q.setIdentity();
                    diff_vins_lio_t.setZero();
                }
                m_state.unlock();
            }
            unlock_lio(estimator);
            m_state.lock();
            double whole_t = t_s.toc();
            printStatistics(estimator, whole_t);
            header = img_msg->header;
            header.frame_id = "world";
  
            if (g_camera_lidar_queue.m_if_have_lidar_data == false)
            {
                pubOdometry(estimator, header);
            }
            else
            {
                pub_LiDAR_Odometry(estimator, state_aft_integration, header);
            }
            pubCameraPose(estimator, header);
            pubKeyPoses(estimator, header);
            pubPointCloud(estimator, header);
            pubTF(estimator, header);
            pubKeyframe(estimator);
            m_state.unlock();
            if (relo_msg != NULL)
            {
                pubRelocalization(estimator);
            }
        }

        m_estimator.unlock();
        m_buf.lock();
        m_state.lock();
        if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
        {
                update();
        }
        m_state.unlock();
        m_buf.unlock();
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "vins_estimator");
    ros::NodeHandle nh("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
    // ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Debug);
    readParameters(nh);
    estimator.setParameter();

    get_ros_parameter(nh, "/lidar_drag_cam_tim", g_camera_lidar_queue.m_lidar_drag_cam_tim, 1.0);
    get_ros_parameter(nh, "/acc_mul_G", g_camera_lidar_queue.m_if_acc_mul_G, 0);
    get_ros_parameter(nh, "/if_lidar_start_first", g_camera_lidar_queue.m_if_lidar_start_first, 1.0);
    get_ros_parameter<int>(nh, "/if_write_to_bag", g_camera_lidar_queue.m_if_write_res_to_bag, false);
    get_ros_parameter<int>(nh, "/if_dump_log", g_camera_lidar_queue.m_if_dump_log, 0);
    get_ros_parameter<std::string>(nh, "/record_bag_name", g_camera_lidar_queue.m_bag_file_name, "./");
    if(g_camera_lidar_queue.m_if_write_res_to_bag)
    {
        g_camera_lidar_queue.init_rosbag_for_recording();
    }
    // ANCHOR - Start lio process
    g_camera_lidar_queue.m_if_lidar_can_start = false;
    if (estimator.m_fast_lio_instance == nullptr)
    {
        estimator.m_fast_lio_instance = new Fast_lio();
    }
#ifdef EIGEN_DONT_PARALLELIZE
    ROS_DEBUG("EIGEN_DONT_PARALLELIZE");
#endif
    ROS_WARN("waiting for image and imu...");

    registerPub(nh);

    ros::Subscriber sub_imu = nh.subscribe(IMU_TOPIC, 20000, imu_callback, ros::TransportHints().tcpNoDelay());
    ros::Subscriber sub_image = nh.subscribe("/feature_tracker/feature", 20000, feature_callback, ros::TransportHints().tcpNoDelay());
    ros::Subscriber sub_restart = nh.subscribe("/feature_tracker/restart", 20000, restart_callback, ros::TransportHints().tcpNoDelay());
    ros::Subscriber sub_relo_points = nh.subscribe("/pose_graph/match_points", 20000, relocalization_callback, ros::TransportHints().tcpNoDelay());

    std::thread measurement_process{process};
    ros::spin();

    return 0;
}
