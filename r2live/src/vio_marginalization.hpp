#pragma once

#include "tools_data_io.hpp"
#include "tools_eigen.hpp"
#include "tools_timer.hpp"
#include "parameters.h"
// #ifndef WINDOW_SIZE
// #define WINDOW_SIZE 10
// #endif
class vio_marginalization
{
  public:
    Eigen::MatrixXd    A, b;
    // Eigen::MatrixXd    m_linearized_jacobians, m_linearized_residuals;
    Eigen::Matrix<double, -1, -1 , Eigen::DontAlign>    m_linearized_jacobians, m_linearized_residuals;
    std::vector< int > idx_to_margin, idx_to_keep;
    int                m_if_enable_debug = 0;
    Eigen::Vector3d m_Ps[(WINDOW_SIZE + 1)];
    Eigen::Vector3d m_Vs[(WINDOW_SIZE + 1)];
    Eigen::Matrix3d m_Rs[(WINDOW_SIZE + 1)];
    Eigen::Vector3d m_Bas[(WINDOW_SIZE + 1)];
    Eigen::Vector3d m_Bgs[(WINDOW_SIZE + 1)];
    Eigen::Matrix3d m_ric[1];
    Eigen::Vector3d m_tic[1];
    int             m_margin_flag = 0;
    double m_td;
    void               printf_vector( std::vector< int > &vec_int, std::string str = std::string( " " ) )
    {
        cout << str << " ";
        for ( int i = 0; i < vec_int.size(); i++ )
        {
            cout << vec_int[ i ] << ", ";
        }
        cout << endl;
    }

    std::vector< int > find_related_visual_landmark_idx(const Eigen::MatrixXd &mat_hessian, std::vector< int > &idx_to_margin, int mar_pos, int visual_size )
    {
        std::vector< int > res_idx_vector;
        int                para_size = mat_hessian.cols();
        int                visual_start_idx = para_size - visual_size;
        // cout << "Related visual idx: ";
        for ( int idx = visual_start_idx; idx < para_size; idx++ )
        {
            if ( mat_hessian.block( mar_pos, idx, 15, 1 ).isZero() == false )
            {
                idx_to_margin.push_back( idx );
                // cout << idx<< ", ";
            }
        }
        // cout << endl;
        return idx_to_margin;
    }

    void marginalize( const Eigen::MatrixXd &mat_hessian, const Eigen::MatrixXd &mat_residual, std::vector< int > &idx_to_margin, std::vector< int > &idx_to_keep )
    {
        // sorting matrices
        int             to_keep_size = idx_to_keep.size();
        int             to_margin_size = idx_to_margin.size();
        int             raw_hessian_size = mat_hessian.rows();
        Eigen::VectorXd res_residual( to_keep_size + to_margin_size );
        Eigen::MatrixXd temp_hessian( to_keep_size + to_margin_size, raw_hessian_size );
        Eigen::MatrixXd res_hessian( to_keep_size + to_margin_size, to_keep_size + to_margin_size );

        for ( int i = 0; i < idx_to_margin.size(); i++ )
        {
            res_residual.row( i ) = mat_residual.row( idx_to_margin[ i ] );
            temp_hessian.row( i ) = mat_hessian.row( idx_to_margin[ i ] );
        }
        for ( int i = 0; i < idx_to_keep.size(); i++ )
        {
            res_residual.row( i + to_margin_size ) = mat_residual.row( idx_to_keep[ i ] );
            temp_hessian.row( i + to_margin_size ) = mat_hessian.row( idx_to_keep[ i ] );
        }

        for ( int i = 0; i < idx_to_margin.size(); i++ )
        {
            res_hessian.col( i ) = temp_hessian.col( idx_to_margin[ i ] );
        }
        for ( int i = 0; i < idx_to_keep.size(); i++ )
        {
            res_hessian.col(i + to_margin_size) = temp_hessian.col(idx_to_keep[i]);
        }
        if (m_if_enable_debug)
        {
            // res_hessian = Common_tools::load_mat_from_txt< double >( "/home/ziv/temp/mar_hessian_A.txt" ).sparseView();
            // res_residual = Common_tools::load_mat_from_txt< double >( "/home/ziv/temp/mar_residual_B.txt" ).sparseView();

            Common_tools::save_matrix_to_txt("/home/ziv/temp/mar_hessian_A_new.txt", res_hessian);
            Common_tools::save_matrix_to_txt("/home/ziv/temp/mar_residual_B_new.txt", res_residual);
        }
        int m = to_margin_size;
        int n = to_keep_size;

        // cout << "To margin size = " << to_margin_size << ", to keep size = " << to_keep_size << endl;
        Eigen::MatrixXd Amm_inv, Amm;
        Eigen::MatrixXd A, b;
        // Amm = res_hessian.block( 0, 0, m, m );
        Amm = 0.5 * (res_hessian.block(0, 0, m, m) + res_hessian.block(0, 0, m, m).transpose());
        // Amm_inv = Amm.fullPivHouseholderQr().solve(Imm);
        // Amm_inv = Amm.colPivHouseholderQr().solve(Imm);
        Amm_inv = Amm.ldlt().solve( Eigen::MatrixXd::Identity( m, m ) );

        Eigen::SparseMatrix< double > bmm = res_residual.segment( 0, m ).sparseView();
        Eigen::SparseMatrix< double > Amr = res_hessian.block( 0, m, m, n ).sparseView();
        Eigen::SparseMatrix< double > Arm = res_hessian.block( m, 0, n, m ).sparseView();
        Eigen::SparseMatrix< double > Arr = res_hessian.block( m, m, n, n ).sparseView();
        Eigen::SparseMatrix< double > brr = res_residual.segment( m, n ).sparseView();
        // t_thread_summing.tic();
        // Eigen::SparseMatrix<double> Arm_Amm_inv = Arm * (Amm_inv.sparseView());
        Eigen::SparseMatrix< double > Amm_inv_spm = Amm_inv.sparseView();
        Eigen::SparseMatrix< double > Arm_Amm_inv = ( Arm ) *Amm_inv_spm;

        A = ( Arr - Arm_Amm_inv * Amr ).toDense();
        b = ( brr - Arm_Amm_inv * bmm ).toDense();
        m_linearized_jacobians = A.llt().matrixL().transpose();
        // m_linearized_residuals = m_linearized_jacobians.transpose().fullPivHouseholderQr().solve( b );
        // m_linearized_residuals = m_linearized_jacobians.transpose().completeOrthogonalDecomposition().solve( b );
        // if ( ( ( m_linearized_jacobians.transpose() * m_linearized_residuals ).isApprox( b, 1e-5 ) == false ) ||
        //      ( ( m_linearized_jacobians.transpose() * m_linearized_jacobians ).isApprox( A, 1e-5 ) == false ) )
        if(1)
        {
            const double eps = 1e-15;
            // std::cout << "fullPivHouseholderQr Accuracy not enough" << std::endl;
            Eigen::SelfAdjointEigenSolver< Eigen::MatrixXd > saes2( A );
            // printf("Eigen solver costs %f ms\r\n", t_thread_summing.toc());
            // t_thread_summing.tic();
            Eigen::VectorXd S = Eigen::VectorXd( ( saes2.eigenvalues().array() > eps ).select( saes2.eigenvalues().array(), 0 ) );
            Eigen::VectorXd S_inv = Eigen::VectorXd( ( saes2.eigenvalues().array() > eps ).select( saes2.eigenvalues().array().inverse(), 0 ) );
            
            Eigen::VectorXd S_sqrt = S.cwiseSqrt();
            Eigen::VectorXd S_inv_sqrt = S_inv.cwiseSqrt();
            
            m_linearized_jacobians = S_sqrt.asDiagonal() * saes2.eigenvectors().transpose();
            m_linearized_residuals = S_inv_sqrt.asDiagonal() * saes2.eigenvectors().transpose() * b;
        }
        if ( m_if_enable_debug )
        {
        Common_tools::save_matrix_to_txt("/home/ziv/temp/mar_linearized_jac_new.txt", m_linearized_jacobians);
        Common_tools::save_matrix_to_txt( "/home/ziv/temp/mar_linearized_res_new.txt", m_linearized_residuals );
        }
        // m_linearized_residuals = m_linearized_jacobians.transpose().ldlt().solve( b );
    }

    void margin_oldest_frame( const Eigen::MatrixXd &mat_hessian, const Eigen::MatrixXd &mat_res, int visual_size )
    {
        m_margin_flag = 0 ;
        int para_size = mat_hessian.cols();
        int visual_start_idx = para_size - visual_size;
        idx_to_margin.clear();
        idx_to_keep.clear();
        int mar_pos = 0;

        idx_to_margin.reserve( 1000 );
        idx_to_keep.reserve( 1000 );
        for ( int idx = mar_pos; idx < mar_pos + 15; idx++ )
        {
            idx_to_margin.push_back( idx );
        }

        // Pose[1] Speed_bias[1]
        for ( int idx = mar_pos + 15; idx < mar_pos + 30; idx++ )
        // for ( int idx = mar_pos + 15; idx < mar_pos + 21; idx++ )
        {
            idx_to_keep.push_back( idx );
        }
        // Pose[2], Pose[3], ... , Pose[WINDOW_SIZE]
        for ( int pose_idx = 2; pose_idx < WINDOW_SIZE + 1; pose_idx++ )
        {
            for ( int idx = pose_idx * 15; idx < pose_idx * 15 + 6; idx++ )
            {
                idx_to_keep.push_back( idx );
            }
        }
        // Extrinsic + Td
        // cout << "Extrinsic: ";
        for (int idx = visual_start_idx - 7; idx < visual_start_idx; idx++)
        {
            // cout << idx << ", ";
            idx_to_keep.push_back(idx);
        }
        // cout << endl;
        std::vector< int > related_visual_idx = find_related_visual_landmark_idx( mat_hessian, idx_to_margin, mar_pos, visual_size );
        if ( m_if_enable_debug )
        {
            cout << "=======Mar Old==========" << endl;
            cout << "Related total visual landmark: " << idx_to_margin.size() - 15 << endl;
            cout << "Total margin size = " << idx_to_margin.size() << endl;
            cout << "Total keep size = " << idx_to_keep.size() << endl;
            printf_vector( idx_to_margin, "To margins: " );
            printf_vector( idx_to_keep, "To keep: " );
        }
        marginalize( mat_hessian, mat_res, idx_to_margin, idx_to_keep );
    }

    void margin_second_new_frame( const Eigen::MatrixXd &mat_hessian, const Eigen::MatrixXd &mat_res, int visual_size )
    {
         m_margin_flag = 1 ;
        int                para_size = mat_hessian.cols();
        int                visual_start_idx = para_size - visual_size;
        int                mar_pos = ( WINDOW_SIZE - 1 ) * 15;
        std::vector< int > idx_to_margin, idx_to_keep;
        idx_to_margin.clear();
        idx_to_keep.clear();
        idx_to_margin.reserve( 1000 );
        idx_to_keep.reserve( 1000 );
        // Pose[WINDOW_SIZE-1]
        for ( int idx = mar_pos; idx < mar_pos + 6; idx++ )
        {
            idx_to_margin.push_back( idx );
        }

        // Pose[0] Speed_bias[0]
        for ( int idx = 0; idx < 15; idx++ )
        {
            idx_to_keep.push_back( idx );
        }
        // Pose[1], Pose[2], ... , Pose[WINDOW_SIZE-1]
        for ( int pose_idx = 1; pose_idx < WINDOW_SIZE - 1; pose_idx++ )
        {
            for ( int idx = pose_idx * 15; idx < pose_idx * 15 + 6; idx++ )
            {
                idx_to_keep.push_back( idx );
            }
        }

        for ( int idx = visual_start_idx - 7; idx < visual_start_idx; idx++ )
        {
            idx_to_keep.push_back( idx );
        }
        if ( m_if_enable_debug )
        {
            cout << "=======Mar Last==========" << endl;
            cout << "Related total visual landmark: " << 0 << endl;
            cout << "Total margin size = " << idx_to_margin.size() << endl;
            cout << "Total keep size = " << idx_to_keep.size() << endl;
            printf_vector( idx_to_margin, "To margins: " );
            printf_vector( idx_to_keep, "To keep: " );
        }
        marginalize( mat_hessian, mat_res, idx_to_margin, idx_to_keep );
    }
};