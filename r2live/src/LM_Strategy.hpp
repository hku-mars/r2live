#pragma once
#include "tools_ceres.hpp"
#include "tools_eigen.hpp"

const double MIMIMUM_DELTA = 1e-15; 

inline Eigen::SparseMatrix<double> schur_complement_woodbury_matrix(Eigen::SparseMatrix<double> &mat_A, Eigen::SparseMatrix<double> &mat_U,
                                                             Eigen::SparseMatrix<double> &mat_C_inv, Eigen::SparseMatrix<double> &mat_V)
{
    // https://en.wikipedia.org/wiki/Woodbury_matrix_identity
    Eigen::SparseMatrix<double> mat_A_inv = mat_A.toDense().completeOrthogonalDecomposition().pseudoInverse().sparseView();
    Eigen::SparseMatrix<double> mat_mid_inv = (mat_C_inv + mat_V * mat_A_inv * mat_U).toDense().completeOrthogonalDecomposition().pseudoInverse().sparseView();
    return mat_A_inv - mat_A_inv * mat_U * mat_mid_inv * mat_V * mat_A_inv;
}

inline Eigen::SparseMatrix<double> sparse_schur_solver(const Eigen::SparseMatrix<double> &mat_H, const Eigen::SparseMatrix<double> &vec_JtX, int dense_block_size, int solver_type = 0)
{
    //Website: http://ceres-solver.org/nnls_solving.html#equation-hblock
    Eigen::SparseMatrix<double> vec_X;
    int mat_C_blk_size = mat_H.cols() - dense_block_size;
    // cout << "Block size = " << mat_H.rows() << " X " << mat_H.cols() << ", dense_blk_size = " << dense_block_size << endl;

    Eigen::SparseMatrix<double> mat_B, mat_E, mat_Et, mat_C, mat_v, mat_w, mat_C_inv, mat_I, mat_S, mat_S_inv, mat_E_C_inv;
    Eigen::SparseMatrix<double> mat_dy, mat_dz;
    mat_B = mat_H.block(0, 0, dense_block_size, dense_block_size);
    mat_E = mat_H.block(0, dense_block_size, dense_block_size, mat_C_blk_size);
    mat_C = mat_H.block(dense_block_size, dense_block_size, mat_C_blk_size, mat_C_blk_size);
    Start_mat_c:
    mat_v = vec_JtX.block(0, 0, dense_block_size, 1);
    mat_w = vec_JtX.block(dense_block_size, 0, mat_C_blk_size, 1);
    mat_Et = mat_E.transpose();
    mat_C_inv.resize(mat_C_blk_size, mat_C_blk_size);
    
    Eigen::LeastSquaresConjugateGradient<Eigen::SparseMatrix<double>> solver;
    // Eigen::SparseQR< Eigen::SparseMatrix< double >, Eigen::COLAMDOrdering< int > > solver;
    int if_mat_C_invertable = 1;
    for (int i = 0; i < mat_C_blk_size; i++)
    {
        if (fabs(mat_C.coeff(i, i)) < MIMIMUM_DELTA)
        {
            mat_C_inv.insert(i, i) = 0.0;
            if_mat_C_invertable = 0;
        }
        else
        {
            mat_C_inv.insert(i, i) = 1.0 / mat_C.coeff(i, i);
        }
    }

    mat_E_C_inv = mat_E * mat_C_inv;
    // if (1)
    if (if_mat_C_invertable)
    {
        mat_S = mat_B - mat_E_C_inv * mat_Et;
        // https://eigen.tuxfamily.org/dox/group__TutorialLinearAlgebra.html
        Eigen::SimplicialLDLT<SparseMatrix<double>> solver;
        solver.compute(mat_S);
        if(solver.info() == Eigen::Success )
        {
            mat_dy = solver.solve((mat_v - mat_E_C_inv * mat_w));   
        }
        else
        {
            mat_dy = mat_S.toDense().completeOrthogonalDecomposition().solve((mat_v - mat_E_C_inv * mat_w).toDense()).sparseView();
        }
    }
    else
    {
        cout << ANSI_COLOR_CYAN_BOLD <<"Unstable update, perfrom schur_complement_woodbury_matrix" << ANSI_COLOR_RESET << endl;
        mat_S_inv = schur_complement_woodbury_matrix(mat_B, mat_E, mat_C, mat_Et);
        mat_dy = mat_S_inv * (mat_v - mat_E_C_inv * mat_w);
    }

    mat_dz = mat_C_inv * (mat_w - mat_Et * mat_dy);
    Eigen::Matrix<double, -1, -1> vec_X_dense;
    vec_X_dense.resize(vec_JtX.rows(), vec_JtX.cols());
    vec_X_dense.block(0, 0, dense_block_size, 1) = mat_dy.toDense();
    vec_X_dense.block(dense_block_size, 0, mat_C_blk_size, 1) = mat_dz.toDense();
    vec_X = vec_X_dense.sparseView();
    return vec_X;
}


struct LM_trust_region_strategy
{
  public:
    double                        radius_ = 1e-4;
    double                        max_radius_;
    const double                  min_diagonal_ = 1e-15;
    const double                  max_diagonal_ = 1e15;
    double                        decrease_factor_ = 2.0;
    bool                          reuse_diagonal_;
    Eigen::SparseMatrix< double > gradient, hessian, last_step;
    std::vector< double >         cost_history, model_cost_changes_history, real_changes_history, step_quality_history, radius_history;

    LM_trust_region_strategy()
    {
        cost_history.reserve(10);
        model_cost_changes_history.reserve(10);
        real_changes_history.reserve(10);
        step_quality_history.reserve(10);
        radius_history.reserve(10);
    };

    ~LM_trust_region_strategy() = default;
    inline Eigen::SparseMatrix< double > compute_step( Eigen::SparseMatrix< double > &jacobian, Eigen::SparseMatrix< double > &residuals, int dense_block_size = 0 )
    {
        Eigen::SparseMatrix< double > current_step;
        int residual_size = jacobian.rows();
        int parameter_size = jacobian.cols();
        // cout << "Residual size = " << residual_size << ", parasize = " << parameter_size << endl;
        double current_cost = residuals.norm();
        gradient = jacobian.transpose() * residuals;
        hessian = jacobian.transpose() * jacobian;
       
        if ( cost_history.size() && model_cost_changes_history.size() )
        {
            double real_cost_change =  cost_history.back() - current_cost;
            double step_quality = real_cost_change / model_cost_changes_history.back();
            real_changes_history.push_back(real_cost_change);
            step_quality_history.push_back( step_quality );
            if(step_quality > 0) // Step is good, 
            {
                 radius_ = radius_ * std::max(1.0 / 3.0, 1.0 - pow(2.0 * step_quality - 1.0, 3));
                 decrease_factor_ = 2.0;
            }
            else
            {
                radius_ = radius_ * decrease_factor_;
                decrease_factor_ *= 2; 
                current_step = -1 * last_step;
                last_step.setZero();
                radius_history.push_back(radius_);
                cost_history.push_back( current_cost );
            }
        }
        else
        {
            radius_ = 1e-6 * sqrt( std::min( std::max( hessian.diagonal().maxCoeff() , min_diagonal_ ), max_diagonal_ ) );
        }

        radius_history.push_back( radius_ );
        cost_history.push_back( current_cost );

        Eigen::SparseMatrix< double > mat_D( parameter_size, parameter_size );

        mat_D.setZero();
        for (int i = 0; i < parameter_size; i++)
        {
            mat_D.coeffRef(i, i) = sqrt(std::min(std::max(hessian.coeff(i, i), min_diagonal_), max_diagonal_) * radius_);
        }

        Eigen::SparseMatrix< double > hessian_temp = hessian + mat_D ;
        Eigen::SparseMatrix<double> jacobian_mul_step, model_cost_mat;
        if(dense_block_size)
        {
            current_step = sparse_schur_solver( hessian_temp, -gradient, dense_block_size );
        }
        else
        {
            current_step = ( hessian_temp.toDense().completeOrthogonalDecomposition().solve( -gradient.toDense() ) ).sparseView();
        }
        jacobian_mul_step = jacobian * current_step;

        model_cost_mat = (-jacobian_mul_step.transpose() * (residuals + jacobian_mul_step / 2.0));
        double model_cost_chanage = model_cost_mat.norm();
        model_cost_changes_history.push_back( model_cost_chanage );

        if ( model_cost_chanage >= 0 )
        {
            last_step = current_step;
            return current_step;
        }
        else
        {
            cout << ANSI_COLOR_RED_BOLD << "Model_cost_chanage = " << model_cost_chanage << " is negative, something error here, please check!!!" << endl;
            return current_step;
        }
    }

    void printf_history()
    {
        if ( cost_history.size() )
        {
            cout << "===== History cost =====" << endl;
            cout << " Iter | Cost | M_cost | R_cost | S_Q | Rad " << endl;
            cout << "[ " << 0 << "] " << cost_history[ 0 ] << " --- " << endl;
            for ( int i = 0; i < cost_history.size() - 1; i++ )
            {
                cout << "[ " << i << "] " << cost_history[ i + 1 ] << " | "
                     << model_cost_changes_history[ i ] << " | "
                     << real_changes_history[ i ] << " | "
                     << step_quality_history[ i ] << " | "
                     << radius_history[ i ] 
                     << endl;
            }
        }
    }
};
