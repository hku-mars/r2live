#ifndef __TOOLS_DATA_IO_HPP__
#define __TOOLS_DATA_IO_HPP__
#include <iostream>
#include <Eigen/Eigen>
#include <string>
#include <vector>
namespace Common_tools
{
using std::cout;
using std::endl;

inline void save_matrix_to_txt( std::string file_name, eigen_mat< -1, -1 > mat )
{

    FILE *fp = fopen( file_name.c_str(), "w+" );
    int   cols_size = mat.cols();
    int   rows_size = mat.rows();
    for ( int i = 0; i < rows_size; i++ )
    {
        for ( int j = 0; j < cols_size; j++ )
        {
            fprintf( fp, "%.15f ", mat( i, j ) );
        }
        fprintf( fp, "\r\n" );
    }
    // cout <<"Save matrix to: "  << file_name << endl;
    fclose( fp );
}

inline void save_matrix_to_txt( std::string file_name, Eigen::SparseMatrix< double > mat )
{
    save_matrix_to_txt( file_name, mat.toDense() );
}

template<typename T=float>
inline Eigen::Matrix<T, -1, -1> mat_from_data_vec(const std::vector< std::vector< T > > & data_vec  )
{   
    Eigen::Matrix<T, -1, -1> res_mat;
    if(data_vec.size() ==0 )
    {
        return res_mat;
    }
    res_mat.resize( data_vec.size(), data_vec[ 0 ].size() );
    for ( int i = 0; i < data_vec.size(); i++ )
    {
        for ( int j = 0; j < data_vec[i].size(); j++ )
        {
            res_mat( i, j ) = data_vec[ i ][ j ];
        }
    }
    return res_mat;
}
  
template<typename T=float>
inline std::vector<std::vector<T>> load_data_from_txt(std::string file_name)
{
    static const int DATA_RESERVE_SIZE = 102400;
    std::vector<std::vector<T>> data_mat;
    FILE * fp;
    // cout << "Load date from: " << file_name.c_str() << endl;
    fp = fopen(file_name.c_str(), "r");
    if(fp == nullptr)
    {
        cout << "Can not load data from" << file_name << ", please check!" << endl;
    }
    else
    { 
        char line_str[DATA_RESERVE_SIZE];
        while(fgets(line_str, DATA_RESERVE_SIZE, fp ) != NULL)
        {
            std::vector<T> data_vec;
            data_vec.reserve(1e4);
            T data = -3e8;
            int index = 0;
            // cout << std::string(line_str) ;
            std::stringstream ss(line_str);
            //while(!ss.eof())
            while(!ss.eof())
            {
                if((ss >> data))
                {
                    data_vec.push_back(data);
                }
            }
            data_mat.push_back( data_vec );
        }
        fclose(fp);
    }
    return data_mat;
}

template<typename T=float>
inline Eigen::Matrix<T, -1, -1> load_mat_from_txt(std::string file_name)
{
    return mat_from_data_vec<T>(load_data_from_txt<T>(file_name));
}

}
#endif