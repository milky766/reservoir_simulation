#ifndef CONSTRUCT_NETWORK_H
#define CONSTRUCT_NETWORK_H

#include "param_RC_robot.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>

// Eigenを使った行列型の定義
using Matrix = Eigen::MatrixXd;
using SparseMatrix = Eigen::SparseMatrix<double>;

// 関数の宣言
void construct_network(const Parameters& params,
                       Matrix& WIn,
                       Matrix& WFb,
                       Matrix& WOut,
                       Matrix& P,
                       SparseMatrix& W);

#endif // CONSTRUCT_NETWORK_H
