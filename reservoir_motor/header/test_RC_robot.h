#ifndef TEST_RC_ROBOT_H
#define TEST_RC_ROBOT_H

#include <Eigen/Dense>
#include <vector>
#include <string>
#include "param_RC_robot.h"

// 別名定義
using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;

// 関数プロトタイプ
void test_RC_robot(const Parameters& params, 
                   const Matrix& W, 
                   const Matrix& WIn, 
                   const Matrix& WFb, 
                   const Matrix& WOut,
                   const std::string& perturbation);

#endif // TEST_RC_ROBOT_H
