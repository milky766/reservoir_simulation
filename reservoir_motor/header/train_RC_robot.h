#ifndef TRAIN_RC_ROBOT_H
#define TRAIN_RC_ROBOT_H

#include <Eigen/Dense>
#include <vector>
#include "param_RC_robot.h"

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;

// 関数プロトタイプ
void train_RC_robot(const Parameters& params, 
                    const Matrix& W, 
                    const Matrix& WIn, 
                    const Matrix& WFb, 
                    Matrix& WOut, 
                    Matrix& P, 
                    const std::vector<std::vector<double>>& input_pattern,
                    const Matrix& target_Out);


double corrcoef(const Eigen::VectorXd& x, const Eigen::VectorXd& y);

#endif // TRAIN_RC_ROBOT_H
