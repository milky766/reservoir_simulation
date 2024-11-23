#ifndef KINEMATICS_H
#define KINEMATICS_H

#include <Eigen/Dense>
#include <vector>

// 別名定義
using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;

// 関数プロトタイプ
Matrix culc_kinematics(const Vector& angles, const std::vector<double>& L);
Matrix culc_kinematics_elbow(const Vector& angles, const std::vector<double>& L);
Matrix culc_inv_kinematics(const Matrix& xy, const std::vector<double>& L);

#endif // KINEMATICS_H
