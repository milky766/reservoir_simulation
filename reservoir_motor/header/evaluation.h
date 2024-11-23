#ifndef EVALUATION_H
#define EVALUATION_H

#include <Eigen/Dense>
#include <iostream>

// 別名定義
using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;

double corrcoef(const Vector& x, const Vector& y);

#endif // EVALUATION_H
