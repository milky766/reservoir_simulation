#include "kinematics.h"
#include <cmath>
#include <complex>

// 末端リンクのキネマティクス計算
Matrix culc_kinematics(const Vector& angles, const std::vector<double>& L) {
    double theta1 = angles(0);
    double theta2 = angles(1);

    double l1 = L[0];
    double l2 = L[1];

    // 計算結果を行列として返す
    Matrix xy(2, 1);
    xy(0, 0) = l1 * std::cos(theta1) + l2 * std::cos(theta1 + theta2);
    xy(1, 0) = l1 * std::sin(theta1) + l2 * std::sin(theta1 + theta2);

    return xy;
}

// 肘リンクのキネマティクス計算
Matrix culc_kinematics_elbow(const Vector& angles, const std::vector<double>& L) {
    double theta1 = angles(0);

    double l1 = L[0];

    // 計算結果を行列として返す
    Matrix xy(2, 1);
    xy(0, 0) = l1 * std::cos(theta1);
    xy(1, 0) = l1 * std::sin(theta1);

    return xy;
}

// 逆運動学の計算
Matrix culc_inv_kinematics(const Matrix& xy, const std::vector<double>& L) {
    double l1 = L[0];
    double l2 = L[1];

    // 入力のx, y座標を取得
    Vector x = xy.row(0).array() + 0;  // x座標
    Vector y = xy.row(1).array() + 2; // y座標
    Vector l0 = (x.array().square() + y.array().square()).sqrt();

    // theta2の計算
    Vector cos_theta2 = -((x.array().square() + y.array().square()) - (l1 * l1 + l2 * l2)) / (2 * l1 * l2);
    Vector sin_theta2 = ((4 * l1 * l1 * l2 * l2) - (l0.array().square() - (l1 * l1 + l2 * l2)).square()).sqrt() / (2 * l1 * l2);

    Vector theta2 = (sin_theta2.array().real().atan2((-cos_theta2).array().real()));

    // kc, ksの計算
    Vector kc = l1 + l2 * theta2.array().cos();
    Vector ks = l2 * theta2.array().sin();

    // theta1の計算
    Vector cos_theta1 = (kc.array() * x.array() + ks.array() * y.array()) / (kc.array().square() + ks.array().square());
    Vector sin_theta1 = (-ks.array() * x.array() + kc.array() * y.array()) / (kc.array().square() + ks.array().square());

    Vector theta1 = sin_theta1.array().atan2(cos_theta1.array());

    // 結果を行列として返す
    Matrix out(2, x.size());
    out.row(0) = theta1;
    out.row(1) = theta2;

    return out;
}
