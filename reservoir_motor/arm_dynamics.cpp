#include "arm_dynamics.h"

Vector arm_dynamics(const Vector& x, const Vector& u, const DynamicsParameters& dynamicsParams) {
    // パラメータの取得
    double m1 = dynamicsParams.m1;
    double m2 = dynamicsParams.m2;
    double b1 = dynamicsParams.b1;
    double b2 = dynamicsParams.b2;
    double l1 = dynamicsParams.L[0];
    double l2 = dynamicsParams.L[1];

    // 入力引数
    Vector torque(2);
    torque << u(0), u(1);
    double theta1 = x(0);
    double theta2 = x(1);
    double theta1dot = x(2);
    double theta2dot = x(3);

    // 慣性行列 M
    double M11 = (m1 + m2) * l1 * l1 + m2 * l2 * l2 + 2 * m2 * l1 * l2 * std::cos(theta2);
    double M12 = m2 * l2 * l2 + m2 * l1 * l2 * std::cos(theta2);
    double M21 = M12;
    double M22 = m2 * l2 * l2;

    Eigen::Matrix2d M;
    M << M11, M12,
         M21, M22;

    // コリオリ力ベクトル V
    double V1 = -m2 * l1 * l2 * (2 * theta1dot * theta2dot + theta2dot * theta2dot) * std::sin(theta2);
    double V2 = m2 * l1 * l2 * theta1dot * theta1dot * std::sin(theta2);

    Vector V(2);
    V << V1, V2;

    // 摩擦ベクトル F
    Vector F(2);
    F << b1 * theta1dot, b2 * theta2dot;

    // 加速度 thetaddot の計算
    Vector thetaddot = M.inverse() * (torque - F - V);

    // 結果を結合
    Vector output(4);
    output << theta1dot, theta2dot, thetaddot(0), thetaddot(1);

    return output;
}
