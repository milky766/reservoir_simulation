#ifndef ARM_DYNAMICS_H
#define ARM_DYNAMICS_H

#include <Eigen/Dense>
#include "param_RC_robot.h"

using Vector = Eigen::VectorXd;

// アームの動力学計算
Vector arm_dynamics(const Vector& x, const Vector& u, const DynamicsParameters& dynamicsParams);

#endif // ARM_DYNAMICS_H
