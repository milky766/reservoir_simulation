#include "param_RC_robot.h"
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "param_RC_robot.h" 
#include "construct_network.h"
#include "train_RC_robot.h"
#include "test_RC_robot.h"

int main() {
    Parameters params;
    Matrix WIn, WFb, WOut, P;
    SparseMatrix W;

    construct_network(params, WIn, WFb, WOut, P, W);
    train_RC_robot(params, W, WIn, WFb, WOut, P);
    test_RC_robot(params, W, WIn, WFb, WOut, "impulse");

    return 0;
}
