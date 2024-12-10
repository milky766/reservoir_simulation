#include "param_RC_robot.h"
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "param_RC_robot.h" 
#include "construct_network.h"
#include "train_RC_robot.h"
#include "test_RC_robot.h"

int main() {
    std::cout << "Program started!" << std::endl;
    Parameters params;
    Matrix WIn, WFb, WOut, P;
    SparseMatrix W;

    std::cout << "Before function call1" << std::endl;
    construct_network(params, WIn, WFb, WOut, P, W);

    std::cout << "Before function call2" << std::endl;
    train_RC_robot(params, W, WIn, WFb, WOut, P);

    std::cout << "Before function call3" << std::endl;
    test_RC_robot(params, W, WIn, WFb, WOut, "impulse");
    
    std::cout << "Program ended!" << std::endl;
    return 0;
}

