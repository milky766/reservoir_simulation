#include "param_RC_robot.h"
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "param_RC_robot.h" 
#include "construct_network.h"

int main() {
    // 全体のパラメータを初期化
    Parameters params;

    // ネットワーク構築用の行列
    Matrix WIn, WFb, WOut, P;
    SparseMatrix W;

    // ネットワークの構築
    construct_network(params, WIn, WFb, WOut, P, W);

    return 0;
}
