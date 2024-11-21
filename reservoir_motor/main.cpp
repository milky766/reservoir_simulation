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


    // 学習データ（入力パターンと目標出力）を準備
    std::vector<std::vector<double>> input_pattern = ...; // データを用意
    Matrix target_Out = ...; // データを用意

    // RNNの学習
    train_RC_robot(params, W, WIn, WFb, WOut, P, input_pattern, target_Out);

    return 0;
}
