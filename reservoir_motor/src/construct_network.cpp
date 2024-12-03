#include "construct_network.h"
#include <iostream>
#include <random>

void construct_network(const Parameters& params,
                       Matrix& WIn,
                       Matrix& WFb,
                       Matrix& WOut,
                       Matrix& P,
                       SparseMatrix& W) {

    int numUnits = params.rnnParams.numUnits;
    int numIn = params.rnnParams.numIn;
    int numOut = params.rnnParams.numOut;
    double p_connect = params.rnnParams.p_connect;
    double scale = params.rnnParams.scale;

    // 乱数生成器
    int seed = 1;
    std::default_random_engine generator(seed);
    std::normal_distribution<double> normal_dist(0.0, 1.0); //標準分布　平均０，分散１
    std::uniform_real_distribution<double> uniform_dist(0.0, 1.0); //一様分布

    // 入力接続 (WIn)
    WIn = Matrix::Zero(numUnits, numIn);
    for (int i = 0; i < numUnits; ++i) {
        for (int j = 0; j < numIn; ++j) {
            WIn(i, j) = normal_dist(generator) * params.rnnParams.input_weight_amp;
        }
    }

    // フィードバック接続 (WFb)
    WFb = Matrix::Zero(numUnits, numOut);
    for (int i = 0; i < numUnits; ++i) {
        for (int j = 0; j < numOut; ++j) {
            WFb(i, j) = normal_dist(generator) * params.rnnParams.feedback_weight_amp;
        }
    }

    // 初期リードアウト接続 (WOut)
    WOut = Matrix::Zero(numOut, numUnits);

    // P行列 (リードアウト学習用)
    P = Matrix::Identity(numUnits, numUnits) / params.trainingSettings.delta;

    // 再帰接続 (W)
    SparseMatrix W_sparse(numUnits, numUnits);
    for (int i = 0; i < numUnits; ++i) {
        for (int j = 0; j < numUnits; ++j) {
            if (uniform_dist(generator) <= p_connect) {
                double value = normal_dist(generator) * scale;
                if (i != j) { // 自己接続を防ぐ
                    W_sparse.insert(i, j) = value;
                }
            }
        }
    }
    W_sparse.makeCompressed();
    W = W_sparse;

    std::cout << "RNN construction completed.\n";
}
