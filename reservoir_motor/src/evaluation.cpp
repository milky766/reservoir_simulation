#include "evaluation.h"
#include <iostream>

// 相関係数を計算する補助関数
double corrcoef(const Vector& x, const Vector& y) {
    // 平均値を計算
    double mean_x = x.mean();
    double mean_y = y.mean();

    // 共分散
    double cov = ((x.array() - mean_x) * (y.array() - mean_y)).mean();

    // 標準偏差
    double std_x = std::sqrt(((x.array() - mean_x).square()).mean());
    double std_y = std::sqrt(((y.array() - mean_y).square()).mean());

    // 相関係数を返す
    return cov / (std_x * std_y);
}
