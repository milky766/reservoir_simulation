#include "train_RC_robot.h"
#include <iostream>
#include <random>

void train_RC_robot(const Parameters& params, 
                    const Matrix& W, 
                    const Matrix& WIn, 
                    const Matrix& WFb, 
                    Matrix& WOut, 
                    Matrix& P, 
                    const std::vector<std::vector<double>>& input_pattern,
                    const Matrix& target_Out) {

    const auto& rnnParams = params.rnnParams;
    const auto& trainParams = params.trainingSettings;
    const auto& controlParams = params.robotControlParams;
    const auto& ioParams = params.ioSettings;


    // 乱数生成器のセットアップ
    std::default_random_engine generator(std::random_device{}());
    std::normal_distribution<double> normal_dist(0.0, 1.0);

    // メインループ（学習ループ）
    for (int j = 0; j < trainParams.n_learn_loops; ++j) {
        std::cout << "  Loop: " << j + 1 << "/" << trainParams.n_learn_loops << ", ";

        // 初期状態のセットアップ
        Vector Xv = Vector::Random(rnnParams.numUnits); // RNNユニットの初期状態
        Vector X = Xv.array().tanh();                  // 活性化関数適用
        Vector Out = Vector::Zero(rnnParams.numOut);   // 出力初期化
        Vector error_cntl = Vector::Zero(rnnParams.numOut);

        // アームの初期状態
        Vector x0 = Vector::Zero(4); // [theta1, theta2, theta1dot, theta2dot]
        Vector u = Vector::Zero(rnnParams.numOut);
        Matrix arm_x = Matrix::Zero(4, ioParams.n_steps + 1);
        arm_x.col(0) = x0;

        // トレーニングウィンドウフラグ
        bool train_window = false;

        // 時間ステップのループ
        for (int i = 0; i < ioParams.n_steps; ++i) {
            // 入力をセットアップ
            Vector Input = Eigen::Map<const Vector>(input_pattern[i].data(), input_pattern[i].size());

            // RNNユニットの更新
            Vector Xv_current = W * X + WIn * Input + WFb * Out;
            Xv += ((-Xv + Xv_current) / trainParams.alpha) * ioParams.dt;
            X = Xv.array().tanh();

            // 出力計算
            Out = WOut * X + trainParams.alpha * arm_x.block(0, i, 2, 1);

            // PD制御の計算
            Vector error_prev = error_cntl;
            error_cntl = Out - arm_x.block(0, i, 2, 1);
            Vector dif = (error_cntl - error_prev) / controlParams.arm_dt;
            u = controlParams.Kp * error_cntl + controlParams.Kd * dif;

            // アームの更新
            if (i >= ioParams.start_pulse) {
                arm_x.col(i + 1) = arm_x.col(i) + arm_dynamics(arm_x.col(i), u, controlParams.L) * ioParams.dt;
            }

            // 学習ウィンドウの開始・終了
            if (i == ioParams.start_pulse) train_window = true;
            if (i == ioParams.tmax) train_window = false;

            // 読み出しの学習
            if (train_window && (i % trainParams.learn_every == 0)) {
                Vector error = Out - target_Out.col(i);

                // RLS (Recursive Least Squares) の更新
                Vector P_old_X = P * X;
                double den = 1.0 + X.dot(P_old_X);
                P -= (P_old_X * P_old_X.transpose()) / den;
                WOut -= (error * P_old_X.transpose()) / den;
            }
        }

        // パフォーマンス評価
        for (int n = 0; n < rnnParams.numOut; ++n) {
            double r2 = corrcoef(arm_x.block(n, ioParams.start_pulse, 1, ioParams.tmax - ioParams.start_pulse).transpose(),
                                 target_Out.row(n));
            std::cout << "R^2=" << r2 << std::endl;
        }
    }
}


// 相関係数の関数
double corrcoef(const Eigen::VectorXd& x, const Eigen::VectorXd& y) {
    double mean_x = x.mean();
    double mean_y = y.mean();
    double cov = ((x.array() - mean_x) * (y.array() - mean_y)).mean();
    double std_x = std::sqrt(((x.array() - mean_x).square()).mean());
    double std_y = std::sqrt(((y.array() - mean_y).square()).mean());
    return cov / (std_x * std_y);
}
