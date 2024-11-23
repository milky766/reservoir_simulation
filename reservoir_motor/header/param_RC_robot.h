#ifndef PARAM_RC_ROBOT_H
#define PARAM_RC_ROBOT_H

#include <vector>
#include <cmath>
#include <string>
#include <iostream>
#include <Eigen/Dense>

/// タスク関連の設定
struct TaskSettings {
    std::string task = "point"; // "point" or "butterfly"
    std::string perturbation = "impulse"; // "impulse", "obstacle", or "none"
    int interval = 10000;
};

/// 再帰ニューラルネットワーク (RNN) の設定
struct RNNParameters {
    
    int numUnits = 1000;                // 全ユニット数
    int numIn = 2;                     // 入力ユニット数
    int numOut = 2;                    // 出力ユニット数
    double g = 1.5;                     // スペクトル半径
    double noise_amp = 0.0;             // ノイズ振幅
    double p_connect = 0.1;             // スパース接続確率
    double scale;                       // スケールファクタ
    double input_weight_amp = 1.0;      // 入力重みの振幅
    double feedback_weight_amp = 3.0;   // フィードバック重みの振幅


    RNNParameters() {
        scale = g / std::sqrt(p_connect * numUnits);
    }
};

/// 学習関連の設定
struct TrainingSettings {
    double msr_train_noise_amp = 0.0;
    double msr_test_noise_amp = 0.0;
    double sys_noise_amp = 0.0;
    double alpha = 1.0;
    double tau = 10.0; //time constant(10ms)
    int learn_every = 2;
    int n_learn_loops = 10;
    int n_test_loops = 1;
    double delta = 1.0;
    double lambda = 1.0; //逐次最小二乗法の忘却率
};

/// ロボット制御関連の設定
struct RobotControlParameters {
    double L[2] = {1.8, 1.8};
    double Kp = 0.04;
    double Kd = 0.25;
    int arm_dt = 1;

};

/// 入力関連の設定
struct InputSettings {
    double input_pulse_value = 1.0;
    int start_pulse = 200; // (ms)
    int reset_duration = 50; // (ms)
    double dt = 1.0; // numerical integration time step
    std::vector<double> time_axis;
    std::vector<std::vector<double>> input_pattern;

    int start_train = start_pulse + reset_duration;
    int interval = 10000;
    int end_train = start_train + interval;

    int start_train_n = static_cast<int>(start_train / dt);
    int end_train_n = static_cast<int>(end_train / dt);

    int tmax = end_train + 200;
    int n_steps = static_cast<int>(tmax / dt);

    int start_pulse_n = static_cast<int>(start_pulse / dt);
    int reset_duration_n = static_cast<int>(reset_duration / dt);

    InputSettings(int interval) : interval(interval) {
        // 時間軸を生成
        time_axis.resize(n_steps);
        for (int i = 0; i < n_steps; ++i) {
            time_axis[i] = i * dt;
        }

        // 入力パターンを生成
        input_pattern.resize(1, std::vector<double>(n_steps, 0.0));
        for (int i = start_pulse_n; i < start_pulse_n + reset_duration_n; ++i) {
            input_pattern[0][i] = input_pulse_value;
        }
    }
};

/// 出力関連の設定
struct OutputSettings {
    Eigen::MatrixXd target_Out; // 目標出力

    OutputSettings(const std::string& task, int interval, int n_steps, int start_train_n, int end_train_n) 
        : target_Out(2, n_steps) {
        if (task == "point") {
            // タスク "point"
            target_Out.setOnes(); // 全ての出力を 1 に設定
        } else if (task == "butterfly") {
            // タスク "butterfly"
            int numButterfly = interval / 1000;
            int numT = interval / numButterfly;
            Eigen::VectorXd thetas = Eigen::VectorXd::LinSpaced(numT, 0, 2 * M_PI);

            Eigen::VectorXd xout(numT), yout(numT);
            for (int i = 0; i < numT; ++i) {
                double theta = thetas[i];
                double butt = 9 - std::sin(theta) + 2 * std::sin(3 * theta) + 2 * std::sin(5 * theta) 
                              - std::sin(7 * theta) + 3 * std::cos(2 * theta) - 2 * std::cos(4 * theta);
                double scale = 14.4734; // 最大半径
                xout[i] = butt * std::cos(theta) / scale;
                yout[i] = butt * std::sin(theta) / scale;
            }

            // 繰り返し
            for (int i = start_train_n; i < end_train_n; ++i) {
                int idx = (i - start_train_n) % numT;
                target_Out(0, i) = xout[idx];
                target_Out(1, i) = yout[idx];
            }

            // 初期値を 1 に設定
            target_Out.row(0).head(start_train_n).setConstant(xout[0]);
            target_Out.row(1).head(start_train_n).setConstant(yout[0]);
        } else {
            throw std::invalid_argument("Unknown task type: " + task);
        }
    }
};


// 動力学関連のパラメータ
struct DynamicsParameters {
    double m1 = 0.5; // 質量1
    double m2 = 0.5; // 質量2
    double b1 = 0.5; // 摩擦係数1
    double b2 = 0.5; // 摩擦係数2
    std::vector<double> L = {1.8, 1.8}; // リンクの長さ
};

/// 全体のパラメータを統合
struct Parameters {
    TaskSettings taskSettings;
    RNNParameters rnnParams;
    TrainingSettings trainingSettings;
    RobotControlParameters robotControlParams;
    InputSettings inputSettings;
    OutputSettings outputSettings;
    DynamicsParameters dynamicsParams;

    Parameters()
        : inputSettings(taskSettings.interval),
          outputSettings(taskSettings.task, taskSettings.interval, inputSettings.n_steps,
                         inputSettings.start_train_n, inputSettings.end_train_n) {}
};



#endif // PARAM_RC_ROBOT_H
