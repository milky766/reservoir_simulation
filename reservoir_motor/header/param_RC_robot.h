#ifndef PARAM_RC_ROBOT_H
#define PARAM_RC_ROBOT_H

#include <vector>
#include <cmath>
#include <string>
#include <iostream>

/// タスク関連の設定
struct TaskSettings {
    std::string task = "point"; // "point" or "butterfly"
    std::string perturbation = "impulse"; // "impulse", "obstacle", or "none"
    int interval = 1000;
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
    int learn_every = 2;
    int n_learn_loops = 10;
    int n_test_loops = 1;
    double delta = 1.0;
};

/// ロボット制御関連の設定
struct RobotControlParameters {
    double L[2] = {1.8, 1.8};
    double Kp = 0.04;
    double Kd = 0.25;
    int arm_dt = 1;

};

/// 入力と出力関連の設定
struct InputOutputSettings {
    double input_pulse_value = 1.0;
    int start_pulse = 200; // (ms)
    int reset_duration = 50; // (ms)
    double ready_level = 0.2;
    double peak_level = 1.0;
    double peak_width = 30.0;

    double dt = 1.0; // numerical integration time step
    int tmax;
    int n２ｈ３４５６7８9０−7*89ｍ，ｋ．/_steps;
    std::vector<double> time_axis;
    std::vector<std::vector<double>> input_pattern;

    InputOutputSettings(int interval) {
        int start_train = start_pulse + reset_duration;
        int end_train = start_train + interval;
        tmax = end_train + 200;
        n_steps = static_cast<int>(tmax / dt);

        // 時間軸を生成
        time_axis.resize(n_steps);
        for (int i = 0; i < n_steps; ++i) {
            time_axis[i] = i * dt;
        }

        // 入力パターンを生成
        input_pattern.resize(1, std::vector<double>(n_steps, 0.0));
        int start_pulse_n = static_cast<int>(start_pulse / dt);
        int reset_duration_n = static_cast<int>(reset_duration / dt);
        for (int i = start_pulse_n; i < start_pulse_n + reset_duration_n; ++i) {
            input_pattern[0][i] = input_pulse_value;
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
    InputOutputSettings ioSettings;
    DynamicsParameters dynamicsParams;

    Parameters()
        : ioSettings(taskSettings.interval) {}

};

#endif // PARAM_RC_ROBOT_H
