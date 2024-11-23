#include "test_RC_robot.h"
#include "evaluation.h"
#include "arm_dynamics.h"
#include <iostream>
#include <random>

void test_RC_robot(const Parameters& params, 
                   const Matrix& W, 
                   const Matrix& WIn, 
                   const Matrix& WFb, 
                   const Matrix& WOut,
                   const std::string& perturbation) {

    const auto& rnnParams = params.rnnParams;
    const auto& trainParams = params.trainingSettings;
    const auto& controlParams = params.robotControlParams;
    const auto& inputParams = params.inputSettings;
    const auto& outputParams = params.outputSettings;

    std::cout << "Testing:" << std::endl;

    // 乱数生成器のセットアップ
    std::default_random_engine generator(std::random_device{}());
    std::normal_distribution<double> normal_dist(0.0, 1.0);

    Matrix Out_test_history = Matrix::Zero(rnnParams.numOut, inputParams.n_steps);
    Matrix error_history = Matrix::Zero(rnnParams.numOut, inputParams.n_steps);
    Matrix u_history = Matrix::Zero(rnnParams.numOut, inputParams.n_steps);

    for (int j = 0; j < trainParams.n_test_loops; ++j) {
        std::cout << "  Loop: " << j + 1 << "/" << trainParams.n_test_loops << ", " << std::endl;

        // 初期条件の設定
        Vector Xv = Vector::Random(rnnParams.numUnits);
        Vector X = Xv.array().tanh();
        Vector Xv_current = Xv;
        Vector Out = Vector::Zero(rnnParams.numOut);

        Vector x0 = Vector::Zero(4);
        Vector u = Vector::Zero(rnnParams.numOut);
        Matrix arm_x = Matrix::Zero(4, inputParams.n_steps + 1);
        arm_x.col(0) = x0;

        Vector arm = arm_x.col(0) + trainParams.msr_test_noise_amp * Vector::Random(4);
        Vector error_cntl = Vector::Zero(rnnParams.numOut);

        for (int i = 0; i < inputParams.n_steps; ++i) {
            if (i % 100 == 0) {
                std::cout << "    Step: " << i + 1 << "/" << inputParams.n_steps << std::endl;
            }

            Vector Input = Eigen::Map<const Vector>(inputParams.input_pattern[i].data(), inputParams.input_pattern[i].size());

            // RNNの更新
            Xv_current = W * X + WIn * Input + WFb * Out;
            Xv += ((-Xv + Xv_current) / trainParams.tau) * inputParams.dt;
            X = Xv.array().tanh();

            // 出力計算
            Out = WOut * X + trainParams.alpha * arm.head(2);
            Out_test_history.col(i) = Out;

            // PD制御
            Vector error_prev = error_cntl;
            error_cntl = Out - arm.head(2);
            Vector dif = (error_cntl - error_prev) / controlParams.arm_dt;
            u = controlParams.Kp * error_cntl + controlParams.Kd * dif;

            if (i < inputParams.start_train_n) {
                u.setZero();
            }

            error_history.col(i) = error_cntl;
            u_history.col(i) = u;

            // Perturbation
            if (perturbation == "impulse" && i == (inputParams.start_train_n + 500)) {
                u -= 0.5 * Vector::Ones(rnnParams.numOut);
            } else if (perturbation == "obstacle" && arm_x(0, i) >= 0.5 && i <= (inputParams.start_train_n + 500)) {
                u.setZero();
                arm_x.block(2, i, 2, 1).setZero(); // 速度をゼロに
            }

            // システムノイズを加える
            if (i >= inputParams.start_train_n) {
                u += trainParams.sys_noise_amp * Vector::Random(rnnParams.numOut);
            }

            // アーム動力学の更新
            arm_x.col(i + 1) = arm_x.col(i) + arm_dynamics(arm_x.col(i), u, controlParams.L) * inputParams.dt;
            arm = arm_x.col(i + 1) + trainParams.msr_test_noise_amp * Vector::Random(4);
        }

        // パフォーマンス評価
        for (int n = 0; n < rnnParams.numOut; ++n) {
            double r2 = corrcoef(Out_test_history.row(n).segment(inputParams.start_train_n, inputParams.end_train_n - inputParams.start_train_n),
                                 outputParams.target_Out.row(n).segment(inputParams.start_train_n, inputParams.end_train_n - inputParams.start_train_n));
            std::cout << "R^2=" << r2 << std::endl;
        }
    }
}
