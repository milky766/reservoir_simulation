#include "train_RC_robot.h"
#include "arm_dynamics.h"
#include "evaluation.h"
#include <iostream>
#include <random>


void train_RC_robot(const Parameters& params, 
                    const Matrix& W, 
                    const Matrix& WIn, 
                    const Matrix& WFb, 
                    Matrix& WOut, 
                    Matrix& P) {

    const auto& rnnParams = params.rnnParams;
    const auto& trainParams = params.trainingSettings;
    const auto& controlParams = params.robotControlParams;
    const auto& inputParams = params.inputSettings;
    const auto& target_Out = params.outputSettings.target_Out;
    const auto& input_pattern = params.inputSettings.input_pattern;
    const auto& dynamicparams = params.dynamicsParams;

    double inv_lambda = 1.0 / trainParams.lambda;


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

         // デバッグ: 初期状態確認
        std::cout << "Debug: Initialized RNN state. Xv size: " 
                  << Xv.rows() << "x" << Xv.cols() << std::endl;


        // アームの初期状態
        Vector x0 = Vector::Zero(4); // [theta1, theta2, theta1dot, theta2dot]
        Vector u = Vector::Zero(rnnParams.numOut);
        Matrix arm_x = Matrix::Zero(4, inputParams.n_steps + 1);

        arm_x.col(0) = x0;
        Vector error_cntl = Vector::Zero(rnnParams.numOut);

        // トレーニングウィンドウフラグ
        bool train_window = false;

        // 時間ステップのループ
        for (int i = 0; i < inputParams.n_steps; ++i) {
            // 入力をセットアップ
            Vector Input = Eigen::Map<const Vector>(input_pattern[i].data(), input_pattern[i].size()); //200~250msの間1の入力がある（インパルス）

            
            // デバッグ: 入力の確認
            std::cout << "Debug: Input size: " << Input.rows() << "x" << Input.cols() << std::endl;
            
            // デバッグ: RNNユニット更新時の行列サイズ確認
            std::cout << "Debug: W size: " << W.rows() << "x" << W.cols() << std::endl;
            std::cout << "Debug: X size: " << X.rows() << "x" << X.cols() << std::endl;
            std::cout << "Debug: WIn size: " << WIn.rows() << "x" << WIn.cols() << std::endl;
            std::cout << "Debug: Input size: " << Input.rows() << "x" << Input.cols() << std::endl;
            std::cout << "Debug: WFb size: " << WFb.rows() << "x" << WFb.cols() << std::endl;
            std::cout << "Debug: Out size: " << Out.rows() << "x" << Out.cols() << std::endl;

            // RNNユニットの更新
            Vector Xv_current = W * X + WIn * Input + WFb * Out;
            Xv += ((-Xv + Xv_current) / trainParams.tau) * inputParams.dt;
            X = Xv.array().tanh();

   


            // 出力計算
            Out = WOut * X + trainParams.alpha * arm_x.block(0, i, 2, 1);

             // デバッグ: 出力計算確認
            std::cout << "Debug: WOut size: " << WOut.rows() << "x" << WOut.cols() << std::endl;
            std::cout << "Debug: X size: " << X.rows() << "x" << X.cols() << std::endl;


            // PD制御の計算
            Vector error_prev = error_cntl;
            error_cntl = Out - arm_x.block(0, i, 2, 1);
            Vector dif = (error_cntl - error_prev) / controlParams.arm_dt;
            u = controlParams.Kp * error_cntl + controlParams.Kd * dif;

            if (i<inputParams.start_pulse_n) {
                u = Vector::Zero(rnnParams.numOut);
            }

            arm_x.col(i + 1) = arm_x.col(i) + arm_dynamics(arm_x.col(i), u, dynamicparams) * inputParams.dt;
            Vector arm = arm_x.col(i + 1) + trainParams.msr_train_noise_amp * Vector::Random(4);

            // 学習ウィンドウの開始・終了
            if (i == inputParams.start_train_n) train_window = true;
            if (i == inputParams.end_train_n) train_window = false;

            // 読み出しの学習
            if (train_window && (i % trainParams.learn_every == 0)) {
                Vector error = Out - target_Out.col(i);

                // RLS (Recursive Least Squares) の更新
                Vector P_old_X = P * X;
                double den = 1.0 + X.dot(P_old_X)*inv_lambda;
                P = P * inv_lambda - (P_old_X * P_old_X.transpose()) * inv_lambda/ den;
                WOut -= (error * P_old_X.transpose()) / den;
            }
        }

        // パフォーマンス評価(start_pulseから最後までの推定値と実値の相関係数)
        for (int n = 0; n < rnnParams.numOut; ++n) {
            double r2 = corrcoef(arm_x.block(n, inputParams.start_pulse, 1, inputParams.tmax - inputParams.start_pulse).transpose(),
                                 target_Out.row(n));
            std::cout << "R^2=" << r2 << std::endl;
        }
    }
}


