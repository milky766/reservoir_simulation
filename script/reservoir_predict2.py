import numpy as np
import matplotlib.pyplot as plt
from model_new import ESN, Tikhonov, generate_teaching_trajectories_move, dynamic_target_angle_sin # model.pyのESNとTikhonovクラスをインポート

# ESNおよびロボットモデルのパラメータ設定
N_x = 400               # リザバーのノード数
input_scale = 0.001       # 入力スケーリング　リザバーへの入力信号の強さ
density = 0.1           # 結合密度
rho = 0.95              # スペクトル半径
leaking_rate = 0.99     # リーキング率
beta = 0.0001           # リッジ回帰の正則化係数
activation_function = np.tanh #活性化関数

# ロボットのパラメータ
link_lengths = [0.1, 0.1]  # 各リンクの長さ (0.1 m)
KP = np.diag([15.0, 15.0])  # PDコントローラの比例ゲイン
KD = np.diag([0.8, 0.8])    # PDコントローラの微分ゲイン
dt = 0.01                   # 積分時間ステップ (10ms)
steps = 1000                # シミュレーションステップ数

# 目標位置（関節角度をラジアンで指定）
target_angles = np.array([np.pi / 2, np.pi / 2])  # 目標位置 (45度)
center_angles = np.array([0.0, 0.0])             # 初期値分布の中心

# 目標位置と時間に応じて変化させるための設定
start_angle = np.array([np.pi / 4, np.pi / 4])  # 開始時の目標角度 (0, 0)
end_angle = target_angles           # 最終目標角度 (45度, 45度)

# 順運動学に基づく目標先端位置
def forward_kinematics(angles):
    x = link_lengths[0] * np.cos(angles[0]) + link_lengths[1] * np.cos(angles[0] + angles[1])
    y = link_lengths[0] * np.sin(angles[0]) + link_lengths[1] * np.sin(angles[0] + angles[1])
    return np.array([x, y])

# 開始位置と終了位置を計算
start_position = forward_kinematics(start_angle)
end_position = forward_kinematics(end_angle)

# ランダムな初期位置とトルクの生成（角度）
np.random.seed(0)
initial_positions_angles = [center_angles + np.random.uniform(0.0, np.pi, size=2) for _ in range(10)] #異なる初期角度のセット10個が格納

# ティーチング軌道の生成
teaching_trajectories, teaching_torques = generate_teaching_trajectories_move(initial_positions_angles, start_angle, end_angle, steps, KP, KD, dt) #各初期位置から目標位置に到達するための軌道データが格納

# ティーチング軌道からESNの学習データを作成   np.vstack(...): 各軌道のデータを垂直方向に連結して、大きなデータセットを生成
u = np.vstack([traj[:-1] for traj in teaching_trajectories])  # 入力としての軌道データ(t)  (0,1,2,3)
d = np.vstack([traj[1:] for traj in teaching_trajectories])   # 出力としての軌道データ(t+1)(1,2,3,4)

# ESNの初期化
N_u = u.shape[1]  # 入力次元
N_y = d.shape[1]  # 出力次元
esn = ESN(N_u, N_y, N_x, density=density, input_scale=input_scale, rho=rho,
          activation_func=activation_function, leaking_rate=leaking_rate)

# Tikhonov正則化(リッジ回帰)による学習
optimizer = Tikhonov(N_x, N_y, beta)
esn.train(u, d, optimizer)

# 新しい初期点から目標位置への軌道生成（自律走行）とトルク計算
np.random.seed(2)
test_initial_positions_angles = [center_angles + np.random.uniform(0.0, np.pi, size=2) for _ in range(10)]
esn_trajectories = []
esn_torques = []

esn_real_trajectories = []
esn_trajectories_error = []

for pos in test_initial_positions_angles:
    esn_predicted_angle_seq , esn_torque_seq, esn_real_angle_seq, esn_error_seq = esn.trajectory1(pos, steps, KP, KD, dt, target_angles)
    
    esn_trajectories.append(np.array(esn_predicted_angle_seq))
    esn_torques.append(np.array(esn_torque_seq))
    esn_real_trajectories.append(np.array(esn_real_angle_seq))
    esn_trajectories_error.append(np.array(esn_error_seq))

# カラーマップを用意
cmap = plt.cm.rainbow
esn_colors = cmap(np.linspace(0, 1, len(esn_trajectories)))
teaching_colors = cmap(np.linspace(0, 1, len(teaching_trajectories)))

# 4つのプロットのセットアップ
fig, axs = plt.subplots(2, 4, figsize=(15, 10))
axs = axs.flatten()  # axsを1次元配列に変換

# target_angleの時間経過を計算
target_angles_over_time = [dynamic_target_angle_sin(step, steps, start_angle, end_angle) for step in range(steps)]
target_angles_over_time = np.array(target_angles_over_time)
target_positions_over_time = np.array([forward_kinematics(angle) for angle in target_angles_over_time])

# 1. ESNによる関節角度の動作生成
axs[0].set_title("ESN-based Adaptive Motion Generation (angles)")
for idx, traj in enumerate(esn_trajectories):
    axs[0].plot(traj[:, 0], traj[:, 1], color=esn_colors[idx], alpha=0.7)
    axs[0].scatter(traj[0, 0], traj[0, 1], color='green', marker='o')  # 初期位置
axs[0].scatter(start_angle[0], start_angle[1], color='black', marker='o', label="Start Angle")
axs[0].scatter(end_angle[0], end_angle[1], color='black', marker='x', label="End Angle")
axs[0].plot(target_angles_over_time[:, 0], target_angles_over_time[:, 1], 'k--', label="Target Angle Path Over Time")  # 開始から終了の経路
axs[0].set_xlabel("Joint Angles1")
axs[0].set_ylabel("Joint Angles2")
axs[0].legend()

# 2. ESNによるXY座標の動作生成
axs[1].set_title("ESN-based Adaptive Motion Generation (XY)")
for idx, traj in enumerate(esn_trajectories):
    xy_traj = np.array([forward_kinematics(angles) for angles in traj])
    axs[1].plot(xy_traj[:, 0], xy_traj[:, 1], color=esn_colors[idx], alpha=0.7)
    axs[1].scatter(xy_traj[0, 0], xy_traj[0, 1], color='green', marker='o')
axs[1].scatter(start_position[0], start_position[1], color='black', marker='o', label="Start Position")
axs[1].scatter(end_position[0], end_position[1], color='black', marker='x', label="End Position")
axs[1].plot(target_positions_over_time[:, 0], target_positions_over_time[:, 1], 'k--', label="Target Position Path Over Time")
axs[1].scatter(0, 0, color='black', marker='o', label="Origin")
axs[1].set_xlabel("X Position")
axs[1].set_ylabel("Y Position")
axs[1].legend()

# 3. ESNの現在角度1の時間変化
axs[2].set_title("ESN Current Angle 1 over Time")
for idx, real_traj in enumerate(esn_real_trajectories):
    axs[2].plot(real_traj[:, 0], color=esn_colors[idx], alpha=0.7, linestyle=":", label=f"Current Angle 1 - Traj {idx+1}")
axs[2].set_xlabel("Time Step")
axs[2].set_ylabel("Current Angle 1")

# 4. ESNによるトルクの時間変化
axs[3].set_title("ESN Torque over Time")
for idx, torque in enumerate(esn_torques):
    axs[3].plot(torque[:, 0], color=esn_colors[idx], label=f"Joint 1 - Trajectory {idx+1}")
    axs[3].plot(torque[:, 1], linestyle="--", color=esn_colors[idx], label=f"Joint 2 - Trajectory {idx+1}")
axs[3].set_xlabel("Time Step")
axs[3].set_ylabel("Torque")

# 5. ティーチング・プレイバック法（関節角度）
axs[4].set_title("Teaching Trajectories (angles)")
for idx, traj in enumerate(teaching_trajectories):
    axs[4].plot(traj[:, 0], traj[:, 1], color=teaching_colors[idx], alpha=0.7)
    axs[4].scatter(traj[0, 0], traj[0, 1], color='green', marker='o')
axs[4].scatter(start_angle[0], start_angle[1], color='black', marker='o', label="Start Angle")
axs[4].scatter(end_angle[0], end_angle[1], color='black', marker='x', label="End Angle")
axs[4].plot(target_angles_over_time[:, 0], target_angles_over_time[:, 1], 'k--', label="Target Angle Path Over Time")
axs[4].set_xlabel("Joint Angles1")
axs[4].set_ylabel("Joint Angles2")
axs[4].legend()

# 6. ティーチング・プレイバック法（XY座標）
axs[5].set_title("Teaching Trajectories (XY)")
for idx, traj in enumerate(teaching_trajectories):
    xy_traj = np.array([forward_kinematics(angles) for angles in traj])
    axs[5].plot(xy_traj[:, 0], xy_traj[:, 1], color=teaching_colors[idx], alpha=0.7)
    axs[5].scatter(xy_traj[0, 0], xy_traj[0, 1], color='green', marker='o')
axs[5].scatter(start_position[0], start_position[1], color='black', marker='o', label="Start Position")
axs[5].scatter(end_position[0], end_position[1], color='black', marker='x', label="End Position")
axs[5].plot(target_positions_over_time[:, 0], target_positions_over_time[:, 1], 'k--', label="Target Position Path Over Time")
axs[5].scatter(0, 0, color='black', marker='o', label="Origin")
axs[5].set_xlabel("X Position")
axs[5].set_ylabel("Y Position")
axs[5].legend()

# 7. ティーチング時の現在角度1の時間変化
axs[6].set_title("Teaching Current Angle 1 over Time")
for idx, traj in enumerate(teaching_trajectories):
    axs[6].plot(traj[:, 0], color=teaching_colors[idx], label=f"Trajectory {idx+1}", alpha=0.7)
axs[6].set_xlabel("Time Step")
axs[6].set_ylabel("Current Angle 1")

# 8. ティーチングデータにおけるトルクの時間変化
axs[7].set_title("Teaching Torque over Time")
for idx, torque in enumerate(teaching_torques):
    axs[7].plot(torque[:, 0], color=teaching_colors[idx], label=f"Joint 1 - Trajectory {idx+1}")
    axs[7].plot(torque[:, 1], linestyle="--", color=teaching_colors[idx], label=f"Joint 2 - Trajectory {idx+1}")
axs[7].set_xlabel("Time Step")
axs[7].set_ylabel("Torque")

plt.tight_layout()
plt.show()