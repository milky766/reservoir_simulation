import numpy as np
import matplotlib.pyplot as plt
from model_new import ESN, Tikhonov, generate_teaching_trajectories   # model.pyのESNとTikhonovクラスをインポート

# ESNおよびロボットモデルのパラメータ設定
#このパラメータ色々触ってもトルクは小さくならなかったから、PD制御のゲインかアルゴリズムに問題があるのかも
#もしくは、もうすでにトルクを小さくするというタスクは達成できているのかも。適切な比較対象がないからそれに気づいていないだけかも

N_x = 400               # リザバーのノード数 #1000でもあんまり変わらない
input_scale = 0.1       # 入力スケーリング　リザバーへの入力信号の強さ 0.01まで下げると目標にたどり着けなくなる　0.1だとかなり追従性いい.カナリ重要なパラメータ
density = 0.1           # 結合密度
rho = 0.95              # スペクトル半径　
leaking_rate = 0.99     # リーキング率
beta = 0.0001           # リッジ回帰の正則化係数　そんなに重要じゃなさそうなパラメータ
activation_function = np.tanh #活性化関数

# ロボットのパラメータ
link_lengths = [0.1, 0.1]  # 各リンクの長さ (0.1 m)
KP = np.diag([15.0, 15.0])  # PDコントローラの比例ゲイン
KD = np.diag([0.4, 0.4])    # PDコントローラの微分ゲイン
dt = 0.005                  # 積分時間ステップ (10ms) 0.005あたりがいいかも
steps = 1000               # シミュレーションステップ数

# 目標位置（関節角度をラジアンで指定）
target_angles = np.array([np.pi / 2, np.pi / 2])  # 目標位置 (45度)
center_angles = np.array([0.0, 0.0])             # 初期値分布の中心

# 順運動学に基づく目標先端位置
def forward_kinematics(angles):
    x = link_lengths[0] * np.cos(angles[0]) + link_lengths[1] * np.cos(angles[0] + angles[1])
    y = link_lengths[0] * np.sin(angles[0]) + link_lengths[1] * np.sin(angles[0] + angles[1])
    return np.array([x, y])

# 目標位置を先端の座標として設定
target_position = forward_kinematics(target_angles)

# ランダムな初期位置とトルクの生成（角度）
np.random.seed(0)
initial_positions_angles = [center_angles + np.random.uniform(0.0, np.pi, size=2) for _ in range(10)] #異なる初期角度のセット10個が格納

# ティーチング軌道の生成
teaching_trajectories, teaching_torques = generate_teaching_trajectories(initial_positions_angles, target_angles, steps, KP, KD, dt) #各初期位置から目標位置に到達するための軌道データが格納

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
esn.train_repeat(u, d, optimizer)

# 新しい初期点から目標位置への軌道生成（自律走行）とトルク計算
np.random.seed(2)
test_initial_positions_angles = [center_angles + np.random.uniform(0.0, np.pi, size=2) for _ in range(10)]
esn_trajectories = []
esn_torques = []

esn_real_trajectories = []
esn_trajectories_error = []

for pos in test_initial_positions_angles:
    esn_predicted_angle_seq , esn_torque_seq, esn_real_angle_seq, esn_error_seq = esn.trajectory5(pos, steps, KP, KD, dt, alpha = 0)
    
    esn_trajectories.append(np.array(esn_predicted_angle_seq))
    esn_torques.append(np.array(esn_torque_seq))
    esn_real_trajectories.append(np.array(esn_real_angle_seq))
    esn_trajectories_error.append(np.array(esn_error_seq))

for idx, traj in enumerate(esn_real_trajectories):
    final_position = traj[-1]  # 軌道の最終ステップの位置
    print(f"Trajectory {idx+1} final position: {final_position}")
    
# カラーマップを用意
cmap = plt.cm.rainbow
esn_colors = cmap(np.linspace(0, 1, len(esn_trajectories)))
teaching_colors = cmap(np.linspace(0, 1, len(teaching_trajectories)))


# 4つのプロットのセットアップ（3行×4列に変更）
fig, axs = plt.subplots(3, 4, figsize=(15, 15))  # プロット領域を増やす
axs = axs.flatten()  # axsを1次元配列に変換

# 1. ESNによる関節角度の動作生成
axs[0].set_title("ESN-based Adaptive Motion Generation (angles)")
for idx, traj in enumerate(esn_trajectories):
    axs[0].plot(traj[:, 0], traj[:, 1], color=esn_colors[idx], alpha=0.7)
    axs[0].scatter(traj[0, 0], traj[0, 1], color='green', marker='o')  # 初期位置
axs[0].scatter(target_angles[0], target_angles[1], color='red', label='Target angles')
axs[0].set_xlabel("Joint Angles1")
axs[0].set_ylabel("Joint Angles2")
axs[0].legend()

# 2. ESNによるXY座標の動作生成
axs[1].set_title("ESN-based Adaptive Motion Generation (XY)")
for idx, traj in enumerate(esn_trajectories):
    xy_traj = np.array([forward_kinematics(angles) for angles in traj])  # 角度からXYに変換
    axs[1].plot(xy_traj[:, 0], xy_traj[:, 1], color=esn_colors[idx], alpha=0.7)
    axs[1].scatter(xy_traj[0, 0], xy_traj[0, 1], color='green', marker='o')  # 初期位置
axs[1].scatter(0, 0, color='black', marker='o', label="Origin")
axs[1].scatter(target_position[0], target_position[1], color='red', label='Target Position')
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

axs[3].set_xlabel("Time Step")
axs[3].set_ylabel("Torque")

# 5. ティーチング・プレイバック法（関節角度）
axs[4].set_title("Teaching Trajectories (angles)")
for idx, traj in enumerate(teaching_trajectories):
    axs[4].plot(traj[:, 0], traj[:, 1], color=teaching_colors[idx], alpha=0.7)
    axs[4].scatter(traj[0, 0], traj[0, 1], color='green', marker='o')  # 初期位置
axs[4].scatter(target_angles[0], target_angles[1], color='red', label='Target angles')
axs[4].set_xlabel("Joint Angles1")
axs[4].set_ylabel("Joint Angles2")
axs[4].legend()

# 6. ティーチング・プレイバック法（XY座標）
axs[5].set_title("Teaching Trajectories (XY)")
for idx, traj in enumerate(teaching_trajectories):
    xy_traj = np.array([forward_kinematics(angles) for angles in traj])  # 角度からXYに変換
    axs[5].plot(xy_traj[:, 0], xy_traj[:, 1], color=teaching_colors[idx], alpha=0.7)
    axs[5].scatter(xy_traj[0, 0], xy_traj[0, 1], color='green', marker='o')
axs[5].scatter(0, 0, color='black', marker='o', label="Origin")
axs[5].scatter(target_position[0], target_position[1], color='red', label='Target Position')
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

axs[7].set_xlabel("Time Step")
axs[7].set_ylabel("Torque")

# 9. ESNによって生成された目標角の時間変化
axs[8].set_title("Generated Target Angle")
for idx, traj in enumerate(esn_trajectories):
    axs[8].plot(traj[:, 0], label=f"Angle 1 - Traj {idx+1}", color=esn_colors[idx], alpha=0.7)
    axs[8].plot(traj[:, 1], label=f"Angle 2 - Traj {idx+1}", linestyle='--', color=esn_colors[idx], alpha=0.7)
axs[8].set_xlabel("Time Step")
axs[8].set_ylabel("Generated Target Angles")
axs[8].legend()

# 10. 現在角と目標角の誤差の時間変化
axs[9].set_title("Current Angle and Target Angle error")
for idx, error in enumerate(esn_trajectories_error):
    axs[9].plot(error[:, 0], label=f"Error in Angle 1 - Traj {idx+1}", color=esn_colors[idx], alpha=0.7)
    axs[9].plot(error[:, 1], label=f"Error in Angle 2 - Traj {idx+1}", linestyle='--', color=esn_colors[idx], alpha=0.7)
axs[9].set_xlabel("Time Step")
axs[9].set_ylabel("Angle Error")
axs[9].legend()

plt.tight_layout()
plt.show()
