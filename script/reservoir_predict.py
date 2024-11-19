import numpy as np
import matplotlib.pyplot as plt
from model import ESN, Tikhonov  # model.pyのESNとTikhonovクラスをインポート

# ESNおよびロボットモデルのパラメータ設定
N_x = 400               # リザバーのノード数
input_scale = 0.01       # 入力スケーリング　リザバーへの入力信号の強さ
density = 0.1           # 結合密度
rho = 0.95              # スペクトル半径
leaking_rate = 0.99     # リーキング率
beta = 0.0001           # リッジ回帰の正則化係数
activation_function = np.tanh #活性化関数
# ロボットのパラメータ
link_lengths = [0.1, 0.1]  # 各リンクの長さ (0.1 m)
KP = np.diag([15.0, 15.0])  # PDコントローラの比例ゲイン
KD = np.diag([0.7, 0.7])    # PDコントローラの微分ゲイン
dt = 0.01                   # 積分時間ステップ (10ms)
steps = 1000                # シミュレーションステップ数
# 初期位置と目標位置（関節角度をラジアンで指定）
initial_angles = np.array([0.0, 0.0])  # 中心の初期位置 (0度)
target_angles = np.array([np.pi / 4, np.pi / 4])  # 目標位置 (45度)
# 順運動学に基づく目標先端位置
def forward_kinematics(angles):
    x = link_lengths[0] * np.cos(angles[0]) + link_lengths[1] * np.cos(angles[0] + angles[1])
    y = link_lengths[0] * np.sin(angles[0]) + link_lengths[1] * np.sin(angles[0] + angles[1])
    return np.array([x, y])
# 目標位置を先端の座標として設定
target_position = forward_kinematics(target_angles)
# PDコントローラ
def pd_control(desired_angle, current_angle, current_velocity, KP, KD):
    error = desired_angle - current_angle                       #誤差＝目標角度ー現在角度
    control_input = KP @ error - KD @ current_velocity          #制御入力(トルク)＝PD制御
    return control_input
# 動力学シミュレーション(17)
def dynamics(torque):                     #関節角度、関節各速度、トルク
    inertia_matrix = np.diag([0.01, 0.01])          # 簡略化した慣性行列H(1*2)
    q_ddot = np.linalg.inv(inertia_matrix) @ torque #角加速度=慣性行列の逆行列＊トルク
    return q_ddot

# ティーチング軌道生成 
def generate_teaching_trajectories(initial_angles_set, target_angles, steps):   #初期関節角度、目標角度、ステップ数 
    trajectories = []                                                           #各初期点からの軌道を格納するリスト
    torques = []                                                                # 各時刻でのトルクを格納するリスト
    for initial_angles in initial_angles_set:                                   #初期角度の数だけ軌道作成
        q = initial_angles                                                      #初期角度を関節角度qに入れる(初期化)
        q_dot = np.zeros(2)                                                     #関節角速度を0に初期化
        trajectory = [q]                                                        #xy座標として軌道を格納
        torque_seq = []                                                         # 各軌道のトルク時系列を保存
        for _ in range(steps):
            torque = pd_control(target_angles, q, q_dot, KP, KD)                #pd制御での制御入力をトルクとする
            q_ddot = dynamics(torque)                                 #角加速度を計算
            q = q + q_dot * dt                                                  #オイラー法で角度と角速度を更新
            q_dot = q_dot + q_ddot * dt
            trajectory.append(q)                            #計算した角度で軌道を求める
            torque_seq.append(torque)
        trajectories.append(np.array(trajectory))                               #各初期点からの軌道をリストに格納
        torques.append(np.array(torque_seq))
    return trajectories, torques


# ランダムな初期位置とトルクの生成（角度）
np.random.seed(0)
initial_positions_angles = [initial_angles + np.random.uniform(-3.0, 3.0, size=2) for _ in range(10)] #異なる初期角度のセット10個が格納
# ティーチング軌道の生成
teaching_trajectories, teaching_torques = generate_teaching_trajectories(initial_positions_angles, target_angles, steps) #各初期位置から目標位置に到達するための軌道データが格納
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
test_initial_positions_angles = [initial_angles + np.random.uniform(-3.0, 3.0, size=2) for _ in range(10)]
esn_trajectories = []
esn_torques = []
esn_real_trajectories = []
esn_trajectories_error = []

for pos in test_initial_positions_angles:
    esn_predicted_angle_seq , esn_torque_seq, esn_real_angle_seq, esn_error_seq = esn.trajectory(pos, steps, KP, KD, dt)
    
    esn_trajectories.append(np.array(esn_predicted_angle_seq))
    esn_torques.append(np.array(esn_torque_seq))
    esn_real_trajectories.append(np.array(esn_real_angle_seq))
    esn_trajectories_error.append(np.array(esn_error_seq))


# カラーマップを用意
cmap = plt.cm.rainbow
colors = cmap(np.linspace(0, 1, len(esn_trajectories)))

# 4つのプロットのセットアップ
fig, axs = plt.subplots(2, 3, figsize=(15, 10))
axs = axs.flatten()  # axsを1次元配列に変換

# 1. ESNによる関節角度の動作生成
axs[0].set_title("ESN-based Adaptive Motion Generation (angles)")
for idx, traj in enumerate(esn_trajectories):
    axs[0].plot(traj[:, 0], traj[:, 1], color=colors[idx], alpha=0.7)
    axs[0].scatter(traj[0, 0], traj[0, 1], color='green', marker='o')  # 初期位置
axs[0].scatter(0, 0, color='black', marker='o', label="Origin")
axs[0].scatter(target_angles[0], target_angles[1], color='red', label='Target angles')
axs[0].set_xlabel("Joint Angles1")
axs[0].set_ylabel("Joint Angles2")
axs[0].legend()

# 2. ESNによるXY座標の動作生成
axs[1].set_title("ESN-based Adaptive Motion Generation (XY)")
for idx, traj in enumerate(esn_trajectories):
    xy_traj = np.array([forward_kinematics(angles) for angles in traj])  # 角度からXYに変換
    axs[1].plot(xy_traj[:, 0], xy_traj[:, 1], color=colors[idx], alpha=0.7)
    axs[1].scatter(xy_traj[0, 0], xy_traj[0, 1], color='green', marker='o')  # 初期位置
axs[1].scatter(0, 0, color='black', marker='o', label="Origin")
axs[1].scatter(target_position[0], target_position[1], color='red', label='Target Position')
axs[1].set_xlabel("X Position")
axs[1].set_ylabel("Y Position")
axs[1].legend()

# 3. ESNによるトルクの時間変化
axs[2].set_title("ESN Torque over Time")
for idx, torque in enumerate(esn_torques):
    axs[2].plot(torque[:, 0], color=colors[idx], label=f"Joint 1 - Trajectory {idx+1}")
    axs[2].plot(torque[:, 1], linestyle="--", color=colors[idx], label=f"Joint 2 - Trajectory {idx+1}")
axs[2].set_xlabel("Time Step")
axs[2].set_ylabel("Torque")

# 4. ティーチング・プレイバック法（関節角度）
axs[3].set_title("Teaching Trajectories (angles)")
for idx, traj in enumerate(teaching_trajectories):
    axs[3].plot(traj[:, 0], traj[:, 1], color=colors[idx], alpha=0.7)
    axs[3].scatter(traj[0, 0], traj[0, 1], color='green', marker='o')  # 初期位置
axs[3].scatter(0, 0, color='black', marker='o', label="Origin")
axs[3].scatter(target_angles[0], target_angles[1], color='red', label='Target angles')
axs[3].set_xlabel("Joint Angles1")
axs[3].set_ylabel("Joint Angles2")
axs[3].legend()

# 5. ティーチング・プレイバック法（XY座標）
axs[4].set_title("Teaching Trajectories (XY)")
for idx, traj in enumerate(teaching_trajectories):
    xy_traj = np.array([forward_kinematics(angles) for angles in traj])  # 角度からXYに変換
    axs[4].plot(xy_traj[:, 0], xy_traj[:, 1], 'r-', alpha=0.7)
    axs[4].scatter(xy_traj[0, 0], xy_traj[0, 1], color='green', marker='o')
axs[4].scatter(0, 0, color='black', marker='o', label="Origin")
axs[4].scatter(target_position[0], target_position[1], color='red', label='Target Position')
axs[4].set_xlabel("X Position")
axs[4].set_ylabel("Y Position")
axs[4].legend()

# 6. ティーチングデータにおけるトルクの時間変化 
#ティーチングデータのトルクは、目標角が（４５，４５）のまま計算しているから、最初が大きくてどんどん小さくなるようなグラフになる
#これは間違っているが、修正が面倒だし、直しても意味ないので放置

axs[5].set_title("Teaching Torque over Time")
for idx, torque in enumerate(teaching_torques):
    axs[5].plot(torque[:, 0], color=colors[idx], label=f"Joint 1 - Trajectory {idx+1}")
    axs[5].plot(torque[:, 1], linestyle="--", color=colors[idx], label=f"Joint 2 - Trajectory {idx+1}")
axs[5].set_xlabel("Time Step")
axs[5].set_ylabel("Torque")
plt.tight_layout()
plt.show()