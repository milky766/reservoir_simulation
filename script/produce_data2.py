import numpy as np
import pandas as pd

# シミュレーションパラメータ
start_time = -0.25   # 開始時刻（秒）
end_time = 1.0       # 終了時刻（秒）
dt = 0.001           # タイムステップ（秒）
step_time = 0.0      # ステップが変わる時刻（秒）

# ロボットアームパラメータ
link_lengths = [1.8, 1.8]  # 各リンクの長さ（単位はメートル）

# データ記録用リスト
data = {
    "Time (s)": [],
    "Joint1 Angle (rad)": [],
    "Joint2 Angle (rad)": [],
    "End Effector X": [],
    "End Effector Y": []
}

# ステップ角度
initial_angles = [0.0, 0.0]  # [rad] 初期角度
target_angles = [1.0, 1.0]   # [rad] ステップ後の角度

# 時間範囲を設定し、シミュレーションループを実行
time_steps = int((end_time - start_time) / dt)
for t in range(time_steps):
    # 現在の時刻を計算
    time = start_time + t * dt
    
    # 角度を設定（ステップ関数）
    if time < step_time:
        joint1_angle = initial_angles[0]
        joint2_angle = initial_angles[1]
    else:
        joint1_angle = target_angles[0]
        joint2_angle = target_angles[1]

    # 手先の位置を計算
    x = link_lengths[0] * np.cos(joint1_angle) + link_lengths[1] * np.cos(joint1_angle + joint2_angle)
    y = link_lengths[0] * np.sin(joint1_angle) + link_lengths[1] * np.sin(joint1_angle + joint2_angle)

    # データを記録
    data["Time (s)"].append(time)
    data["Joint1 Angle (rad)"].append(joint1_angle)
    data["Joint2 Angle (rad)"].append(joint2_angle)
    data["End Effector X"].append(x)
    data["End Effector Y"].append(y)

# データをCSVファイルに保存
df = pd.DataFrame(data)
df.to_csv("step_angle_data.csv", index=False)
print("Data saved to step_angle_data.csv")
