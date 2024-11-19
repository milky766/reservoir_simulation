import numpy as np
import pandas as pd

# シミュレーションパラメータ
time_steps = 1000  # 時間ステップ数
dt = 0.01          # 各時間ステップの長さ（秒）

# ロボットアームパラメータ
link_lengths = [1.0, 1.0]  # 各リンクの長さ（単位は仮にメートル）

# 初期角度
initial_angles = [45, -45]  # 初期角度（仮に設定）

# 目標位置
target_position = np.array([0.0, 0.0])  # 原点が目標位置

# データ記録用リスト
data = {
    "Time (s)": [],
    "Joint1 Angle (deg)": [],
    "Joint2 Angle (deg)": [],
    "End Effector X": [],
    "End Effector Y": [],
    "Target X": [],
    "Target Y": [],
}

# 逆運動学関数
def inverse_kinematics(target, link_lengths):
    x, y = target
    l1, l2 = link_lengths
    
    # 距離の計算
    d = np.sqrt(x**2 + y**2)
    if d > (l1 + l2):
        raise ValueError("Target is out of reach")

    # コサインの法則を使用して角度を計算
    theta2 = np.arccos((d**2 - l1**2 - l2**2) / (2 * l1 * l2))
    theta1 = np.arctan2(y, x) - np.arctan2(l2 * np.sin(theta2), l1 + l2 * np.cos(theta2))
    
    return np.degrees(theta1), np.degrees(theta2)

# 初期角度から目標位置に向かうまでの角度変化をシミュレーション
for t in range(time_steps):
    # 時刻を計算
    time = t * dt
    
    # 初期角度から目標角度に向かって線形補間
    try:
        target_angle1, target_angle2 = inverse_kinematics(target_position, link_lengths)
    except ValueError:
        print("Target position is out of reach.")
        break

    joint1_angle = initial_angles[0] * (1 - t / time_steps) + target_angle1 * (t / time_steps)
    joint2_angle = initial_angles[1] * (1 - t / time_steps) + target_angle2 * (t / time_steps)

    # 手先の位置を計算
    x = link_lengths[0] * np.cos(np.radians(joint1_angle)) + link_lengths[1] * np.cos(np.radians(joint1_angle + joint2_angle))
    y = link_lengths[0] * np.sin(np.radians(joint1_angle)) + link_lengths[1] * np.sin(np.radians(joint1_angle + joint2_angle))

    # データを記録
    data["Time (s)"].append(time)
    data["Joint1 Angle (deg)"].append(joint1_angle)
    data["Joint2 Angle (deg)"].append(joint2_angle)
    data["End Effector X"].append(x)
    data["End Effector Y"].append(y)
    data["Target X"].append(target_position[0])
    data["Target Y"].append(target_position[1])

# データをCSVファイルに保存
df = pd.DataFrame(data)
df.to_csv("output.csv", index=False)
print("Data saved to output.csv")
