import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# データ読み込み
# data = pd.read_csv("output.csv")
data = pd.read_csv("step_angle_data.csv")
# グラフ作成
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

# ジョイント角度
axs[0, 0].plot(data["Time (s)"], np.degrees(data["Joint1 Angle (rad)"]), label="Joint1 Angle")
axs[0, 0].plot(data["Time (s)"], np.degrees(data["Joint2 Angle (rad)"]), label="Joint2 Angle")
axs[0, 0].set_xlabel("Time (s)")
axs[0, 0].set_ylabel("Angle (deg)")
axs[0, 0].legend()
axs[0, 0].set_title("Joint Angles")

# 手先のX-Y軌跡
axs[0, 1].plot(data["End Effector X"], data["End Effector Y"], label="End Effector Path")
axs[0, 1].set_xlabel("X Position (m)")
axs[0, 1].set_ylabel("Y Position (m)")
axs[0, 1].legend()
axs[0, 1].set_title("End Effector Trajectory")

# トルク（シミュレーションでは固定値にするなど）
axs[1, 0].plot(data["Time (s)"], [0.5] * len(data["Time (s)"]), label="Torque1")
axs[1, 0].plot(data["Time (s)"], [0.5] * len(data["Time (s)"]), label="Torque2")
axs[1, 0].set_xlabel("Time (s)")
axs[1, 0].set_ylabel("Torque (N*m)")
axs[1, 0].legend()
axs[1, 0].set_title("Torque")

# 電流（シミュレーションでは固定値にするなど）
axs[1, 1].plot(data["Time (s)"], [300] * len(data["Time (s)"]), label="Current1")
axs[1, 1].plot(data["Time (s)"], [300] * len(data["Time (s)"]), label="Current2")
axs[1, 1].set_xlabel("Time (s)")
axs[1, 1].set_ylabel("Current (mA)")
axs[1, 1].legend()
axs[1, 1].set_title("Current")

plt.tight_layout()
plt.show()
