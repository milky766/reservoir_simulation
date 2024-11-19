import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from model import ESN, Tikhonov

# CSVファイルの絶対パス
data_path = '/home/nedo-nmd-02/ROBOTIS/DynamixelSDK/c++/example/protocol2.0/reserver/data/generated_data_by_simulation/step_angle_data.csv'

# CSVファイルを読み込む
data = pd.read_csv(data_path)
angles = data[['Angle1', 'Angle2']].values  # Input angles (rad)
target = np.array([1, 1])  # Target angle to converge to (1,1)

# Initialize ESN parameters with adjusted values
N_u = 2  # Input dimension
N_y = 2  # Output dimension
N_x = 300  # Reservoir nodes (increased to allow more dynamic behavior)
density = 0.2  # Adjusted reservoir density
input_scale = 1.0
rho = 1.1  # Adjusted spectral radius
leaking_rate = 0.8  # Adjusted leaky rate
beta = 1e-3  # Adjusted regularization parameter

# Instantiate the ESN and Tikhonov optimizer
esn = ESN(N_u=N_u, N_y=N_y, N_x=N_x, density=density, input_scale=input_scale, rho=rho, 
          leaking_rate=leaking_rate)  # Set a fixed seed for reproducibility
optimizer = Tikhonov(N_x=N_x, N_y=N_y, beta=beta)

# Train the ESN model on the provided data
esn.train(angles, angles, optimizer)

# Generate trajectories from multiple initial conditions towards target
initial_conditions = [
    [0, 0], [np.pi/5, np.pi/5], [2*np.pi/5, np.pi/5], [np.pi/5, 2*np.pi/5], [np.pi, 0], [0, np.pi]
]

# Plot the phase space trajectories with more steps and different colors for each trajectory
plt.figure(figsize=(10, 8))
colors = plt.cm.viridis(np.linspace(0, 1, len(initial_conditions)))  # Use a color map for distinct colors

for init, color in zip(initial_conditions, colors):
    # Simulate trajectory starting from initial condition with more steps
    trajectory = esn.run(np.array([init] * 300))  # Further increased steps

    # Plot trajectory in phase space (Angle1 vs Angle2) with distinct color
    plt.plot(trajectory[:, 0], trajectory[:, 1], label=f"Start: {init}", color=color)

# Plot target point
plt.scatter(*target, color="black", marker="o", s=100, label="Target (1,1)")

# Label the plot
plt.title("Phase Space Trajectories Converging to Target (1,1) [rad]")
plt.xlabel("Angle1 (rad)")
plt.ylabel("Angle2 (rad)")
plt.legend()
plt.grid(True)

# 画像として保存
plt.savefig('/home/nedo-nmd-02/ROBOTIS/DynamixelSDK/c++/example/protocol2.0/reserver/data/generated_data_by_simulation/trajectory_phase_space_v5.png')  # 画像を保存するパス
plt.show()
