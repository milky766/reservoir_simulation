#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model import ESN, Tikhonov  # 上記のESNモジュールがmodel.pyに保存されていると仮定

# CSVデータを読み込み
data = pd.read_csv('step_angle_data.csv')  # 角度データのCSVファイル名に置き換えてください
print(data.columns)
# 正しいカラム名に変更
input_data = data[['Angle1', 'Angle2']].values  # 入力データ（ジョイント角度）
target_data = data[['Angle1', 'Angle2']].values  # 教師データ（目標角度が同じ場合に仮定）

# ESNモデルのパラメータ設定6+
N_u = input_data.shape[1]  # 入力次元（ジョイント角度の数）
N_y = target_data.shape[1]  # 出力次元（目標角度の数）
N_x = 1000  # リザーバのノード数（調整が必要）

# ESNモデルを初期化
esn = ESN(N_u=N_u, N_y=N_y, N_x=N_x, density=0.1, input_scale=1.0,
          rho=0.95, activation_func=np.tanh, leaking_rate=0.99)

# Tikhonov正則化（リッジ回帰）による学習
beta = 1e-4  # 正則化パラメータ（調整が必要）
optimizer = Tikhonov(N_x, N_y, beta)

# 過渡期間の設定（リザーバの初期状態が安定するまでのデータを除外）
trans_len = 50  # 過渡期間の長さ（調整が必要）

# ESNを使って学習
print("Training the ESN model...")
esn.train(input_data, target_data, optimizer, trans_len=trans_len)

# 学習済みモデルを使って予測
print("Predicting using the trained ESN model...")
predicted_output = esn.predict(input_data)

# 予測結果と教師データのプロット
plt.figure(figsize=(12, 6))

# Joint1の角度
plt.subplot(2, 1, 1)
plt.plot(target_data[:, 0], label='Target Angle1', color='blue')
plt.plot(predicted_output[:, 0], label='Predicted Angle1', color='orange')
plt.xlabel('Time step')
plt.ylabel('Angle1 (deg)')
plt.legend()
plt.title('Target vs Predicted Angles for Joint1')

# Joint2の角度
plt.subplot(2, 1, 2)
plt.plot(target_data[:, 1], label='Target Angle2', color='blue')
plt.plot(predicted_output[:, 1], label='Predicted Angle2', color='orange')
plt.xlabel('Time step')
plt.ylabel('Angle2 (deg)')
plt.legend()
plt.title('Target vs Predicted Angles for Joint2')

plt.tight_layout()
plt.show()
