import numpy as np
import matplotlib.pyplot as plt
from model_new import ESN, Tikhonov, generate_teaching_trajectories   # model.pyのESNとTikhonovクラスをインポート

# ESNおよびロボットモデルのパラメータ設定
#このパラメータ色々触ってもトルクは小さくならなかったから、PD制御のゲインかアルゴリズムに問題があるのかも
#もしくは、もうすでにトルクを小さくするというタスクは達成できているのかも。適切な比較対象がないからそれに気づいていないだけかも
N_x = 1000 # リザバーのノード数 #1000でもあんまり変わらない
input_scale = 0.1    # 入力スケーリング　リザバーへの入力信号の強さ 0.01まで下げると目標にたどり着けなくなる　0.1だとかなり追従性いい.カナリ重要なパラメータ
density = 0.1        # 結合密度
rho = 0.95           # スペクトル半径　
leaking_rate = 0.99     # リーキング率
beta = 0.0001           # リッジ回帰の正則化係数　そんなに重要じゃなさそうなパラメータ
activation_function = np.tanh #活性化関数

# ロボットのパラメータ
link_lengths = [0.1, 0.1]  # 各リンクの長さ (0.1 m)
KP = np.diag([15.0, 15.0])  # PDコントローラの比例ゲイン
KD = np.diag([0.4, 0.4])    # PDコントローラの微分ゲイン
dt = 0.01                   # 積分時間ステップ (10ms) 0.005あたりがいいかも
steps = 1000               # シミュレーションステップ数



# ESNの初期化
N_u = 2  # 入力次元
N_y = 2  # 出力次元
esn = ESN(N_u, N_y, N_x, density=density, input_scale=input_scale, rho=rho,
          activation_func=activation_function, leaking_rate=leaking_rate)

W = esn.Reservoir.get_weights()


print(W)  # 行列の内容

print(np.max(np.abs(np.linalg.eig(W)[0])))  # スペクトル半径の確認
print(np.mean(W), np.std(W))  # 平均と標準偏差の確認
