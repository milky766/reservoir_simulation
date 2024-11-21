import numpy as np
import networkx as nx

# 恒等写像
def identity(x):
    return x

# 入力層
class Input:
    # 入力結合重み行列Winの初期化
    def __init__(self, N_u, N_x, input_scale, seed=0):
        '''
        param N_u: 入力次元
        param N_x: リザバーのノード数
        param input_scale: 入力スケーリング
        '''
        # 一様分布に従う乱数
        np.random.seed(seed=seed)
        # self.Win = np.random.rand(N_x, N_u)
        self.Win = np.random.uniform(-input_scale, input_scale, (N_x, N_u))

    # 入力結合重み行列Winによる重みづけ
    def __call__(self, u):
        '''
        param u: N_u次元のベクトル
        return: N_x次元のベクトル
        '''
        return np.dot(self.Win, u)
    
class Input_fire:
    # 入力結合重み行列Winの初期化
    def __init__(self, N_u, N_x, gIn, seed=0):
        '''
        param N_u: 入力次元
        param N_x: リザバーのノード数
        param input_scale: 入力スケーリング
        '''
        # 一様分布に従う乱数
        np.random.seed(seed=seed)
        # self.Win = np.random.rand(N_x, N_u)
        self.Win = np.random.normal(0, np.sqrt(gIn), (N_x, N_u))  # 分散gInの正規分布
        
    # 入力結合重み行列Winによる重みづけ
    def __call__(self, u):
        '''
        param u: N_u次元のベクトル
        return: N_x次元のベクトル
        '''
        return np.dot(self.Win, u)


# リザバー
class Reservoir:
    # リカレント結合重み行列Wの初期化
    def __init__(self, N_x, density, rho, activation_func, leaking_rate,
                 seed=0):
        '''
        param N_x: リザバーのノード数
        param density: ネットワークの結合密度
        param rho: リカレント結合重み行列のスペクトル半径
        param activation_func: ノードの活性化関数
        param leaking_rate: leaky integratorモデルのリーク率
        param seed: 乱数の種
        '''
        self.seed = seed
        # self.W = self.make_connection_2(N_x, density, rho, 1.5) #一様分布から正規分布に変更
        self.W = self.make_connection(N_x, density, rho) #一様分布から正規分布に変更
        self.x = np.zeros(N_x)  # リザバー状態ベクトルの初期化
        self.activation_func = activation_func
        self.alpha = leaking_rate

    # リカレント結合重み行列の生成
    def make_connection(self, N_x, density, rho):
        # Erdos-Renyiランダムグラフ
        m = int(N_x*(N_x-1)*density/2)  # 総結合数
        G = nx.gnm_random_graph(N_x, m, self.seed)

        # 行列への変換(結合構造のみ）
        connection = nx.to_numpy_array(G)
        W = np.array(connection)

        # 非ゼロ要素を一様分布に従う乱数として生成
        rec_scale = 1.0
        np.random.seed(seed=self.seed)
        W *= np.random.uniform(-rec_scale, rec_scale, (N_x, N_x))

        # スペクトル半径の計算
        eigv_list = np.linalg.eig(W)[0]
        sp_radius = np.max(np.abs(eigv_list))

        # 指定のスペクトル半径rhoに合わせてスケーリング
        W *= rho / sp_radius

        return W
    
    # # リカレント結合重み行列の生成
    # def make_connection_2(self, N_x, density, rho, g_Rec): #一様分布から正規分布に変更
    #     # Erdos-Renyiランダムグラフ
    #     m = int(N_x*(N_x-1)*density/2)  # 総結合数
    #     G = nx.gnm_random_graph(N_x, m, self.seed)

    #     # 行列への変換(結合構造のみ）
    #     connection = nx.to_numpy_array(G)
    #     W = np.array(connection)

    #     # 非ゼロ要素を正規分布に従う乱数として生成
    #     rec_scale = g_Rec/(np.sqrt(density*N_x))
    #     standard_deviation = np.sqrt(rec_scale)
    #     np.random.seed(seed=self.seed)
    #     W *= np.random.normal(loc=0, scale=standard_deviation, size=(N_x, N_x))
        
    #     # スペクトル半径の計算
    #     eigv_list = np.linalg.eig(W)[0]
    #     sp_radius = np.max(np.abs(eigv_list))

    #     # 指定のスペクトル半径rhoに合わせてスケーリング
    #     W *= rho / sp_radius

    #     return W
    def make_connection_2(self, N_x, density, rho, g_Rec): 
        # 結合確率 density に基づいてランダムなマスク行列を生成
        np.random.seed(self.seed)  # 再現性のためのシード設定
        W_mask = np.random.rand(N_x, N_x)  # 一様分布 [0, 1)
        W_mask[W_mask <= density] = 1  # density 以下の値を 1 に
        W_mask[W_mask < 1] = 0  # 残りの値を 0 に

        # 正規分布に従う重み行列を生成
        scale = g_Rec / np.sqrt(density * N_x)  # スケール係数
        W = np.random.randn(N_x, N_x) * scale  # 標準正規分布からスケーリング
        
        # マスクを適用（要素ごとの掛け算）
        W = W * W_mask
        np.fill_diagonal(W, 0)  # 対角成分を 0 に

        # スペクトル半径を計算
        # eigv_list = np.linalg.eigvals(W)  # 固有値
        # sp_radius = np.max(np.abs(eigv_list))  # 最大固有値の絶対値（スペクトル半径）

        # # 指定のスペクトル半径 rho に合わせてスケーリング
        # W *= rho / sp_radius

        return W

    def get_weights(self):
        return self.W

    # リザバー状態ベクトルの更新
    def __call__(self, x_in):
        '''
        param x_in: 更新前の状態ベクトル
        return: 更新後の状態ベクトル
        '''
        #self.x = self.x.reshape(-1, 1)
        self.x = (1.0 - self.alpha) * self.x \
                 + self.alpha * self.activation_func(np.dot(self.W, self.x) \
                 + x_in)
        return self.x

    # リザバー状態ベクトルの初期化
    def reset_reservoir_state(self):
        self.x *= 0.0

class Reservoir_fire:
    # リカレント結合重み行列Wの初期化
    def __init__(self, N_x, density, rho, activation_func, tau, dt = 0.01
                 seed=0):
        '''
        param N_x: リザバーのノード数
        param density: ネットワークの結合密度
        param rho: リカレント結合重み行列のスペクトル半径
        param activation_func: ノードの活性化関数
        param leaking_rate: leaky integratorモデルのリーク率
        param seed: 乱数の種
        '''
        self.seed = seed
        self.tau = tau
        # self.W = self.make_connection_2(N_x, density, rho, 1.5) #一様分布から正規分布に変更
        self.W = self.make_connection(N_x, density, rho) #一様分布から正規分布に変更
        self.x = np.zeros(N_x)  # リザバー状態ベクトルの初期化
        self.activation_func = activation_func
        self.dt = dt


    # リカレント結合重み行列の生成
    def make_connection(self, N_x, density, rho):
        # Erdos-Renyiランダムグラフ
        m = int(N_x*(N_x-1)*density/2)  # 総結合数
        G = nx.gnm_random_graph(N_x, m, self.seed)

        # 行列への変換(結合構造のみ）
        connection = nx.to_numpy_array(G)
        W = np.array(connection)

        # 非ゼロ要素を一様分布に従う乱数として生成
        rec_scale = 1.0
        np.random.seed(seed=self.seed)
        W *= np.random.uniform(-rec_scale, rec_scale, (N_x, N_x))

        # スペクトル半径の計算
        eigv_list = np.linalg.eig(W)[0]
        sp_radius = np.max(np.abs(eigv_list))

        # 指定のスペクトル半径rhoに合わせてスケーリング
        W *= rho / sp_radius

        return W
    
    # # リカレント結合重み行列の生成
    # def make_connection_2(self, N_x, density, rho, g_Rec): #一様分布から正規分布に変更
    #     # Erdos-Renyiランダムグラフ
    #     m = int(N_x*(N_x-1)*density/2)  # 総結合数
    #     G = nx.gnm_random_graph(N_x, m, self.seed)

    #     # 行列への変換(結合構造のみ）
    #     connection = nx.to_numpy_array(G)
    #     W = np.array(connection)

    #     # 非ゼロ要素を正規分布に従う乱数として生成
    #     rec_scale = g_Rec/(np.sqrt(density*N_x))
    #     standard_deviation = np.sqrt(rec_scale)
    #     np.random.seed(seed=self.seed)
    #     W *= np.random.normal(loc=0, scale=standard_deviation, size=(N_x, N_x))
        
    #     # スペクトル半径の計算
    #     eigv_list = np.linalg.eig(W)[0]
    #     sp_radius = np.max(np.abs(eigv_list))

    #     # 指定のスペクトル半径rhoに合わせてスケーリング
    #     W *= rho / sp_radius

    #     return W
    def make_connection_2(self, N_x, density, rho, g_Rec): 
        # 結合確率 density に基づいてランダムなマスク行列を生成
        np.random.seed(self.seed)  # 再現性のためのシード設定
        W_mask = np.random.rand(N_x, N_x)  # 一様分布 [0, 1)
        W_mask[W_mask <= density] = 1  # density 以下の値を 1 に
        W_mask[W_mask < 1] = 0  # 残りの値を 0 に

        # 正規分布に従う重み行列を生成
        scale = g_Rec / np.sqrt(density * N_x)  # スケール係数
        W = np.random.randn(N_x, N_x) * scale  # 標準正規分布からスケーリング #sqrt(scale)ではない？
        
        # マスクを適用（要素ごとの掛け算）
        W = W * W_mask
        np.fill_diagonal(W, 0)  # 対角成分を 0 に

        # スペクトル半径を計算
        # eigv_list = np.linalg.eigvals(W)  # 固有値
        # sp_radius = np.max(np.abs(eigv_list))  # 最大固有値の絶対値（スペクトル半径）

        # # 指定のスペクトル半径 rho に合わせてスケーリング
        # W *= rho / sp_radius

        return W

    def get_weights(self):
        return self.W

    # リザバー状態ベクトルの更新
    def __call__(self, x_in):
        '''
        param x_in: 更新前の状態ベクトル
        return: 更新後の状態ベクトル
        '''
        #self.x = self.x.reshape(-1, 1)
        x_current = np.dot(self.W, self.x) + x_in
        self.x = self.x +((-self.x +x_current)/self.tau)*self.dt 
            
        return self.x

    # リザバー状態ベクトルの初期化
    def reset_reservoir_state(self):
        self.x *= 0.0

# 出力層
class Output:
    # 出力結合重み行列の初期化
    def __init__(self, N_x, N_y, seed=0):
        '''
        param N_x: リザバーのノード数
        param N_y: 出力次元
        param seed: 乱数の種
        '''
        # 正規分布に従う乱数
        np.random.seed(seed=seed)
        #self.Wout = np.random.normal(size=(N_y, N_x))
        self.Wout = np.zeros((N_y, N_x))


    # 出力結合重み行列による重みづけ
    def __call__(self, x):
        '''
        param x: N_x次元のベクトル
        return: N_y次元のベクトル
        '''
        #return np.dot(self.Wout, x)
        return np.tanh(np.dot(self.Wout, x))

    # 学習済みの出力結合重み行列を設定
    def setweight(self, Wout_opt):
        self.Wout = Wout_opt
        
class Output_fire:
    # 出力結合重み行列の初期化
    def __init__(self, N_x, N_y, seed=0):
        '''
        param N_x: リザバーのノード数
        param N_y: 出力次元
        param seed: 乱数の種
        '''
        # 正規分布に従う乱数
        np.random.seed(seed=seed)
        #self.Wout = np.random.normal(size=(N_y, N_x))
        self.Wout = np.zeros((N_y, N_x))


    # 出力結合重み行列による重みづけ
    def __call__(self, x):
        '''
        param x: N_x次元のベクトル
        return: N_y次元のベクトル
        '''
        #return np.dot(self.Wout, x)
        return np.tanh(np.dot(self.Wout, x))

    # 学習済みの出力結合重み行列を設定
    def setweight(self, Wout_opt):
        self.Wout = Wout_opt


# 出力フィードバック
class Feedback:
    # フィードバック結合重み行列の初期化
    def __init__(self, N_y, N_x, fb_scale, seed=0):
        '''
        param N_y: 出力次元
        param N_x: リザバーのノード数
        param fb_scale: フィードバックスケーリング
        param seed: 乱数の種
        '''
        # 一様分布に従う乱数
        np.random.seed(seed=seed)
        self.Wfb = np.random.uniform(-fb_scale, fb_scale, (N_x, N_y))

    # フィードバック結合重み行列による重みづけ
    def __call__(self, y):
        '''
        param y: N_y次元のベクトル
        return: N_x次元のベクトル
        '''
        return np.dot(self.Wfb, y)


# Moore-Penrose擬似逆行列
class Pseudoinv:
    def __init__(self, N_x, N_y):
        '''
        param N_x: リザバーのノード数
        param N_y: 出力次元
        '''
        self.X = np.empty((N_x, 0))
        self.D = np.empty((N_y, 0))
        
    # 状態集積行列および教師集積行列の更新
    def __call__(self, d, x):
        x = np.reshape(x, (-1, 1))
        d = np.reshape(d, (-1, 1))
        self.X = np.hstack((self.X, x))
        self.D = np.hstack((self.D, d))
        
    # Woutの最適解（近似解）の導出
    def get_Wout_opt(self):
        Wout_opt = np.dot(self.D, np.linalg.pinv(self.X))
        return Wout_opt


# リッジ回帰（beta=0のときは線形回帰）
class Tikhonov:
    def __init__(self, N_x, N_y, beta):
        '''
        param N_x: リザバーのノード数
        param N_y: 出力次元
        param beta: 正則化パラメータ
        '''
        self.beta = beta
        self.X_XT = np.zeros((N_x, N_x))
        self.D_XT = np.zeros((N_y, N_x))
        self.N_x = N_x

    # 学習用の行列の更新
    def __call__(self, d, x):
        x = np.reshape(x, (-1, 1))
        d = np.reshape(d, (-1, 1))
        self.X_XT += np.dot(x, x.T)
        self.D_XT += np.dot(d, x.T)

    # Woutの最適解（近似解）の導出
    def get_Wout_opt(self):
        X_pseudo_inv = np.linalg.inv(self.X_XT \
                                     + self.beta*np.identity(self.N_x))
        Wout_opt = np.dot(self.D_XT, X_pseudo_inv)
        return Wout_opt


# 逐次最小二乗（RLS）法
class RLS:
    def __init__(self, N_x, N_y, delta, lam, update):
        '''
        param N_x: リザバーのノード数
        param N_y: 出力次元
        param delta: 行列Pの初期条件の係数（P=delta*I, 0<delta<<1）
        param lam: 忘却係数 (0<lam<1, 1に近い値)
        param update: 各時刻での更新繰り返し回数
        '''
        self.delta = delta
        self.lam = lam
        self.update = update
        self.P = (1.0/self.delta)*np.eye(N_x, N_x) 
        self.Wout = np.zeros([N_y, N_x])
        
    # Woutの更新
    def __call__(self, d, x):
        x = np.reshape(x, (-1, 1))
        for i in np.arange(self.update):
            v = d - np.dot(self.Wout, x)
            gain = (1/self.lam*np.dot(self.P, x))
            gain = gain/(1+1/self.lam*np.dot(np.dot(x.T, self.P), x))
            self.P = 1/self.lam*(self.P-np.dot(np.dot(gain, x.T), self.P))
            self.Wout += np.dot(v, gain.T)

        return self.Wout

# PDコントローラ
def pd_control(desired_angle, current_angle, current_velocity, KP, KD):
        error = desired_angle - current_angle                       #誤差＝目標角度ー現在角度
        control_input = KP @ error - KD @ current_velocity          #制御入力(トルク)＝PD制御
        return control_input

# 動力学シミュレーション(17)
def dynamics(q, q_dot, torque):                     #関節角度、関節各速度、トルク
        inertia_matrix = np.diag([0.01, 0.01])          # 簡略化した慣性行列H(1*2)
        q_ddot = np.linalg.inv(inertia_matrix) @ torque #角加速度=慣性行列の逆行列＊トルク
        return q_ddot

# ティーチング軌道生成
def generate_teaching_trajectories(initial_angles_set, target_angles, steps, KP, KD, dt):   #初期関節角度、目標角度、ステップ数
        trajectories = []                                                           #各初期点からの軌道を格納するリスト
        torques = []                                                                # 各時刻でのトルクを格納するリスト
        for initial_angles in initial_angles_set:                                   #初期角度の数だけ軌道作成
            q = initial_angles                                                      #初期角度を関節角度qに入れる(初期化)
            q_dot = np.zeros(2)                                                     #関節角速度を0に初期化
            trajectory = [q]                                                        #xy座標として軌道を格納
            torque_seq = []                                                         # 各軌道のトルク時系列を保存
            for _ in range(steps):
                torque = pd_control(target_angles, q, q_dot, KP, KD)                #pd制御での制御入力をトルクとする
                q_ddot = dynamics(q, q_dot, torque)                                 #角加速度を計算                                              #オイラー法で角度と角速度を更新
                q = q + q_dot * dt
                q_dot = q_dot + q_ddot * dt    
                trajectory.append(q)                            #計算した角度で軌道を求める
                torque_seq.append(torque)
            trajectories.append(np.array(trajectory))                               #各初期点からの軌道をリストに格納
            torques.append(np.array(torque_seq))
        return trajectories, torques

# ティーチング軌道生成
def generate_teaching_trajectories_move(initial_angles_set, start_angle, end_angle, steps, KP, KD, dt):   
        """
        初期関節角度、目標角度の開始と終了、ステップ数に基づいてティーチング軌道を生成する。
        """
        trajectories = []  # 各初期点からの軌道を格納するリスト
        torques = []       # 各時刻でのトルクを格納するリスト
        transition_steps = int(2 / dt)  # 1秒間かけて目標角度を変化させる

        for initial_angles in initial_angles_set:  # 初期角度の数だけ軌道作成
            q = initial_angles  # 初期角度を関節角度qに入れる(初期化)
            q_dot = np.zeros(2)  # 関節角速度を0に初期化
            trajectory = [q]  # 軌道を格納
            torque_seq = []   # 各軌道のトルク時系列を保存

            for step in range(steps):
                # 時間に応じて目標角度を更新
                target_angle = dynamic_target_angle_sin(step, transition_steps, start_angle, end_angle)
                
                # pd制御での制御入力をトルクとする
                torque = pd_control(target_angle, q, q_dot, KP, KD)
                # 角加速度を計算
                q_ddot = dynamics(q, q_dot, torque)
                # オイラー法で角度と角速度を更新
                q = q + q_dot * dt
                q_dot = q_dot + q_ddot * dt

                # 計算した角度とトルクを記録
                trajectory.append(q)
                torque_seq.append(torque)

            # 軌道とトルクをリストに格納
            trajectories.append(np.array(trajectory))
            torques.append(np.array(torque_seq))

        return trajectories, torques


# トルクと角度の計算    
def calculate_torque_and_angle(current_angle, target_angle, q_dot, KP, KD, dt):  
    torque = pd_control(target_angle,current_angle, q_dot, KP, KD)               
    q_ddot = dynamics(current_angle, q_dot, torque)                                
    current_angle = current_angle + q_dot * dt                                                  
    q_dot = q_dot + q_ddot * dt
    return current_angle, q_dot, torque, q_ddot  

# ルンゲ・クッタ法を使ったトルクと角度の計算
def calculate_torque_and_angle_rk4(current_angle, q_dot, KP, KD, dt, esn_pred_angle):
    """
    4次のルンゲ・クッタ法を使用して角度、速度、トルクを計算します。
    """
    # PD制御によるトルク計算
    def torque_func(angle, velocity):
        return pd_control(esn_pred_angle, angle, velocity, KP, KD)

    # 角速度の微分方程式（加速度計算）
    def dynamics_func(angle, velocity, torque):
        return dynamics(angle, velocity, torque)

    # k1
    torque_k1 = torque_func(current_angle, q_dot)
    q_ddot_k1 = dynamics_func(current_angle, q_dot, torque_k1)
    
    # k2
    torque_k2 = torque_func(current_angle + 0.5 * dt * q_dot, q_dot + 0.5 * dt * q_ddot_k1)
    q_ddot_k2 = dynamics_func(current_angle + 0.5 * dt * q_dot, q_dot + 0.5 * dt * q_ddot_k1, torque_k2)
    
    # k3
    torque_k3 = torque_func(current_angle + 0.5 * dt * (q_dot + 0.5 * dt * q_ddot_k1), q_dot + 0.5 * dt * q_ddot_k2)
    q_ddot_k3 = dynamics_func(current_angle + 0.5 * dt * (q_dot + 0.5 * dt * q_ddot_k1), q_dot + 0.5 * dt * q_ddot_k2, torque_k3)
    
    # k4
    torque_k4 = torque_func(current_angle + dt * (q_dot + 0.5 * dt * q_ddot_k2), q_dot + dt * q_ddot_k3)
    q_ddot_k4 = dynamics_func(current_angle + dt * (q_dot + 0.5 * dt * q_ddot_k2), q_dot + dt * q_ddot_k3, torque_k4)
    
    # 最終的な更新
    current_angle += dt * (q_dot + (q_ddot_k1 + 2 * q_ddot_k2 + 2 * q_ddot_k3 + q_ddot_k4) / 6)
    q_dot += dt * (q_ddot_k1 + 2 * q_ddot_k2 + 2 * q_ddot_k3 + q_ddot_k4) / 6
    torque = (torque_k1 + 2 * torque_k2 + 2 * torque_k3 + torque_k4) / 6

    return current_angle, q_dot, torque

# 目標角度を動的に生成する関数
def dynamic_target_angle(step, total_steps, start_angle, end_angle):
    """ 
    目標角度を時間とともに動的に変化させる。
    step: 現在のステップ
    total_steps: 目標角度の移動が完了するまでのステップ数
    start_angle: 初期角度（開始時の目標角度）
    end_angle: 最終角度（移動完了時の目標角度）
    """
    if step <= total_steps:
        # 線形補間による目標角度の変化
        return start_angle + (end_angle - start_angle) * (step / total_steps)
    else:
        # 目標角度移動完了後は固定
        return end_angle 

# 目標角度を動的に生成する関数
def dynamic_target_angle_sin(step, total_steps, start_angle, end_angle):
    """ 
    目標角度を時間とともに曲線的に変化させる。
    step: 現在のステップ
    total_steps: 目標角度の移動が完了するまでのステップ数
    start_angle: 初期角度（開始時の目標角度）
    end_angle: 最終角度（移動完了時の目標角度）
    """
    if step <= total_steps:
        # 正規化された時間
        t = step / total_steps

        # 一方の軸（例えば x 軸）を線形的に移動
        linear_component = start_angle[0] + (end_angle[0] - start_angle[0]) * t
        
        # もう一方の軸（例えば y 軸）をsin的に変化させ、0から1まで滑らかに変化
        sin_component = start_angle[1] + (end_angle[1] - start_angle[1]) * np.sin(np.pi * t / 2)

        return np.array([linear_component, sin_component])
    else:
        # 目標角度移動完了後は固定
        return end_angle

# 円を描くように目標角度を動的に生成する関数
def dynamic_target_angle_circular(step, total_steps, center_angle, start_angle, end_angle):
    """
    目標角度を円を描くように時間とともに動的に変化させる。
    step: 現在のステップ
    total_steps: 目標角度が1周または部分的な円を描くまでのステップ数
    center_angle: 円の中心角度（[中心角度1, 中心角度2]）
    start_angle: 開始時の角度（円周上の開始点）
    end_angle: 円周上の終了点（戻ってくる角度）
    """
    # 円周角をステップ数に基づいて進める
    angle_progress = (2 * np.pi * step / total_steps)
    
    # 各座標の円周上の位置を計算
    target_angle_x = center_angle[0] + (start_angle[0] - center_angle[0]) * np.cos(angle_progress) - (start_angle[1] - center_angle[1]) * np.sin(angle_progress)
    target_angle_y = center_angle[1] + (start_angle[0] - center_angle[0]) * np.sin(angle_progress) + (start_angle[1] - center_angle[1]) * np.cos(angle_progress)
    
    return np.array([target_angle_x, target_angle_y])


# エコーステートネットワーク
class ESN:# バッチ学習
    def train(self, U, D, optimizer, trans_len = None):
        '''
        U: 教師データの入力, データ長×N_u
        D: 教師データの出力, データ長×N_y
        optimizer: 学習器
        trans_len: 過渡期の長さ
        return: 学習前のモデル出力, データ長×N_y
        '''
        train_len = len(U)
        if trans_len is None:
            trans_len = 0
        Y = []


        # 時間発展
        for n in range(train_len):
            x_in = self.Input(U[n])

            # フィードバック結合
            if self.Feedback is not None:
                x_back = self.Feedback(self.y_prev)
                x_in += x_back

            # ノイズ
            if self.noise is not None:
                x_in += self.noise

            # リザバー状態ベクトル
            x = self.Reservoir(x_in)

            # 分類問題の場合は窓幅分の平均を取得
            if self.classification:
                self.window = np.append(self.window, x.reshape(1, -1),
                                        axis=0)
                self.window = np.delete(self.window, 0, 0)
                x = np.average(self.window, axis=0)

            # 目標値
            d = D[n]
            d = self.inv_output_func(d)

            # 学習器
            if n > trans_len:  # 過渡期を過ぎたら
                optimizer(d, x)

            # 学習前のモデル出力
            y = self.Output(x)
            Y.append(self.output_func(y))
            self.y_prev = d

        # 学習済みの出力結合重み行列を設定
        self.Output.setweight(optimizer.get_Wout_opt())

        # モデル出力（学習前）
        return np.array(Y)
    # 各層の初期化
    def __init__(self, N_u, N_y, N_x, density=0.05, input_scale=1.0,
                 rho=0.95, activation_func=np.tanh, fb_scale = None,
                 fb_seed=0, noise_level = None, leaking_rate=1.0,
                 output_func=identity, inv_output_func=identity,
                 classification = False, average_window = None,
                 gIn = 1, dt = 0.01):
        '''
        param N_u: 入力次元
        param N_y: 出力次元
        param N_x: リザバーのノード数
        param density: リザバーのネットワーク結合密度
        param input_scale: 入力スケーリング
        param rho: リカレント結合重み行列のスペクトル半径
        param activation_func: リザバーノードの活性化関数
        param fb_scale: フィードバックスケーリング（default: None）
        param fb_seed: フィードバック結合重み行列生成に使う乱数の種
        param leaking_rate: leaky integratorモデルのリーク率
        param output_func: 出力層の非線形関数（default: 恒等写像）
        param inv_output_func: output_funcの逆関数
        param classification: 分類問題の場合はTrue（default: False）
        param average_window: 分類問題で出力平均する窓幅（default: None）
        '''
        self.Input = Input(N_u, N_x, input_scale)
        self.Input_fire = Input_fire(N_u, N_x, gIn)
        self.Reservoir = Reservoir(N_x, density, rho, activation_func, 
                                   leaking_rate)
        self.Reservoir_fire = Reservoir_fire(N_x, density, rho, activation_func, dt)
        self.Output = Output(N_x, N_y)
        self.Output_fire = Output_fire(N_x, N_y)
        self.N_u = N_u
        self.N_y = N_y
        self.N_x = N_x
        self.y_prev = np.zeros(N_y)
        self.output_func = output_func
        self.inv_output_func = inv_output_func
        self.classification = classification

        # 出力層からのリザバーへのフィードバックの有無
        if fb_scale is None:
            self.Feedback = None
        else:
            self.Feedback = Feedback(N_y, N_x, fb_scale, fb_seed)

        # リザバーの状態更新おけるノイズの有無
        if noise_level is None:
            self.noise = None
        else:
            np.random.seed(seed=0)
            self.noise = np.random.uniform(-noise_level, noise_level, 
                                           (self.N_x, 1))

        # 分類問題か否か
        if classification:
            if average_window is None:
                raise ValueError('Window for time average is not given!')
            else:
                self.window = np.zeros((average_window, N_x))

    # バッチ学習
    def train(self, U, D, optimizer, trans_len = None):
        '''
        U: 教師データの入力, データ長×N_u
        D: 教師データの出力, データ長×N_y
        optimizer: 学習器
        trans_len: 過渡期の長さ
        return: 学習前のモデル出力, データ長×N_y
        '''
        train_len = len(U)
        if trans_len is None:
            trans_len = 0
        Y = []

        # 時間発展
        for n in range(train_len):
            x_in = self.Input(U[n])

            # フィードバック結合
            if self.Feedback is not None:
                x_back = self.Feedback(self.y_prev)
                x_in += x_back

            # ノイズ
            if self.noise is not None:
                x_in += self.noise

            # リザバー状態ベクトル
            x = self.Reservoir(x_in)

            # 分類問題の場合は窓幅分の平均を取得
            if self.classification:
                self.window = np.append(self.window, x.reshape(1, -1),
                                        axis=0)
                self.window = np.delete(self.window, 0, 0)
                x = np.average(self.window, axis=0)

            # 目標値
            d = D[n]
            d = self.inv_output_func(d)

            # 学習器
            if n > trans_len:  # 過渡期を過ぎたら
                optimizer(d, x)

            # 学習前のモデル出力
            y = self.Output(x)
            Y.append(self.output_func(y))
            self.y_prev = d

        # 学習済みの出力結合重み行列を設定
        self.Output.setweight(optimizer.get_Wout_opt())

        # モデル出力（学習前）
        return np.array(Y)
    
    def train_repeat(self, U, D, optimizer, n_learn_loops=10, trans_len=None):

        train_len = len(U)
        if trans_len is None:
            trans_len = 0

        Y_history = []  # 各ループごとの学習前出力を保存するリスト

        for j in range(n_learn_loops):
            print(f"Learning loop: {j+1}/{n_learn_loops}")
            Y = []  # 各ループのモデル出力

            # 時間発展
            for n in range(train_len):
                x_in = self.Input_fire(Umghfnnhfg
                                       [n])

                # フィードバック結合
                if self.Feedback is not None:
                    x_back = self.Feedback(self.y_prev)
                    x_in += x_back

                # ノイズ
                if self.noise is not None:
                    x_in += self.noise

                # リザバー状態ベクトル
                x = self.Reservoir_fire(x_in)

                # 分類問題の場合は窓幅分の平均を取得
                if self.classification:
                    self.window = np.append(self.window, x.reshape(1, -1), axis=0)
                    self.window = np.delete(self.window, 0, 0)
                    x = np.average(self.window, axis=0)

                # 目標値
                d = D[n]
                d = self.inv_output_func(d)

                # 学習器
                if n > trans_len:  # 過渡期を過ぎたら
                    optimizer(d, x)

                # 学習前のモデル出力
                y = self.Output(x)
                Y.append(self.output_func(y))
                self.y_prev = d

            # 現在のループのモデル出力を保存
            Y_history.append(np.array(Y))

            # 学習済みの出力結合重み行列を更新
            self.Output.setweight(optimizer.get_Wout_opt())

            # 学習進捗の確認 (例: コサイン類似度やMSEを表示)
            performance = np.mean((np.array(Y) - D)**2)
            print(f"  Loop {j+1}, MSE: {performance:.6f}")

        # 全ループでのモデル出力履歴を返す
        return np.array(Y_history)


    # バッチ学習後の予測
    def predict(self, U):
        test_len = len(U)
        Y_pred = []

        # 時間発展
        for n in range(test_len):
            x_in = self.Input(U[n])

            # フィードバック結合
            if self.Feedback is not None:
                x_back = self.Feedback(self.y_prev)
                x_in += x_back

            # リザバー状態ベクトル
            x = self.Reservoir(x_in)

            # 分類問題の場合は窓幅分の平均を取得
            if self.classification:
                self.window = np.append(self.window, x.reshape(1, -1),
                                        axis=0)
                self.window = np.delete(self.window, 0, 0)
                x = np.average(self.window, axis=0)

            # 学習後のモデル出力
            y_pred = self.Output(x)
            Y_pred.append(self.output_func(y_pred))
            self.y_prev = y_pred

        # モデル出力（学習後）
        return np.array(Y_pred)
        
    # バッチ学習後の軌道予測これがただしい
    def trajectory1(self, U0, steps, KP, KD, dt, target_angles): 
        Y_pred = [U0] #予測した角度
        torque_set =[np.zeros(2)] #予測したトルク
        current_angle_set = [U0]
        q_dot_set = [np.zeros(2)]
        q_ddot_set = [np.zeros(2)]
        error_set = [np.zeros(2)]

        # 時間発展
        for n in range(steps):
            
            x_in = self.Input(current_angle_set[n])
            x = self.Reservoir(x_in)
            y_pred = self.Output(x)
            
            Y_pred.append(y_pred) 
            
            # PDコントローラ
            current_angle, q_dot, torque, q_ddot = calculate_torque_and_angle(current_angle_set[n], Y_pred[n+1], q_dot_set[n], KP, KD, dt)
            
            current_angle_set.append(current_angle)
            q_dot_set.append(q_dot)
            q_ddot_set.append(q_ddot)
            torque_set.append(torque)
            
            error = Y_pred[n+1] - current_angle_set[n+1]
            error_set.append(error)
            
        # モデル出力（学習後）
        return np.array(Y_pred) , np.array(torque_set), np.array(current_angle_set), np.array(error_set)

    # バッチ学習後の軌道予測（初期角度とトルクの計算）
    def trajectory2(self, U0, steps, KP, KD, dt, target_angles): 
        Y_pred = [U0]                   # 予測した角度
        torque_set = [np.zeros(2)]      # 予測したトルク
        current_angle_set = [U0]
        q_dot_set = [np.zeros(2)]
        error_set = [np.zeros(2)]

        # 時間発展
        for n in range(steps):
        
            x_in = self.Input(current_angle_set[n])     # 現在の角度からリザバー入力を生成
            x = self.Reservoir(x_in)                    # リザバー状態更新
            
            y_pred = self.Output(x)                     # 角度予測
            
            Y_pred.append(y_pred) 
            
            # PDコントローラを用いてトルクを計算
            torque = pd_control(target_angles, y_pred, q_dot_set[n], KP, KD)
            q_ddot = dynamics(y_pred, q_dot_set[n], torque)
            
            # 次のステップの角度、速度を更新
            next_angle = y_pred + q_dot_set[n] * dt
            q_dot = q_dot_set[n] + q_ddot * dt
            
            # 角度と速度の更新
            current_angle_set.append(next_angle)
            q_dot_set.append(q_dot)
            torque_set.append(torque)
            
            # 誤差の記録
            error = y_pred - next_angle
            error_set.append(error)
        
        return np.array(Y_pred), np.array(torque_set), np.array(current_angle_set), np.array(error_set)

    # バッチ学習後の軌道予測（初期角度とトルクの計算） 「ウォームアップ」フェーズを導入
    def trajectory3(self, U0, steps, KP, KD, dt, target_angles): 
        Y_pred = [U0]                   # 予測した角度
        torque_set = [np.zeros(2)]      # 予測したトルク
        current_angle_set = [U0]
        q_dot_set = [np.zeros(2)]
        error_set = [np.zeros(2)]

        # 時間発展
        for n in range(steps):
            x_in = self.Input(current_angle_set[n])     # 現在の角度からリザバー入力を生成
            x = self.Reservoir(x_in)                    # リザバー状態更新
            
            # ウォームアップフェーズの導入
            if n < 5:  # 最初の10ステップはウォームアップ
                y_pred = self.Output(x)  # 通常の予測
                y_pred = y_pred * 0.1 + Y_pred[-1] * 0.9  # 前回の予測を少し混ぜる
            else:
                y_pred = self.Output(x)  # 通常の予測フェーズ
            
            Y_pred.append(y_pred)
            
            # PDコントローラを用いてトルクを計算
            torque = pd_control(target_angles, y_pred, q_dot_set[n], KP, KD)
            q_ddot = dynamics(y_pred, q_dot_set[n], torque)
            
            # 次のステップの角度、速度を更新
            next_angle = y_pred + q_dot_set[n] * dt
            q_dot = q_dot_set[n] + q_ddot * dt
            
            # 角度と速度の更新
            current_angle_set.append(next_angle)
            q_dot_set.append(q_dot)
            torque_set.append(torque)
            
            # 誤差の記録
            error = y_pred - next_angle
            error_set.append(error)
        
        return np.array(Y_pred), np.array(torque_set), np.array(current_angle_set), np.array(error_set)

    #ルンゲ・クッタ法
    def trajectory4(self, U0, steps, KP, KD, dt, target_angles): 
        Y_pred = [U0]  # 予測した角度
        torque_set = [np.zeros(2)]  # 予測したトルク
        current_angle_set = [U0]
        q_dot_set = [np.zeros(2)]
        error_set = [np.zeros(2)]

        # 時間発展
        for n in range(steps):
            x_in = self.Input(current_angle_set[n])
            x = self.Reservoir(x_in)
            y_pred = self.Output(x)
            
            Y_pred.append(y_pred) 

            # ルンゲ・クッタ法でトルクと角度を計算
            current_angle, q_dot, torque = calculate_torque_and_angle_rk4(
                current_angle_set[n], q_dot_set[n], KP, KD, dt, y_pred
            )
            
            current_angle_set.append(current_angle)
            q_dot_set.append(q_dot)
            torque_set.append(torque)
            
            error = y_pred - current_angle
            error_set.append(error)
            
        # モデル出力（学習後）
        return np.array(Y_pred), np.array(torque_set), np.array(current_angle_set), np.array(error_set)
    
    #500ステップ目にトルクの変化が発生
    def trajectory5(self, U0, steps, KP, KD, dt, alpha = 1): 
        Out_set = [U0]  # 予測した角度
        torque_set = [np.zeros(2)]  # 予測したトルク
        current_angle_set = [U0]
        q_dot_set = [np.zeros(2)]
        q_ddot_set = [np.zeros(2)]
        error_cntl_set = [np.zeros(2)]

        # 外部トルクの大きさと適用するステップ
        external_torque = np.array([5,5])  # 任意の外部トルク
        apply_step = 500  # 外部トルクを加えるステップ

        # 時間発展
        for n in range(steps):
            
            x_in = self.Input(current_angle_set[n])
            x = self.Reservoir(x_in)
            y_pred = self.Output(x)
            
            
            
            Out = y_pred + alpha * current_angle_set[n]  
            Out_set.append(Out)
            

                
            # PDコントローラでトルクを計算
            current_angle, q_dot, torque, q_ddot = calculate_torque_and_angle(
                current_angle_set[n], Out_set[n+1], q_dot_set[n], KP, KD, dt
            )
            
            # 外部トルクの適用
            if n == apply_step:
                torque += external_torque  # 外部トルクを追加
                q_ddot = dynamics(current_angle, q_dot, torque)  # 新たなトルクで角加速度を再計算
                q_dot += q_ddot * dt  # 角速度更新
                current_angle += q_dot * dt  # 角度更新


            # 状態を更新
            current_angle_set.append(current_angle)
            q_dot_set.append(q_dot)
            q_ddot_set.append(q_ddot)
            torque_set.append(torque)
            
            error_cntl = Out_set[n+1] - current_angle_set[n+1]
            error_cntl_set.append(error_cntl)
            
        # モデル出力（学習後）
        return np.array(Out_set), np.array(torque_set), np.array(current_angle_set), np.array(error_cntl_set)

    # バッチ学習後の予測（自律系のフリーラン）
    def run(self, U):
        test_len = len(U)
        Y_pred = []
        y = U[0]

        # 時間発展
        for n in range(test_len):
            x_in = self.Input(y)

            # フィードバック結合
            if self.Feedback is not None:
                x_back = self.Feedback(self.y_prev)
                x_in += x_back

            # リザバー状態ベクトル
            x = self.Reservoir(x_in)

            # 学習後のモデル出力
            y_pred = self.Output(x)
            Y_pred.append(self.output_func(y_pred))
            y = y_pred
            self.y_prev = y

        return np.array(Y_pred)

    # オンライン学習と予測
    def adapt(self, U, D, optimizer):
        data_len = len(U)
        Y_pred = []
        Wout_abs_mean = []

        # 出力結合重み更新
        for n in np.arange(0, data_len, 1):
            x_in = self.Input(U[n])
            x = self.Reservoir(x_in)
            d = D[n]
            d = self.inv_output_func(d)
            
            # 学習
            Wout = optimizer(d, x)

            # モデル出力
            y = np.dot(Wout, x)
            Y_pred.append(y)
            Wout_abs_mean.append(np.mean(np.abs(Wout)))

        return np.array(Y_pred), np.array(Wout_abs_mean)
