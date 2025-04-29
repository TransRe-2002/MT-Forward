import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import bicgstab, spilu, LinearOperator
from concurrent.futures import ThreadPoolExecutor, as_completed


class MT1DForwardFEM:
    def __init__(self, rho, h, t_sample=71, DZ=30.0):
        """
        初始化 MT 1D FEM 正演类
        参数:
            rho: 各层电阻率 (Ωm) 列表
            h: 各层厚度 (m) 列表
            t_sample: 采样周期数
            DZ: 单元尺寸 (m)
        """
        self.DZ = DZ
        self.t_sample = t_sample
        self.T = np.logspace(-3, 4, t_sample)
        self.rho_node, self.NE, self.NP = self._generate_grid(rho, h)
        
        self.pc = np.zeros(t_sample)
        self.ph = np.zeros(t_sample)
        
        self.ME = np.zeros((4, self.NE), dtype=int)
        for i in range(self.NE):
            self.ME[0, i] = 1 + i * 3 - 1
            self.ME[1, i] = 2 + i * 3 - 1
            self.ME[2, i] = 3 + i * 3 - 1
            self.ME[3, i] = 4 + i * 3 - 1
        
        # 计算结果存储
        self.Z_TE_surface = np.zeros(t_sample, dtype=np.complex128)
        self.Ex_TE_surface = np.zeros(t_sample, dtype=np.complex128)
        self.Hy_TE_surface = np.zeros(t_sample, dtype=np.complex128)
        self._run_forward()
        
    def _generate_grid(self, rho, h):
        rho_node = []
        for i in range(np.size(h)):
            nodes = int(h[i] / (self.DZ / 3))
            rho_node += [rho[i]] * nodes
            
        skin_depth = max(np.sum(h), 1e3 * np.sqrt(rho[-1]))
        rho_node += [rho[-1]] * max(100, int(skin_depth / (self.DZ / 3)))
        last = max(0, 3 - len(rho_node) % 3) + 1
        rho_node += [rho[-1]] * last

        assert((len(rho_node) - 1) % 3 == 0)
        NZ = int((np.size(rho_node) - 1) / 3)
        
        return np.array(rho_node), NZ, 1 + NZ * 3
        
    def _solve_frequency(self, tt):
        K1 = lil_matrix((self.NP, self.NP), dtype=complex)
        K2 = lil_matrix((self.NP, self.NP), dtype=complex)
        K3 = lil_matrix((self.NP, self.NP), dtype=complex)
        P = np.zeros(self.NP, dtype=complex)

        for h in range(self.NE):
            K = np.array([
                [148, -189,  54, -13],
                [-189, 432, -297, 54],
                [54, -297, 432, -189],
                [-13, 54, -189, 148]
            ]) / (40 * self.DZ)

            for j in range(4):
                NJ = self.ME[j, h]
                for k in range(4):
                    NK = self.ME[k, h]
                    K1[NJ, NK] += K[j, k]

        mu = 4e-7 * np.pi
        w = 2 * np.pi / self.T[tt]
        m = 1j * w * mu
        eps = 8.8419e-12
        
        for h in range(self.NE):
            idx = np.array([self.ME[0, h], self.ME[1, h], self.ME[2, h], self.ME[3, h]])
            r = 1 / self.rho_node[idx]
            K = np.zeros((4, 4))
            K[0, 0] = 357 * r[0] + 216 * r[1] - 81 * r[2] + 20 * r[3]
            K[1, 0] = 216 * r[0] + 324 * r[1] - 162 * r[2] + 18 * r[3]
            K[1, 1] = 324 * r[0] + 2187 * r[1] + 0 * r[2] + 81 * r[3]
            K[2, 0] = -81 * r[0] - 162 * r[1] + 81 * r[2] + 18 * r[3]
            K[2, 1] = -162 * r[0] + 0 * r[1] - 0 * r[2] - 162 * r[3]
            K[2, 2] = 81 * r[0] + 0 * r[1] + 2187 * r[2] + 324 * r[3]
            K[3, 0] = 20 * r[0] + 18 * r[1] + 18 * r[2] + 20 * r[3]
            K[3, 1] = 18 * r[0] + 81 * r[1] - 162 * r[2] - 81 * r[3]
            K[3, 2] = 18 * r[0] - 162 * r[1] + 324 * r[2] + 216 * r[3]
            K[3, 3] = 20 * r[0] - 81 * r[1] + 216 * r[2] + 357 * r[3]
            K[0, 1] = K[1, 0]
            K[0, 2] = K[2, 0]
            K[0, 3] = K[3, 0]
            K[1, 2] = K[2, 1]
            K[1, 3] = K[3, 1]
            K[2, 3] = K[3, 2]

            for j in range(4):
                NJ = self.ME[j, h]
                for k in range(4):
                    NK = self.ME[k, h]
                    K2[NJ, NK] += K[j, k] * m * self.DZ / 6720 + w**2 * mu * eps

        a = np.sqrt(-m / rho[-1] - w**2 * mu * eps)
        K3[-1, -1] = a

        v = K1 - K2 + K3
        v = v.tocsc()
        P[0] = 1
        
        # 用于观察非0元素分布
        # if tt == 1:
        #     plt.figure(figsize=(10, 10))
        #     plt.spy(v, markersize=2, color='blue')
        #     plt.title("Sparse Matrix Non-zero Elements Distribution")
        #     plt.xlabel("Column Index")
        #     plt.ylabel("Row Index")
        #     plt.show()
        
        # 求解
        # 计算地表和半空间场
        try:
            ilu = spilu(v, fill_factor=50, drop_tol=1e-8)  # 增加填充元并设置丢弃容差
            M = LinearOperator(v.shape, ilu.solve)
        except Exception as e:
            print(f"ILU 分解失败：{e}")
            exit()

        # 使用预处理的 BiCGSTAB 求解
        Ex, info = bicgstab(v, P, M=M, rtol=1e-16, maxiter=50)
        
        if info == 0:
            residual = np.linalg.norm(v.dot(Ex) - P)
            print(f"收敛成功，最终残差：{residual:.2e}")
        else:
            print(f"未收敛，状态码：{info}")
        # Ex = spsolve(v, P)
        
        Hy = (-11 * Ex[0] + 18 * Ex[1] - 9 * Ex[2] + 2 * Ex[3]) / (self.DZ * 2 * m)
        Z = Ex[0] / Hy
        self.pc[tt] = np.abs(Z)**2 / (w * mu)
        self.ph[tt] = -np.angle(Z, deg=True)
        
    def _run_forward(self):
        """ 并行计算所有周期的正演 """
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(self._solve_frequency, tt): tt for tt in range(self.t_sample)}
            for future in as_completed(futures):
                future.result()
        
    def plot_result_surface(self):
        """ 绘制视电阻率和相位曲线 """
        fig, axes = plt.subplots(2, 1, figsize=(16, 9))
        x_impedance = self.T
        axes[0].set_title("Apparent Resistivity")
        axes[0].semilogx(x_impedance, self.pc, 'b-')
        axes[1].set_title("Phase")
        axes[1].semilogx(x_impedance, self.ph, 'r-')
        plt.show()
        
if __name__ == '__main__':
    rho = [100, 50, 10, 50, 30, 15, 100]       # 各层电阻率
    h = [450, 700, 650, 400, 1850, 3500]                 # 各层厚度
    t_sample = 141                                # 周期样本数
    DZ = 15.0                                           # 网格间距
    
    import time
    start_time = time.time()
    model = MT1DForwardFEM(rho, h, t_sample, DZ)
    model.plot_result_surface()
    end_time = time.time()
    print(f"运行时间：{end_time - start_time:.6f} 秒")

    