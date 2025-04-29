import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import bicgstab, LinearOperator, spilu
from concurrent.futures import ThreadPoolExecutor, as_completed

class MT1DForwardFDM:
    def __init__(self, rho, h, t_sample=71, dz=20.0):
        """
        初始化 MT 1D FDM 正演类
        参数:
            rho: 各层电阻率 (Ωm) 列表
            h: 各层厚度 (m) 列表
            t_sample: 采样周期数 (要求 7 * x + 1)
            dz: 网格间隔 (m)
        """
        
        # 处理 t_sample 参数
        self.t_sample = t_sample
        self.dz = dz
        self.T = np.logspace(-3, 4, t_sample)  # 10^-3 ~ 10^4s
        # 计算网格划分
        self._generate_grid(rho, h)
        self.pc = np.zeros(t_sample)
        self.ph = np.zeros(t_sample)
        # 计算结果存储
        self.Z_TE_surface = np.zeros(t_sample, dtype=np.complex128)
        self.Ex_TE_surface = np.zeros(t_sample, dtype=np.complex128)
        self.Hy_TE_surface = np.zeros(t_sample, dtype=np.complex128)
        self._run_forward()
        
        
    def _generate_grid(self, rho, h):
        """ 根据地层参数生成网格化电阻率列表 """
        r = []
        for j in range(len(h)):
            n_layers = max(1, int(h[j] / self.dz))
            r += [rho[j]] * n_layers
        
        # 计算深度，扩展底部
        skin_depth = max(np.sum(h), 1e3 * np.sqrt(rho[-1]))
        bottom_layers = max(100, int(skin_depth / self.dz))
        r += [rho[-1]] * (bottom_layers + 1)

        self.nz = len(r)
        self.r = np.array(r)
        
    def _solve_frequency(self, tt):
        """ 计算单个频率的 Ex, Hy 并返回相应的视电阻率和相位 """
        i = 1j
        Omega = 2 * np.pi / self.T[tt]
        mu = 4 * np.pi * 1e-7
        eps = 8.8419e-12

        # 构造稀疏矩阵 K 和右端项 P
        K = lil_matrix((self.nz, self.nz), dtype=np.complex128)
        P = lil_matrix((self.nz, 1), dtype=np.complex128)

        # 激发
        K[0, 0] = 1
        P[0] = 1

        # 内部差分
        for j in range(1, self.nz - 1):
            if j == 1 or j == self.nz - 2:
                K[j, j-1:j+2] = [
                    1/self.dz**2,
                    mu * eps * Omega**2 + i * mu * Omega / self.r[j-1] - 2/self.dz**2,
                    1/self.dz**2
                ]
            else:
                K[j, j-2:j+3] = [
                    -1 / (12 * self.dz**2),
                    4 / (3 * self.dz**2),
                    mu * eps * Omega**2 + i * mu * Omega / self.r[j-1] - 5 / (2 * self.dz**2),
                    4 / (3 * self.dz**2),
                    -1 / (12 * self.dz**2)
                ]

        # 底部吸收边界
        K[self.nz - 1, self.nz - 2] = -1 / self.dz
        K[self.nz - 1, self.nz - 1] = 1 / self.dz \
            + np.sqrt(- i * Omega * mu / self.r[-1] - Omega**2 * mu * eps)
        
        K = K.tocsc()
        P = P.toarray().flatten()
        
        # 用于观察非0元素分布
        # if tt == 1:
        #     plt.figure(figsize=(10, 10))
        #     plt.spy(K, markersize=2, color='blue')
        #     plt.title("Sparse Matrix Non-zero Elements Distribution")
        #     plt.xlabel("Column Index")
        #     plt.ylabel("Row Index")
        #     plt.show()
        
        # 求解
        # 计算地表和半空间场
        try:
            ilu = spilu(K, fill_factor=50, drop_tol=1e-8)  # 增加填充元并设置丢弃容差
            M = LinearOperator(K.shape, ilu.solve)
        except Exception as e:
            print(f"ILU 分解失败：{e}")
            exit()

        # 使用预处理的 BiCGSTAB 求解
        Ex, info = bicgstab(K, P, M=M, rtol=1e-16, maxiter=50)
        
        if info == 0:
            residual = np.linalg.norm(K.dot(Ex) - P)
            print(f"收敛成功，最终残差：{residual:.2e}")
        else:
            print(f"未收敛，状态码：{info}")
        # Ex = spsolve(K, P)
        
        Ex_g = Ex[0]
        Hy_g = (-11*Ex[0] + 18*Ex[1] -9 * Ex[2] + 2*Ex[3]) / (6*self.dz) / (i * mu * Omega)
        Z_TE_g = Ex_g / Hy_g
        self.Ex_TE_surface[tt] = Ex_g
        self.Hy_TE_surface[tt] = Hy_g
        self.Z_TE_surface[tt] = Z_TE_g
        self.pc[tt] = np.abs(Z_TE_g)**2 / (mu * Omega)
        self.ph[tt] = -np.angle(Z_TE_g, deg=True)

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
    import time
    # 示例参数
    rho = [100, 50, 10, 50, 30, 15, 100]       # 各层电阻率
    h = [450, 700, 650, 400, 1850, 3500]            # 各层厚度
    t_sample = 141                                # 周期样本数
    dz = 5.0                                           # 网格间距

    # 调用并行 FDM 正演函数
    start_time = time.time()
    model_1 = MT1DForwardFDM(rho, h, t_sample, dz)
    model_1.plot_result_surface()
    end_time = time.time()
    print(f"运行时间：{end_time - start_time:.6f} 秒")