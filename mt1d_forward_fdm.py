import matplotlib.pyplot as plt
import numpy as np
import warnings
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
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
            mode: 'TE' or 'TM' （选择极化模式）
        """
        
        # 处理 t_sample 参数
        if (x := (t_sample - 1) % 7) != 0:
            t_sample_new = t_sample + 7 - x
            warnings.warn(f"t_sample={t_sample} 无效，已重设为 {t_sample_new}", UserWarning)
            t_sample = t_sample_new

        self.rho = rho
        self.h = h
        self.t_sample = t_sample
        self.dz = dz
        self.T = np.logspace(-3, 4, t_sample)  # 10^-3 ~ 10^4s
        
        # 计算网格划分
        self._generate_grid()
        self.pc = np.zeros(t_sample)
        self.ph = np.zeros(t_sample)
        # 计算结果存储
        self.Z_TE_surface = np.zeros(t_sample, dtype=np.complex128)
        self.Ex_TE_surface = np.zeros(t_sample, dtype=np.complex128)
        self.Hy_TE_surface = np.zeros(t_sample, dtype=np.complex128)
        self._run_forward()
        
        
    def _generate_grid(self):
        """ 根据地层参数生成网格化电阻率列表 """
        r = []
        for j in range(len(self.h)):
            n_layers = max(1, int(self.h[j] / self.dz))
            r += [self.rho[j]] * n_layers
        
        # 计算深度，扩展底部
        skin_depth = max(np.sum(self.h), 1e3 * np.sqrt(self.rho[-1]))
        bottom_layers = max(100, int(skin_depth / self.dz))
        r += [self.rho[-1]] * bottom_layers

        self.nz = len(r) + 2  # 两个附加层用于边界处理
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

        # 地表激发
        K[0, 0:2] = [1/2, 1/2]
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
        sqrt_term = np.sqrt(i * Omega * mu / self.r[-1] + mu * eps * Omega**2)
        K[self.nz - 1, self.nz - 1] = 1 / self.dz + sqrt_term
        
        # K[self.nz - 1, self.nz - 1] = 1
        
        K = K.tocsr()
        P = P.toarray().flatten()
        
        # 求解
        # 计算地表和半空间场
        Ex = spsolve(K, P)
        Ex_g = (Ex[0] + Ex[1]) / 2
        Hy_g = (Ex[1] - Ex[0]) / self.dz / (i * mu * Omega)
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

    def get_pc(self):
        return self.pc
    
    def get_ph(self):
        return self.ph
    
    def get_Z_surface(self):
        return self.Z_TE_surface

    def get_E_surface(self):
        return self.Ex_TE_surface
    
    def get_H_surface(self):
        return self.Hy_TE_surface

    def plot_result_surface(self):
        """ 绘制视电阻率和相位曲线 """
        fig, axes = plt.subplots(2, 1, figsize=(16, 9))
        x_impedance = self.T
        axes[0].set_title("Apparent Resistivity")
        axes[0].semilogx(x_impedance, self.pc, 'b-')
        axes[1].set_title("Phase")
        axes[1].semilogx(x_impedance, self.ph, 'r-')
        plt.show()


# 示例参数
rho = [1000, 100, 1e6]       # 各层电阻率
h = [500, 500]                 # 各层厚度
t_sample = 71                                # 周期样本数
dz = 10.0                                           # 网格间距

# 调用并行 FDM 正演函数
model_1 = MT1DForwardFDM(rho, h, t_sample, dz)
model_1.plot_result_surface()
