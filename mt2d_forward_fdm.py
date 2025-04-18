import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from concurrent.futures import ProcessPoolExecutor, as_completed

# 模型生成
class Model():
    def __init__(self, y=10000, z=10000, dy=10.0, dz=10.0, rho_background=500, **anomalies):
        self.dy = dy
        self.dz = dz
        self.ny = int(y / dy)
        self.nz = int(z / dz)
        self.rho = np.ones((self.ny, self.nz)) * rho_background
    
        for key, anomaly in anomalies.items():
            (y0, z0) = anomaly['top_left']
            (y1, z1) = anomaly['bottom_right']
            rho_anomaly = anomaly['rho']

            iy0 = int(y0 / self.dy)
            iy1 = int(y1 / self.dy)
            iz0 = int(z0 / self.dz)
            iz1 = int(z1 / self.dz)

            iy0 = max(0, min(self.ny - 1, iy0))
            iy1 = max(0, min(self.ny - 1, iy1))
            iz0 = max(0, min(self.nz - 1, iz0))
            iz1 = max(0, min(self.nz - 1, iz1))

            self.rho[iz0 : iz1 + 1, iy0 : iy1 + 1] = rho_anomaly
        
   
class MT2DForwardFDM():
    def __init__(self, model, t_sample=71):
        self.dy = model.dy
        self.dz = model.dz
        
        # 网格扩展一圈作为边界
        self.ny = model.ny + 2
        self.nz = model.nz + 2
        
        # 导入模型
        self.rho = model.rho
        
        # 设置采样周期
        self.t_sample = t_sample
        self.T = np.logspace(-3, 4, self.t_sample)
        
        # 用于存储地表处不同周期下的阻抗、视电阻率、相位
        self.Z_period_surface = np.zeros((t_sample, self.ny), dtype=np.complex128)
        self.pc_period_surface = np.zeros((t_sample, self.ny))
        self.ph_period_surface = np.zeros((t_sample, self.ny))
        
        # 执行正演
        self._run_forward()
        
    def _pos(self, y, z):
        return z * self.ny + y


    def _rho(self, y, z):
        # 计算原网格的 y 索引
        orig_y = 0 if y == 0 else (-1 if y == self.ny-1 else y-1)
        # 计算原网格的 z 索引
        orig_z = 0 if z == 0 else (-1 if z == self.nz-1 else z-1)
        return self.rho[orig_y, orig_z]
    
    def _solve_single_period(self, tt):
        i = 1j
        Omega = 2 * np.pi / self.T[tt]
        mu = 4 * np.pi * 1e-7
        eps = 8.8419e-12
        
        P = sp.sparse.lil_matrix((self.ny * self.nz, 1), dtype=np.complex128)
        K = sp.sparse.lil_matrix((self.ny * self.nz, self.ny * self.nz), 
            dtype=np.complex128)
        
        # 创建有限差分系数稀疏矩阵
        for z in range(self.nz):
            for y in range(self.ny):
                if y == 0:
                    # 左边界，第二类边界
                    K[self._pos(0, z), self._pos(0, z)] = -1/self.dy
                    K[self._pos(0, z), self._pos(1, z)] = 1/self.dy
                elif y == self.ny-1:
                    #右边界，第二类边界
                    K[self._pos(self.ny-1, z), self._pos(self.ny-1, z)] = 1/self.dy
                    K[self._pos(self.ny-1, z), self._pos(self.ny-2, z)] = -1/self.dy
                #上边界，第一类边界
                elif z == 0 and y != 0 and y != self.ny - 1:
                    K[self._pos(y, 0), self._pos(y, 0)] = 1/2
                    K[self._pos(y, 0), self._pos(y, 1)] = 1/2
                #下边界，第三类边界
                elif z == self.nz - 1 and y != 0 and y != self.ny - 1:
                    K[self._pos(y, self.nz-1), self._pos(y, self.nz-1)] = 1/self.dz \
                        + np.sqrt(i * Omega * mu / self._rho(y, self.nz-1) \
                        + mu * eps * Omega**2)
                    K[self._pos(y, self.nz-1), self._pos(y, self.nz-2)] = -1/self.dz
                elif (y == 1 or y == self.ny - 2) and (1 <= z <= self.nz - 2) or \
                    (z == 1 or z == self.nz - 2) and (1 <= y <= self.ny - 2):
                    K[self._pos(y, z), self._pos(y-1, z)] = 1 / self.dy**2
                    K[self._pos(y, z), self._pos(y+1, z)] = 1 / self.dy**2
                    K[self._pos(y, z), self._pos(y, z-1)] = 1 / self.dz**2
                    K[self._pos(y, z), self._pos(y, z+1)] = 1 / self.dz**2
                    K[self._pos(y, z), self._pos(y, z)] = mu * eps * Omega**2 +\
                        i * mu * Omega / self._rho(y, z) - 2/self.dy**2 - 2/self.dz**2
                else:
                    K[self._pos(y, z), self._pos(y-2, z)] = -1 / (12 * self.dy**2)
                    K[self._pos(y, z), self._pos(y+2, z)] = -1 / (12 * self.dy**2)
                    K[self._pos(y, z), self._pos(y, z-2)] = -1 / (12 * self.dz**2)
                    K[self._pos(y, z), self._pos(y, z+2)] = -1 / (12 * self.dz**2)
                    K[self._pos(y, z), self._pos(y-1, z)] = 4 / (3 * self.dy**2)
                    K[self._pos(y, z), self._pos(y+1, z)] = 4 / (3 * self.dy**2)
                    K[self._pos(y, z), self._pos(y, z-1)] = 4 / (3 * self.dz**2)
                    K[self._pos(y, z), self._pos(y, z+1)] = 4 / (3 * self.dz**2)
                    K[self._pos(y, z), self._pos(y, z)] = mu * eps * Omega**2 +\
                        i * mu * Omega / self._rho(y, z) - 5 / (2 * self.dy**2) \
                        - 5 / (2 * self.dz**2)
                    
        # 激发源设置, 直接归一化
        for y in range(1, self.ny):
            P[self._pos(y, 0)] = 1
    
        K = K.tocsr()
        P = P.toarray().flatten()
    
        Ex = sp.sparse.linalg.spsolve(K, P)
        Ex_grid = np.zeros((self.ny, self.nz), dtype=np.complex128)
        for z in range(self.nz):
            for y in range(self.ny):
                Ex_grid[y, z] = Ex[self._pos(y, z)]
        
        Ex_surface = (Ex_grid[1:-1, 1] + Ex_grid[1:-1, 0]) / 2
        Hy_surface = (Ex_grid[1:-1, 1] - Ex_grid[1:-1, 0]) / self.dz / (i * mu * Omega) 
        # Hz_surface = ((Ex_grid[2:-1, 1] + Ex_grid[2:-1, 0]) / 2 - (Ex_grid[0:-3, 1] + Ex_grid[0:-3, 0]) / 2) / (2 * dz) / (i * mu * Omega)
    
        Z_surface = Ex_surface / Hy_surface
        pc_surface = np.abs(Z_surface)**2 / (Omega * mu)
        ph_surface = -np.angle(Z_surface, deg=True)
                
        return tt, Z_surface, pc_surface, ph_surface
    
    def _run_forward(self):
        """ 并行计算所有周期的正演 """
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(self._solve_single_period, tt) for tt in range(self.t_sample)]
            for f in as_completed(futures):
                tt, Z_surface, pc_surface, ph_surface = f.result()
                self.Z_period_surface[tt, :] = Z_surface
                self.pc_period_surface[tt, :] = pc_surface
                self.ph_period_surface[tt, :] = ph_surface


model = Model(12000, 6000, 100.0, 100.0, 100, 
    anomaly_1={'top_left': (2000, 1000), 'bottom_right': (4000, 3000), 'rho': 10},
    anomaly_2={'top_left': (8000, 1000), 'bottom_right': (10000, 3000), 'rho': 1000})
    
forward = MT2DForwardFDM(model, 71)

pc = forward.pc_period_surface
ph = forward.ph_period_surface

x = np.linspace(int(100.0), 12000, int(12000 / 50.0))
y = np.logspace(-3, 4, 71)

X, Y = np.meshgrid(x, y)

fig, axes = plt.subplots(1, 2, figsize=(16, 9))
axes[0].set_title("Apparent Resistivity")
cf_pc = axes[0].contourf(X, Y, pc, levels=30, cmap='viridis')
cs_pc = axes[0].contour(X, Y, pc, levels=20, colors='black', linewidths=0.8)
axes[0].clabel(cs_pc, inline=True, fontsize=8, fmt="%.2f")
cbar_pc = fig.colorbar(cf_pc, ax=axes[0])
axes[0].set_xlabel('Position / m')
axes[0].set_ylabel('Period / s')
axes[0].set_yscale('log')
axes[0].invert_yaxis() 
axes[0].xaxis.set_ticks_position('top')
axes[0].xaxis.set_label_position('top')

axes[1].set_title("Phase")
cf_ph = axes[1].contourf(X, Y, ph, levels=30, cmap='viridis')
cs_ph = axes[1].contour(X, Y, ph, levels=20, colors='black', linewidths=0.8)
axes[1].clabel(cs_ph, inline=True, fontsize = 8, fmt="%.2f")
cbar_ph = fig.colorbar(cf_ph, ax=axes[1])
axes[1].set_xlabel('Position / m')
axes[1].set_ylabel('Period / s')
axes[1].set_yscale('log')
axes[1].invert_yaxis()
axes[1].xaxis.set_ticks_position('top')
axes[1].xaxis.set_label_position('top')

plt.show()

