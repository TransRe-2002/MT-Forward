import matplotlib.pyplot as plt
import numpy as np
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import bicgstab, spilu, LinearOperator
from numpy.linalg import norm
from os import cpu_count

# 模型生成
class Model():
    def __init__(self, y=10000, z=10000, dy=100.0, dz=100.0, rho_background=500, **anomalies):
        self.dy = dy
        self.dz = dz
        self.y = y
        self.z = z
        self.ny = int(y / dy)
        self.nz = int(z / dz)
        self.rho = np.ones((self.ny, self.nz)) * rho_background
        
        for key, value in anomalies.items():
            self.add_anomaly(value['top_left'], value['bottom_right'], value['rho'])
    
    def add_layer(self, top, bottom, rho):
        self.add_anomaly((0, top), (self.y, bottom), rho)
    
    def add_anomaly(self, top_left, bottom_right, rho):
        (y0, z0) = top_left
        (y1, z1) = bottom_right
        rho_anomaly = rho

        iy0 = int(y0 / self.dy)
        iy1 = int(y1 / self.dy)
        iz0 = int(z0 / self.dz)
        iz1 = int(z1 / self.dz)

        iy0 = max(0, min(self.ny - 1, iy0))
        iy1 = max(0, min(self.ny - 1, iy1))
        iz0 = max(0, min(self.nz - 1, iz0))
        iz1 = max(0, min(self.nz - 1, iz1))

        self.rho[iy0 : iy1+1, iz0 : iz1+1] = rho_anomaly
        
    def print_model(self):
        y = np.linspace(int(self.dy/2), int(self.y-self.dy/2), self.ny)
        z = -np.linspace(int(self.dz/2), int(self.z-self.dz/2), self.nz)
            
        Y, Z = np.meshgrid(y, z)
            
        # Plotting
        fig, ax = plt.subplots(figsize=(16, 10))
        cp = ax.pcolormesh(Y, Z, self.rho.T, cmap='viridis', shading='auto')
        fig.colorbar(cp, label='Density (rho)')
        ax.set_title('Density Model')
        ax.set_xlabel('Y (meters)')
        ax.set_ylabel('Z (meters)')
        
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')
        
        plt.show()

class MT2DForwardFDM_TE():
    def __init__(self, model, t_sample=61, air=5000):
        # 导入模型
        self.dy = model.dy
        self.dz = model.dz
        self.y = model.y
        self.z = model.z
        self.ny_model = model.ny
        self.nz_model = model.nz
        self.rho = model.rho
        
        # 设置空气层
        self.nz_air = int(air/self.dz)
        
        # 网格上扩展空气层，下扩展一层作为边界
        # 网格左右扩展10层作为边界
        self.ny = self.ny_model + 20
        # 如果没有空气层就向上延拓一层
        self.nz = self.nz_model + max(self.nz_air, 1) + 1
        
        # 用于存储地表处不同周期下的阻抗、视电阻率、相位
        self.Z_period_surface = np.zeros((t_sample, self.ny_model), dtype=np.complex128)
        self.pc_period_surface = np.zeros((t_sample, self.ny_model))
        self.ph_period_surface = np.zeros((t_sample, self.ny_model))
        self.Tipple_period_surface = np.zeros((t_sample, self.ny_model), dtype=np.complex128)
        self.Tipple_real_period_surface = np.zeros((t_sample, self.ny_model))
        self.Tipple_imag_period_surface = np.zeros((t_sample, self.ny_model))
        self.Tipple_abs_period_surface = np.zeros((t_sample, self.ny_model))
        
        # 设置采样周期
        self.t_sample = t_sample
        self.T = np.logspace(-3, 3, self.t_sample)

        # 执行正演
        self._run_forward()
        
    def _pos(self, y, z):
        return z * self.ny + y

    def _rho(self, y, z):
        if (z < self.nz_air):
            return 1e12
        # 计算原网格的 y 索引
        orig_y = 0 if y < 10 else (-1 if y > self.ny-11 else y-10)
        # 计算原网格的 z 索引
        orig_z = 0 if z == self.nz_air else (-1 if z == self.nz-1 else z-self.nz_air-1)
        return self.rho[orig_y, orig_z]
    
    def _solve_single_period(self, tt):
        i = 1j
        Omega = 2 * np.pi / self.T[tt]
        mu = 4 * np.pi * 1e-7
        eps = 8.8419e-12
        
        P = lil_matrix((self.ny * self.nz, 1), dtype=np.complex128)
        K = lil_matrix((self.ny * self.nz, self.ny * self.nz), 
            dtype=np.complex128)
        
        # 创建有限差分系数稀疏矩阵
        # 上边界，第一类边界
        for y in range(self.ny):
            K[self._pos(y, 0), self._pos(y, 0)] = 1

        # 下边界，第三类边界
        for y in range(self.ny):
            K[self._pos(y, self.nz-1), self._pos(y, self.nz-1)] = 1 / self.dz + np.sqrt(-i * Omega * mu / self._rho(y, self.nz-1) - mu * eps * Omega**2)
            K[self._pos(y, self.nz-1), self._pos(y, self.nz-2)] = -1 / self.dz

        # 左边界，第二类边界
        for z in range(1, self.nz - 1):  # 跳过顶部和底部
            K[self._pos(0, z), self._pos(0, z)] = -1 / self.dy
            K[self._pos(0, z), self._pos(1, z)] = 1 / self.dy

        # 右边界，第二类边界
        for z in range(1, self.nz - 1):  # 跳过顶部和底部
            K[self._pos(self.ny-1, z), self._pos(self.ny-1, z)] = 1 / self.dy
            K[self._pos(self.ny-1, z), self._pos(self.ny-2, z)] = -1 / self.dy

        # 对于边界内的节点
        for z in range(1, self.nz - 1):
            for y in range(1, self.ny - 1):
                if y == 1 or y == self.ny-2 or z == 1 or z == self.nz-2:
                    K[self._pos(y, z), self._pos(y-1, z)] = 1 / self.dy**2
                    K[self._pos(y, z), self._pos(y+1, z)] = 1 / self.dy**2
                    K[self._pos(y, z), self._pos(y, z-1)] = 1 / self.dz**2
                    K[self._pos(y, z), self._pos(y, z+1)] = 1 / self.dz**2
                    K[self._pos(y, z), self._pos(y, z)] = mu * eps * Omega**2 \
                        + i * mu * Omega / self._rho(y, z) - 2 / self.dy**2 - 2 / self.dz**2

        # 对于更复杂的高阶差分处理（使用 9 点差分）
        for z in range(2, self.nz - 2):
            for y in range(2, self.ny - 2):
                K[self._pos(y, z), self._pos(y-2, z)] = -1 / (12 * self.dy**2)
                K[self._pos(y, z), self._pos(y+2, z)] = -1 / (12 * self.dy**2)
                K[self._pos(y, z), self._pos(y, z-2)] = -1 / (12 * self.dz**2)
                K[self._pos(y, z), self._pos(y, z+2)] = -1 / (12 * self.dz**2)
                K[self._pos(y, z), self._pos(y-1, z)] = 4 / (3 * self.dy**2)
                K[self._pos(y, z), self._pos(y+1, z)] = 4 / (3 * self.dy**2)
                K[self._pos(y, z), self._pos(y, z-1)] = 4 / (3 * self.dz**2)
                K[self._pos(y, z), self._pos(y, z+1)] = 4 / (3 * self.dz**2)
                K[self._pos(y, z), self._pos(y, z)] = mu * eps * Omega**2 + i \
                    * mu * Omega / self._rho(y, z) - 5 / (2 * self.dy**2) - 5 / (2 * self.dz**2)
                    
        # 激发源设置, 直接归一化
        for y in range(self.ny):
            P[self._pos(y, 0)] = 1
    
        K = K.tocsc()
        P = P.toarray()

        # 求解
        # 计算地表和半空间场
        try:
            ilu = spilu(K, fill_factor=100, drop_tol=1e-8)
            M = LinearOperator(K.shape, ilu.solve)
        except Exception as e:
            print(f"ILU 分解失败：{e}")
            exit()

        # 使用预处理的 BiCGSTAB 求解
        Ex, info = bicgstab(K, P, M=M, rtol=1e-16, maxiter=1000)
        
        if info == 0:
            residual = norm(K.dot(Ex) - P)
            print(f"收敛成功，最终残差：{residual:.2e}")
        else:
            print(f"未收敛，状态码：{info}")
        
        # 地表电场（z=0和z=1层）
        Ex_z0 = np.array([Ex[self._pos(y, max(self.nz_air, 1))] for y in range(10, self.ny-10)])
        Ex_z1 = np.array([Ex[self._pos(y, max(self.nz_air, 1)+1)] for y in range(10, self.ny-10)])
        Ex_z2 = np.array([Ex[self._pos(y, max(self.nz_air, 1)+2)] for y in range(10, self.ny-10)])
        Ex_z3 = np.array([Ex[self._pos(y, max(self.nz_air, 1)+3)] for y in range(10, self.ny-10)])
        Ex_y1 = np.array([Ex[self._pos(y, max(self.nz_air, 1))] for y in range(9, self.ny-11)])
        Ex_y2 = np.array([Ex[self._pos(y, max(self.nz_air, 1))] for y in range(11, self.ny-9)])
        
        Ex_surface = Ex_z0
        Hy_surface = (-11 * Ex_z0 + 18 * Ex_z1 - 9 * Ex_z2 + 2 * Ex_z3) / (6 * self.dz) \
            / (i * mu * Omega)
        Hz_surface = -(Ex_y2 - Ex_y1) / (2 * self.dy) / (i * mu * Omega)
        
        Z_surface = Ex_surface / Hy_surface
        pc_surface = np.abs(Z_surface)**2 / (Omega * mu)
        ph_surface = -np.angle(Z_surface, deg=True)
        Tipple_surface = Hz_surface / Hy_surface
        Tipple_real_surface = np.real(Tipple_surface)
        Tipple_imag_surface = np.imag(Tipple_surface)
        Tipple_abs_surface = np.abs(Tipple_surface)
        
        self.Z_period_surface[tt, :] = Z_surface
        self.pc_period_surface[tt, :] = pc_surface
        self.ph_period_surface[tt, :] = ph_surface
        self.Tipple_period_surface[tt, :] = Tipple_surface
        self.Tipple_real_period_surface[tt, :] = Tipple_real_surface
        self.Tipple_imag_period_surface[tt, :] = Tipple_imag_surface
        self.Tipple_abs_period_surface[tt, :] = Tipple_abs_surface
        
        # 垃圾回收
        del K, P, ilu, Ex
        gc.collect()
        
    def _run_forward(self):
        """ 并行计算所有周期的正演 """
        with ThreadPoolExecutor(max_workers=cpu_count()/4) as executor:
            futures = [executor.submit(self._solve_single_period, tt) for tt in range(self.t_sample)]
            for f in as_completed(futures):
                try:
                    f.result()
                except Exception as e:
                    print(f"某个周期计算失败: {str(e)}，请检查")
             
    def draw_pc_ph_period(self):
        x = np.linspace(int(self.dy/2), int(self.y-self.dy/2), self.ny_model)
        y = self.T
        
        X, Y = np.meshgrid(x, y)
        
        fig, axes = plt.subplots(1, 2, figsize=(20, 9))
        axes[0].set_title("Apparent Resistivity")
        cf_pc = axes[0].contourf(X, Y, self.pc_period_surface, levels=20, cmap='viridis')
        cs_pc = axes[0].contour(X, Y, self.pc_period_surface, levels=20, colors='black', linewidths=0.8)
        axes[0].clabel(cs_pc, inline=True, fontsize=8, fmt="%.2f")
        cbar_pc = fig.colorbar(cf_pc, ax=axes[0])
        axes[0].set_xlabel('Position / m')
        axes[0].set_ylabel('Period / s')
        axes[0].set_yscale('log')
        axes[0].invert_yaxis() 
        axes[0].xaxis.set_ticks_position('top')
        axes[0].xaxis.set_label_position('top')

        axes[1].set_title("Phase")
        cf_ph = axes[1].contourf(X, Y, self.ph_period_surface, levels=20, cmap='viridis')
        cs_ph = axes[1].contour(X, Y, self.ph_period_surface, levels=20, colors='black', linewidths=0.8)
        axes[1].clabel(cs_ph, inline=True, fontsize = 8, fmt="%.2f")
        cbar_ph = fig.colorbar(cf_ph, ax=axes[1])
        axes[1].set_xlabel('Position / m')
        axes[1].set_ylabel('Period / s')
        axes[1].set_yscale('log')
        axes[1].invert_yaxis()
        axes[1].xaxis.set_ticks_position('top')
        axes[1].xaxis.set_label_position('top')

        plt.show()

    def draw_tipple_period(self):
        x = np.linspace(int(self.dy/2), int(self.y-self.dy/2), self.ny_model)
        y = self.T
        
        X, Y = np.meshgrid(x, y)
        
        fig, axes = plt.subplots(1, 3, figsize=(24, 7))
        axes[0].set_title("Tipple Real")
        cf_tr = axes[0].contourf(X, Y, self.Tipple_real_period_surface, levels=20, cmap='viridis')
        cs_tr = axes[0].contour(X, Y, self.Tipple_real_period_surface, levels=20, colors='black', linewidths=0.8)
        axes[0].clabel(cs_tr, inline=True, fontsize=8, fmt="%.2f")
        cbar_tr = fig.colorbar(cf_tr, ax=axes[0])
        axes[0].set_xlabel('Position / m')
        axes[0].set_ylabel('Period / s')
        axes[0].set_yscale('log')
        axes[0].invert_yaxis() 
        axes[0].xaxis.set_ticks_position('top')
        axes[0].xaxis.set_label_position('top')

        axes[1].set_title("Tipple Imagine")
        cf_ti = axes[1].contourf(X, Y, self.Tipple_imag_period_surface, levels=20, cmap='viridis')
        cs_ti = axes[1].contour(X, Y, self.Tipple_imag_period_surface, levels=20, colors='black', linewidths=0.8)
        axes[1].clabel(cs_ti, inline=True, fontsize = 8, fmt="%.2f")
        cbar_ti = fig.colorbar(cf_ti, ax=axes[1])
        axes[1].set_xlabel('Position / m')
        axes[1].set_ylabel('Period / s')
        axes[1].set_yscale('log')
        axes[1].invert_yaxis()
        axes[1].xaxis.set_ticks_position('top')
        axes[1].xaxis.set_label_position('top')

        axes[2].set_title("Tipple Absolute Value")
        cf_ta = axes[2].contourf(X, Y, self.Tipple_abs_period_surface, levels=20, cmap='viridis')
        cs_ta = axes[2].contour(X, Y, self.Tipple_abs_period_surface, levels=20, colors='black', linewidths=0.8)
        axes[2].clabel(cs_ta, inline=True, fontsize = 8, fmt="%.2f")
        cbar_ta = fig.colorbar(cf_ta, ax=axes[2])
        axes[2].set_xlabel('Position / m')
        axes[2].set_ylabel('Period / s')
        axes[2].set_yscale('log')
        axes[2].invert_yaxis()
        axes[2].xaxis.set_ticks_position('top')
        axes[2].xaxis.set_label_position('top')
        
        plt.show()

class MT2DForwardFDM_TM():
    def __init__(self, model, t_sample=61):
        # 导入模型
        self.dy = model.dy
        self.dz = model.dz
        self.y = model.y
        self.z = model.z
        self.ny_model = model.ny
        self.nz_model = model.nz
        self.rho = model.rho
        
        # 网格左右扩展10层作为边界
        self.ny = self.ny_model + 20
        # 网格上下扩展1层作为边界
        self.nz = self.nz_model + 2
        
        # 用于存储地表处不同周期下的阻抗、视电阻率、相位
        self.Z_period_surface = np.zeros((t_sample, self.ny_model), dtype=np.complex128)
        self.pc_period_surface = np.zeros((t_sample, self.ny_model))
        self.ph_period_surface = np.zeros((t_sample, self.ny_model))

        # 设置采样周期
        self.t_sample = t_sample
        self.T = np.logspace(-3, 3, self.t_sample)

        # 执行正演
        self._run_forward()
        
    def _pos(self, y, z):
        return z * self.ny + y

    def _rho(self, y, z):
        # 计算原网格的 y 索引
        orig_y = 0 if y < 10 else (-1 if y > self.ny-11 else y-10)
        # 计算原网格的 z 索引
        orig_z = 0 if z == 0 else (-1 if z == self.nz-1 else z-1)
        return self.rho[orig_y, orig_z]
    
    def _solve_single_period(self, tt):
        i = 1j
        Omega = 2 * np.pi / self.T[tt]
        mu = 4 * np.pi * 1e-7
        eps = 8.8419e-12
        
        P = lil_matrix((self.ny * self.nz, 1), dtype=np.complex128)
        K = lil_matrix((self.ny * self.nz, self.ny * self.nz), 
            dtype=np.complex128)
        
        # 创建有限差分系数稀疏矩阵
        # 上边界，第一类边界
        for y in range(self.ny):
            K[self._pos(y, 0), self._pos(y, 0)] = 1

        # 下边界，第三类边界
        for y in range(self.ny):
            K[self._pos(y, self.nz-1), self._pos(y, self.nz-1)] = 1 / self.dz + np.sqrt(-i * Omega * mu / self._rho(y, self.nz-1) - mu * eps * Omega**2)
            K[self._pos(y, self.nz-1), self._pos(y, self.nz-2)] = -1 / self.dz

        # 左边界，第二类边界
        for z in range(1, self.nz - 1):  # 跳过顶部和底部
            K[self._pos(0, z), self._pos(0, z)] = -1 / self.dy
            K[self._pos(0, z), self._pos(1, z)] = 1 / self.dy

        # 右边界，第二类边界
        for z in range(1, self.nz - 1):  # 跳过顶部和底部
            K[self._pos(self.ny-1, z), self._pos(self.ny-1, z)] = 1 / self.dy
            K[self._pos(self.ny-1, z), self._pos(self.ny-2, z)] = -1 / self.dy

        # 对于边界内的节点
        for z in range(1, self.nz - 1):
            for y in range(1, self.ny - 1):
                if y == 1 or y == self.ny-2 or z == 1 or z == self.nz-2:
                    K[self._pos(y, z), self._pos(y-1, z)] = 1 / self.dy**2
                    K[self._pos(y, z), self._pos(y+1, z)] = 1 / self.dy**2
                    K[self._pos(y, z), self._pos(y, z-1)] = 1 / self.dz**2
                    K[self._pos(y, z), self._pos(y, z+1)] = 1 / self.dz**2
                    K[self._pos(y, z), self._pos(y, z)] = mu * eps * Omega**2 \
                        + i * mu * Omega / self._rho(y, z) - 2 / self.dy**2 - 2 / self.dz**2

        # 对于更复杂的高阶差分处理（使用 9 点差分）
        for z in range(2, self.nz - 2):
            for y in range(2, self.ny - 2):
                K[self._pos(y, z), self._pos(y-2, z)] = -1 / (12 * self.dy**2)
                K[self._pos(y, z), self._pos(y+2, z)] = -1 / (12 * self.dy**2)
                K[self._pos(y, z), self._pos(y, z-2)] = -1 / (12 * self.dz**2)
                K[self._pos(y, z), self._pos(y, z+2)] = -1 / (12 * self.dz**2)
                K[self._pos(y, z), self._pos(y-1, z)] = 4 / (3 * self.dy**2)
                K[self._pos(y, z), self._pos(y+1, z)] = 4 / (3 * self.dy**2)
                K[self._pos(y, z), self._pos(y, z-1)] = 4 / (3 * self.dz**2)
                K[self._pos(y, z), self._pos(y, z+1)] = 4 / (3 * self.dz**2)
                K[self._pos(y, z), self._pos(y, z)] = mu * eps * Omega**2 + i \
                    * mu * Omega / self._rho(y, z) - 5 / (2 * self.dy**2) - 5 / (2 * self.dz**2)
                    
        # 激发源设置, 直接归一化
        for y in range(self.ny):
            P[self._pos(y, 0)] = 1
    
        K = K.tocsr()
        P = P.toarray().flatten()
    
        # 求解
        # 计算地表和半空间场
        try:
            ilu = spilu(K, fill_factor=30, drop_tol=1e-6)
            M = LinearOperator(K.shape, ilu.solve)
        except Exception as e:
            print(f"ILU 分解失败：{e}")
            exit()

        # 使用预处理的 BiCGSTAB 求解
        Hx, info = bicgstab(K, P, M=M, rtol=1e-12, maxiter=5000)
        
        if info == 0:
            residual = norm(K.dot(Hx) - P)
            print(f"收敛成功，最终残差：{residual:.2e}")
        else:
            print(f"未收敛，状态码：{info}")
        # 地表电场（z=0和z=1层）
        Hx_z0 = np.array([Hx[self._pos(y, 0)] for y in range(10, self.ny-10)])
        Hx_z1 = np.array([Hx[self._pos(y, 1)] for y in range(10, self.ny-10)])
        Hx_z2 = np.array([Hx[self._pos(y, 2)] for y in range(10, self.ny-10)])
        Hx_z3 = np.array([Hx[self._pos(y, 3)] for y in range(10, self.ny-10)])

        
        Hx_surface = Hx_z0
        Ey_surface = (-11 * Hx_z0 + 18 * Hx_z1 - 9 * Hx_z2 + 2 * Hx_z3) / (6 * self.dz) \
            / (1 / np.array([self._rho(y, 0) for y in range(10, self.ny-10)]) - i * Omega * eps)

        Z_surface = -Ey_surface / Hx_surface
        pc_surface = np.abs(Z_surface)**2 / (Omega * mu)
        ph_surface = -np.angle(Z_surface, deg=True)

        self.Z_period_surface[tt, :] = Z_surface
        self.pc_period_surface[tt, :] = pc_surface
        self.ph_period_surface[tt, :] = ph_surface
        
        # 垃圾回收
        del K, P, ilu, Hx
        gc.collect()

    def _run_forward(self):
        """ 并行计算所有周期的正演 """
        with ThreadPoolExecutor(max_workers=cpu_count()/2) as executor:
            futures = [executor.submit(self._solve_single_period, tt) for tt in range(self.t_sample)]
            for f in as_completed(futures):
                try:
                    f.result()
                except Exception as e:
                    print(f"某个周期计算失败: {str(e)}，请检查")
             
    def draw_pc_ph_period(self):
        x = np.linspace(int(self.dy/2), int(self.y-self.dy/2), self.ny_model)
        y = self.T
        
        X, Y = np.meshgrid(x, y)
        
        fig, axes = plt.subplots(1, 2, figsize=(20, 9))
        axes[0].set_title("Apparent Resistivity")
        cf_pc = axes[0].contourf(X, Y, self.pc_period_surface, levels=20, cmap='viridis')
        cs_pc = axes[0].contour(X, Y, self.pc_period_surface, levels=20, colors='black', linewidths=0.8)
        axes[0].clabel(cs_pc, inline=True, fontsize=8, fmt="%.2f")
        cbar_pc = fig.colorbar(cf_pc, ax=axes[0])
        axes[0].set_xlabel('Position / m')
        axes[0].set_ylabel('Period / s')
        axes[0].set_yscale('log')
        axes[0].invert_yaxis() 
        axes[0].xaxis.set_ticks_position('top')
        axes[0].xaxis.set_label_position('top')

        axes[1].set_title("Phase")
        cf_ph = axes[1].contourf(X, Y, self.ph_period_surface, levels=20, cmap='viridis')
        cs_ph = axes[1].contour(X, Y, self.ph_period_surface, levels=20, colors='black', linewidths=0.8)
        axes[1].clabel(cs_ph, inline=True, fontsize = 8, fmt="%.2f")
        cbar_ph = fig.colorbar(cf_ph, ax=axes[1])
        axes[1].set_xlabel('Position / m')
        axes[1].set_ylabel('Period / s')
        axes[1].set_yscale('log')
        axes[1].invert_yaxis()
        axes[1].xaxis.set_ticks_position('top')
        axes[1].xaxis.set_label_position('top')

        plt.show()

if __name__ == '__main__':
    model1 = Model(12000, 6000, 50.0, 50.0, 100)
    
    model1.add_anomaly((2000, 1000), (4000, 3000), 10)
    model1.add_anomaly((8000, 1000), (10000, 3000), 1000)

    model1.print_model()
    # TE 模式
    forward1_1 = MT2DForwardFDM_TE(model1, 61)
    forward1_1.draw_pc_ph_period()
    forward1_1.draw_tipple_period()
    
    # TM 模式
    # forward1_2 = MT2DForwardFDM_TM(model1, 71)
    # forward1_2.draw_pc_ph_period()

    # model2 = Model(8000, 4000, 100.0, 100.0, 100)
    # model2.add_anomaly((3000, 1000), (5000, 3000), 10)
    # model2.print_model()
    
    # # 有空气层
    # forward2_1 = MT2DForwardFDM(model2, 51)
    # forward2_1.draw_pc_ph_period()
    # forward2_1.draw_tipple_period()
    
    # # 无空气层
    # forward2_2 = MT2DForwardFDM(model2, 51, 0)
    # forward2_2.draw_pc_ph_period()
    # forward2_2.draw_tipple_period()
    
    # model3 = Model(1000, 5000, 100.0, 50.0, 100)
    # model3.add_layer(0, 1000, 10)
    # model3.print_model()
    # forward3 = MT2DForwardFDM(model3, 51)
    # forward3.draw_pc_ph_period()
    # forward3.draw_tipple_period()
    
    
    # model4 = Model(8000, 8000, 100.0, 100.0, 100)
    # model4.add_anomaly((3000, 5000), (5000, 7000), 10)
    # model4.print_model()
    
    # forward4 = MT2DForwardFDM_TE(model4, 51)
    # forward4.draw_pc_ph_period()
    # forward4.draw_tipple_period()