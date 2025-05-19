import matplotlib.pyplot as plt
from matplotlib import colormaps, colors

import numpy as np
import scipy as sp
from scipy.sparse import lil_matrix, csc_matrix
from scipy.sparse.linalg import spsolve, bicgstab, spilu, LinearOperator
from numpy.linalg import norm
from os import cpu_count

class Model:
    def __init__(self, x=10000.0, y=10000.0, z=10000.0,
                 dx=1000.0, dy=1000.0, dz=1000.0, 
                 rho_background=500.0, *anomalies):
        self.x = x
        self.y = y
        self.z = z
        self.nx = int(x / dx) + 1
        self.ny = int(y / dy) + 1
        self.nz = int(z / dz) + 1
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.rho_background = rho_background
        
        self.rho = np.ones((self.nx, self.ny, self.nz)) * rho_background
        for ano in anomalies:
            self.add_anomaly(ano[0], ano[1], ano[2], ano[3], 
                             ano[4], ano[5], ano[6])

    def add_anomaly(self, x_front, x_rear, y_left, y_right, 
                    z_top, z_bottom, rho):
        ix0 = int(x_front / self.dx)
        ix1 = int(x_rear / self.dx) 
        iy0 = int(y_left / self.dy)
        iy1 = int(y_right / self.dy)
        iz0 = int(z_top / self.dz)
        iz1 = int(z_bottom / self.dz)
        
        ix0 = max(0, min(self.nx, ix0))
        ix1 = max(0, min(self.nx, ix1))
        iy0 = max(0, min(self.ny, iy0))
        iy1 = max(0, min(self.ny, iy1))
        iz0 = max(0, min(self.nz, iz0))
        iz1 = max(0, min(self.nz, iz1))
        
        self.rho[ix0: ix1+1, iy0: iy1+1, iz0: iz1+1] = rho
        
    def print_model_voxel(self):
        x = np.linspace(0, self.x, self.nx)
        y = np.linspace(0, self.y, self.ny)
        z = -np.linspace(0, self.z, self.nz)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        filled = np.full((self.nx-1, self.ny-1, self.nz-1), True)
        filled[self.nx//2+1:, self.ny//2+1:, :] = False
        
        norm = colors.Normalize(vmin=self.rho.min(), vmax=self.rho.max())
        cmap = colormaps['viridis']
        rgba = cmap(norm(self.rho[:self.nx-1, :self.ny-1, :self.nz-1]))
        
        fig = plt.figure(figsize=(16, 10))
        fig.set_size_inches(16, 10, forward=True)
        ax = fig.add_subplot(111, projection='3d')
        
        ax.voxels(
            X, Y, Z,
            filled, 
            edgecolor='none', 
            facecolors=rgba, 
            alpha=1, antialiased=True
            )
        
        ax.set_box_aspect((self.nx, self.ny, self.nz))
        # 设置标签和视角
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.view_init(elev=30, azim=30)
        
        # 4. 添加 colorbar
        mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array(self.rho)  # 传入全模型数据数组即可 :contentReference[oaicite:0]{index=0}
        cbar = fig.colorbar(mappable, ax=ax, shrink=0.6, pad=0.1)
        cbar.set_label('Resistivity (Ω·m)')  # 根据需要设置标签 :contentReference[oaicite:1]{index=1}
        
        plt.show()

    def print_model_slice(self):
        x = np.linspace(0, self.x, self.nx)
        y = np.linspace(0, self.y, self.ny)
        z = -np.linspace(0, self.z, self.nz)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')


        
        
if __name__ == '__main__':
    model = Model(6000, 6000, 3000, 250, 250, 250, 100)
    model.add_anomaly(2500, 3500, 2000, 4000, 250, 2250, 0.5)
    model.print_model_voxel()
