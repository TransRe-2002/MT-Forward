import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy.sparse import lil_matrix, csc_matrix
from scipy.sparse.linalg import spsolve, bicgstab, spilu, LinearOperator
from numpy.linalg import norm
from os import cpu_count

class Model:
    def __init__(self, x=10000, y=10000, z=10000, dx=1000, dy=1000, dz=1000, rho_background=500, **anomalies):
        self.x = x
        self.y = y
        self.z = z
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.nx = int(x / dx)
        self.ny = int(y / dy)
        self.nz = int(z / dz)
        self.rho = np.ones((self.nx, self.ny, self.nz)) * rho_background

    def add_anomaly(self, x_front, x_back, y_left, y_right, z_up, z_down, rho):
        pass

