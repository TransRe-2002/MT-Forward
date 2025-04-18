import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve


def generate_model(rho, h, DZ):
    rho_node = []
    for i in range(np.size(h)):
        nodes = int(h[i] / DZ / 3)
        rho_node += [rho[i]] * nodes
    
    last = max(0, 3 - np.size(rho_node) % 3) + 1
    rho_node += [rho[-1]] * last
    
    rho_node = np.array(rho_node)
    NZ = int((np.size(rho_node) - 1) / 3)
    return rho_node, NZ, DZ

def mt1d_forward_fem(rho, NZ, DZ):
    NE = NZ
    NP = 1 + NE * 3
    L = DZ if isinstance(DZ, np.ndarray) else DZ * np.ones(NE)

    ME = np.zeros((4, NE), dtype=int)
    for i in range(NE):
        ME[0, i] = 1 + i * 3 - 1
        ME[1, i] = 2 + i * 3 - 1
        ME[2, i] = 3 + i * 3 - 1
        ME[3, i] = 4 + i * 3 - 1

    f = np.logspace(-3, 4, 71)
    NP = int(NP)
    pc = np.zeros_like(f)
    ph = np.zeros_like(f)

    for ff in range(len(f)):
        K1 = lil_matrix((NP, NP), dtype=complex)
        K2 = lil_matrix((NP, NP), dtype=complex)
        K3 = lil_matrix((NP, NP), dtype=complex)
        P = np.zeros(NP, dtype=complex)

        for h in range(NE):
            l = L[h]
            K = np.array([
                [148, -189,  54, -13],
                [-189, 432, -297, 54],
                [54, -297, 432, -189],
                [-13, 54, -189, 148]
            ]) / (40 * l)

            for j in range(4):
                NJ = ME[j, h]
                for k in range(4):
                    NK = ME[k, h]
                    K1[NJ, NK] += K[j, k]

        mu = 4e-7 * np.pi
        w = 2 * np.pi * f[ff]
        m = 1j * w * mu

        for h in range(NE):
            l = L[h]
            idx = np.array([ME[0, h], ME[1, h], ME[2, h], ME[3, h]])
            r = 1 / rho[idx]
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
                NJ = ME[j, h]
                for k in range(4):
                    NK = ME[k, h]
                    K2[NJ, NK] += K[j, k] * m * l / 6720

        a = np.sqrt(-m / rho[-1])
        K3[-1, -1] = a

        v = K1 - K2 + K3
        v = v.tocsr()
        v[-1, :] = 0
        v[-1, -2] = -1 / DZ
        v[-1, -1] = 1 / DZ - i * np.sqrt(m / rho[-1])

        DJ = 0
        v[DJ, DJ] *= 1e10
        P[DJ] = v[DJ, DJ] * 1

        Ex = spsolve(v, P)
        DE = (-11 * Ex[DJ] + 18 * Ex[DJ + 1] - 9 * Ex[DJ + 2] + 2 * Ex[DJ + 3]) / L[0] / 2
        Hy = DE / m
        Z = Ex[DJ] / Hy
        pc[ff] = np.abs(Z)**2 / (w * mu)
        ph[ff] = -np.angle(Z, deg=True)

    return pc, ph

if __name__ == '__main__':
    rho = [100, 50, 10, 50, 30, 15, 100]       # 各层电阻率
    h = [450, 700, 650, 400, 1850, 3500]                 # 各层厚度
    t_sample = 71                                # 周期样本数
    DZ = 30.0                                           # 网格间距
    
    rho_node, NZ, DZ = generate_model(rho, h, DZ)
    pc, ph = mt1d_forward_fem(rho_node, NZ, DZ)
    
    fig, axes = plt.subplots(2, 1, figsize=(16, 9))
    x_impedance = np.logspace(-3, 4, 71)
    axes[0].set_title("Apparent Resistivity")
    axes[0].semilogx(x_impedance, pc, 'b-')
    axes[1].set_title("Phase")
    axes[1].semilogx(x_impedance, ph, 'r-')
    plt.show()
    