import numpy as np
import matplotlib.pyplot as plt 

def mt1d_forward(rho, h, t_sample=71):
    """解析法的一维大地电磁正演"""
    # rho:每一层的电阻率
    # h:每一层的高度
    mu = (4e-7) * np.pi
    T = np.logspace(-3, 4, t_sample)
    Omega = 2 * np.pi / T
    i = complex(0, 1)
    
    k = np.zeros((len(rho), len(T)), dtype=np.complex128)
    
    for n in range(len(rho)):
        # k[n, :] = np.sqrt(-i * 2 * np.pi * mu / (T * rho[n]))
        k[n, :] = np.sqrt(-i * 2 * np.pi * mu / (T * rho[n]) - mu * 8.8419e-012 * Omega ** 2) # 位移电流
    
    m = len(rho) - 1
    y = -(i * mu * 2 * np.pi) / (T * k[m, :])             # 底层阻抗
    
    for nn in range(m-1, -1, -1):
        A = -(i * 2 * np.pi * mu) / (T * k[nn, :])
        B = np.exp(-2 * k[nn, :] * h[nn])
        y = A * (A * (1 - B) + y * (1 + B)) / (A * (1 + B) + y * (1 - B))
    
    pc = (T / (mu * 2 * np.pi)) * (np.abs(y) ** 2)        # 视电阻率
    ph = -np.angle(y, deg=True)            # 相位
    
    return pc, ph

rho = [1000, 100, 1e6]       # 各层电阻率
h = [500, 500]                 # 各层厚度
t_sample = 71                                # 周期样本数

pc, ph = mt1d_forward(rho, h, t_sample)

fig, axes = plt.subplots(2, 1, figsize=(16, 9))

axes[0].set_title("Impedance")
x_impedance = np.logspace(-3, 4, t_sample)
axes[0].semilogx(x_impedance, pc, 'b-')


axes[1].set_title("Phase")
x_phase = np.logspace(-3, 4, t_sample)
axes[1].semilogx(x_phase, ph, 'r-')

# 显示图形
plt.show()
        
        
        
        
        
        
        
    
    
