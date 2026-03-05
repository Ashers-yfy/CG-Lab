import taichi as ti

# GPU初始化
ti.init(arch=ti.gpu)

# 仿真参数
N = 256         # 粒子数量
DT = 0.001      # 时间步长
G = 1.0         # 万有引力常数
EPS = 1e-3      # 防止除0