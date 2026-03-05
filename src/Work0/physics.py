import taichi as ti
from .config import N, DT, G, EPS

# 粒子属性
pos = ti.Vector.field(2, dtype=ti.f32, shape=N)
vel = ti.Vector.field(2, dtype=ti.f32, shape=N)
force = ti.Vector.field(2, dtype=ti.f32, shape=N)

@ti.kernel
def initialize():
    for i in range(N):
        pos[i] = ti.Vector([ti.random(), ti.random()])
        vel[i] = ti.Vector([0.0, 0.0])

@ti.kernel
def compute_gravity(mouse: ti.types.vector(2, ti.f32)):
    for i in range(N):
        f = ti.Vector([0.0, 0.0])
        # 粒子间万有引力
        for j in range(N):
            if i != j:
                r = pos[j] - pos[i]
                dist = r.norm() + EPS
                f += G * r / dist**3
        # 鼠标吸引力
        r_mouse = mouse - pos[i]
        dist_mouse = r_mouse.norm() + EPS
        f += 5.0 * r_mouse / dist_mouse**2
        force[i] = f

@ti.kernel
def update():
    for i in range(N):
        vel[i] += force[i] * DT
        pos[i] += vel[i] * DT