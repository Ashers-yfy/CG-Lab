import taichi as ti
import pygame
import numpy as np
import sys

ti.init(arch=ti.cpu)

# ─────────────────────────────────────────────
# 参数
# ─────────────────────────────────────────────
N         = 20
mass      = 1.0
dt        = 5e-4
k_s       = 10000.0
k_d       = 1.0
gravity   = ti.Vector([0.0, -9.8, 0.0])
max_vel   = 50.0

max_springs = N * N * 4

# ─────────────────────────────────────────────
# Taichi Fields
# ─────────────────────────────────────────────
x          = ti.Vector.field(3, dtype=float, shape=N * N)
v          = ti.Vector.field(3, dtype=float, shape=N * N)
f          = ti.Vector.field(3, dtype=float, shape=N * N)
is_fixed   = ti.field(dtype=int,   shape=N * N)

x_next     = ti.Vector.field(3, dtype=float, shape=N * N)
v_next     = ti.Vector.field(3, dtype=float, shape=N * N)
f_next     = ti.Vector.field(3, dtype=float, shape=N * N)

spring_pairs   = ti.Vector.field(2, dtype=int,   shape=max_springs)
spring_lengths = ti.field(dtype=float, shape=max_springs)
num_springs    = ti.field(dtype=int,   shape=())

# numpy buffer 用于读取位置
pos_np = np.zeros((N * N, 3), dtype=np.float32)

# ─────────────────────────────────────────────
# 初始化
# ─────────────────────────────────────────────
@ti.kernel
def init_positions():
    for i, j in ti.ndrange(N, N):
        idx = i * N + j
        x[idx] = ti.Vector([i * 0.05 - 0.5, 0.8, j * 0.05 - 0.5])
        v[idx] = ti.Vector([0.0, 0.0, 0.0])
        f[idx] = ti.Vector([0.0, 0.0, 0.0])
        is_fixed[idx] = 1 if (j == 0 and (i == 0 or i == N - 1)) else 0

@ti.kernel
def init_springs():
    for i, j in ti.ndrange(N, N):
        idx = i * N + j
        if i < N - 1:
            nb = (i + 1) * N + j
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c]   = ti.Vector([idx, nb])
            spring_lengths[c] = (x[idx] - x[nb]).norm()
        if j < N - 1:
            nb = i * N + (j + 1)
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c]   = ti.Vector([idx, nb])
            spring_lengths[c] = (x[idx] - x[nb]).norm()

def init_cloth():
    num_springs[None] = 0
    init_positions()
    init_springs()

# ─────────────────────────────────────────────
# 力计算 & 钳制（ti.func 内联）
# ─────────────────────────────────────────────
@ti.func
def compute_forces_on(pos: ti.template(), vel: ti.template(), force: ti.template(), kd: float):
    for i in range(N * N):
        force[i] = gravity * mass - kd * vel[i]
    for i in range(num_springs[None]):
        a  = spring_pairs[i][0]
        b  = spring_pairs[i][1]
        d  = pos[a] - pos[b]
        dn = d.norm()
        if dn > 1e-6:
            fs = -k_s * (dn - spring_lengths[i]) * (d / dn)
            ti.atomic_add(force[a],  fs)
            ti.atomic_add(force[b], -fs)

@ti.func
def clamp_vel(vel: ti.template(), i: int):
    spd = vel[i].norm()
    if spd > max_vel:
        vel[i] = vel[i] / spd * max_vel

# ─────────────────────────────────────────────
# 积分器
# ─────────────────────────────────────────────
@ti.kernel
def step_explicit(kd: float):
    compute_forces_on(x, v, f, kd)
    for i in range(N * N):
        if is_fixed[i] == 0:
            x[i] += v[i] * dt
            v[i] += (f[i] / mass) * dt
            clamp_vel(v, i)

@ti.kernel
def step_semi_implicit(kd: float):
    compute_forces_on(x, v, f, kd)
    for i in range(N * N):
        if is_fixed[i] == 0:
            v[i] += (f[i] / mass) * dt
            clamp_vel(v, i)
            x[i] += v[i] * dt

@ti.kernel
def step_implicit_iter(kd: float):
    for i in range(N * N):
        v_next[i] = v[i]
        x_next[i] = x[i]
    for _ in ti.static(range(3)):
        compute_forces_on(x_next, v_next, f_next, kd)
        for i in range(N * N):
            if is_fixed[i] == 0:
                v_next[i] = v[i] + (f_next[i] / mass) * dt
                clamp_vel(v_next, i)
                x_next[i] = x[i] + v_next[i] * dt
    for i in range(N * N):
        v[i] = v_next[i]
        x[i] = x_next[i]

# ─────────────────────────────────────────────
# 2D 投影工具
# ─────────────────────────────────────────────
W, H = 900, 750

def project(p3, cam_dist=3.0):
    """简单正交投影：x→屏幕x，y→屏幕y，忽略z"""
    sx = int(p3[0] * 400 + W // 2)
    sy = int(-p3[1] * 400 + H // 2 + 80)
    return sx, sy

# ─────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────
def main():
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Mass-Spring Cloth Simulation")
    clock  = pygame.time.Clock()
    font   = pygame.font.Font(None, 22)
    font_b = pygame.font.Font(None, 24)

    init_cloth()

    solver   = 1        # 0=显式 1=半隐式 2=隐式
    paused   = False
    kd_val   = 1.0
    solver_names = ["Explicit Euler", "Semi-Implicit Euler", "Implicit Euler"]
    solver_colors = [(220,80,80), (80,200,80), (80,150,255)]

    # 按钮区域
    BTN_W, BTN_H = 210, 32
    btn_rects = [
        pygame.Rect(20, 60 + i * 42, BTN_W, BTN_H) for i in range(3)
    ]
    btn_pause = pygame.Rect(20, 210, BTN_W, BTN_H)
    btn_reset = pygame.Rect(20, 255, BTN_W, BTN_H)
    btn_kd_up   = pygame.Rect(170, 310, 30, 26)
    btn_kd_down = pygame.Rect(20,  310, 30, 26)

    def draw_button(rect, label, active=False, color=None):
        c = color if color else ((60,120,200) if active else (50,50,70))
        pygame.draw.rect(screen, c, rect, border_radius=6)
        pygame.draw.rect(screen, (120,120,160), rect, 1, border_radius=6)
        txt = font.render(label, True, (230,230,230))
        screen.blit(txt, (rect.x + 8, rect.y + 8))

    while True:
        clock.tick(60)
        mx, my = pygame.mouse.get_pos()
        clicked = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                clicked = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                if event.key == pygame.K_r:
                    init_cloth()
                if event.key == pygame.K_1: solver = 0; init_cloth()
                if event.key == pygame.K_2: solver = 1; init_cloth()
                if event.key == pygame.K_3: solver = 2; init_cloth()

        if clicked:
            for i, r in enumerate(btn_rects):
                if r.collidepoint(mx, my):
                    solver = i; init_cloth()
            if btn_pause.collidepoint(mx, my):
                paused = not paused
            if btn_reset.collidepoint(mx, my):
                init_cloth()
            if btn_kd_up.collidepoint(mx, my):
                kd_val = min(kd_val + 0.5, 10.0)
            if btn_kd_down.collidepoint(mx, my):
                kd_val = max(kd_val - 0.5, 0.1)

        # ── 物理更新 ──
        if not paused:
            for _ in range(40):
                if solver == 0:   step_explicit(kd_val)
                elif solver == 1: step_semi_implicit(kd_val)
                else:             step_implicit_iter(kd_val)

        # ── 读取位置到 numpy ──
        pos_np[:] = x.to_numpy()

        # ── 绘制 ──
        screen.fill((18, 18, 28))

        # 弹簧线
        ns = num_springs[None]
        sp = spring_pairs.to_numpy()
        for k in range(ns):
            a, b = sp[k]
            pa = project(pos_np[a])
            pb = project(pos_np[b])
            pygame.draw.line(screen, (80, 100, 160), pa, pb, 1)

        # 质点
        for i in range(N * N):
            p = project(pos_np[i])
            color = (255, 220, 80) if is_fixed[i] else (80, 180, 255)
            pygame.draw.circle(screen, color, p, 3)

        # ── GUI 面板 ──
        pygame.draw.rect(screen, (28, 28, 42), (10, 10, 240, 360), border_radius=10)
        pygame.draw.rect(screen, (60,60,100),  (10, 10, 240, 360), 1, border_radius=10)

        screen.blit(font_b.render("Solver", True, (200,200,220)), (20, 20))
        for i, r in enumerate(btn_rects):
            active = (i == solver)
            draw_button(r, ("▶ " if active else "  ") + solver_names[i],
                        active, solver_colors[i] if active else None)

        screen.blit(font_b.render("Controls", True, (200,200,220)), (20, 192))
        draw_button(btn_pause, "⏸ Pause" if not paused else "▶ Resume")
        draw_button(btn_reset, "↺ Reset  [R]")

        screen.blit(font_b.render(f"Damping kd: {kd_val:.1f}", True, (200,200,220)), (20, 292))
        draw_button(btn_kd_down, " -", color=(60,60,90))
        draw_button(btn_kd_up,   " +", color=(60,60,90))

        # 状态栏
        status = "PAUSED" if paused else "RUNNING"
        col    = (200,180,60) if paused else (80,220,80)
        screen.blit(font.render(f"[{status}]  Solver: {solver_names[solver]}  kd={kd_val:.1f}", True, col),
                    (20, H - 28))
        screen.blit(font.render("Keys: 1/2/3=solver  Space=pause  R=reset", True, (100,100,130)),
                    (20, H - 48))

        pygame.display.flip()

if __name__ == "__main__":
    main()