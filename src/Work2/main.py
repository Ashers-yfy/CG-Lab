import taichi as ti
import numpy as np

ti.init(arch=ti.cpu)

# ──────────────────────────────────────────────
# 常量
# ──────────────────────────────────────────────
WIDTH, HEIGHT = 800, 800
NUM_SEGMENTS = 1000
MAX_CONTROL_POINTS = 100

# ──────────────────────────────────────────────
# GPU 缓冲区
# ──────────────────────────────────────────────
pixels             = ti.Vector.field(3, dtype=ti.f32, shape=(WIDTH, HEIGHT))
curve_points_field = ti.Vector.field(2, dtype=ti.f32, shape=(NUM_SEGMENTS + 1))
gui_points         = ti.Vector.field(2, dtype=ti.f32, shape=MAX_CONTROL_POINTS)

# ──────────────────────────────────────────────
# CPU：De Casteljau 算法（基础任务）
# ──────────────────────────────────────────────
def de_casteljau(points, t):
    pts = [np.array(p, dtype=float) for p in points]
    while len(pts) > 1:
        pts = [(1 - t) * pts[i] + t * pts[i + 1] for i in range(len(pts) - 1)]
    return pts[0]

# ──────────────────────────────────────────────
# CPU：均匀三次 B 样条（选做2）
# ──────────────────────────────────────────────
B_MAT = np.array([
    [-1,  3, -3,  1],
    [ 3, -6,  3,  0],
    [-3,  0,  3,  0],
    [ 1,  4,  1,  0],
], dtype=float) / 6.0

def bspline_point(p0, p1, p2, p3, t):
    T = np.array([t**3, t**2, t, 1.0])
    P = np.array([p0, p1, p2, p3])
    return T @ B_MAT @ P

def compute_bspline(ctrl_pts, n_samples):
    n = len(ctrl_pts)
    if n < 4:
        return []
    n_segs = n - 3
    sps = max(1, n_samples // n_segs)
    pts = []
    for i in range(n_segs):
        p0, p1, p2, p3 = ctrl_pts[i], ctrl_pts[i+1], ctrl_pts[i+2], ctrl_pts[i+3]
        end = sps + 1 if i == n_segs - 1 else sps
        for j in range(end):
            pts.append(bspline_point(p0, p1, p2, p3, j / sps))
    return pts

# ──────────────────────────────────────────────
# GPU Kernels
# ──────────────────────────────────────────────
@ti.kernel
def clear_pixels():
    for i, j in pixels:
        pixels[i, j] = ti.Vector([0.05, 0.05, 0.10])

@ti.kernel
def draw_curve_basic(n: ti.i32, r: ti.f32, g: ti.f32, b: ti.f32):
    """基础版：单像素绘制"""
    for i in range(n):
        x = int(curve_points_field[i][0] * WIDTH)
        y = int(curve_points_field[i][1] * HEIGHT)
        if 0 <= x < WIDTH and 0 <= y < HEIGHT:
            pixels[x, y] = ti.Vector([r, g, b])

@ti.kernel
def draw_curve_aa(n: ti.i32, r: ti.f32, g: ti.f32, b: ti.f32):
    """选做1：反走样，3×3 邻域高斯加权"""
    for i in range(n):
        fx = curve_points_field[i][0] * WIDTH
        fy = curve_points_field[i][1] * HEIGHT
        cx = int(fx)
        cy = int(fy)
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                nx = cx + dx
                ny = cy + dy
                if 0 <= nx < WIDTH and 0 <= ny < HEIGHT:
                    dist2 = (float(nx) + 0.5 - fx)**2 + (float(ny) + 0.5 - fy)**2
                    w = ti.exp(-dist2 / (2.0 * 0.64))
                    old = pixels[nx, ny]
                    pixels[nx, ny] = ti.Vector([
                        ti.min(old[0] + r * w, 1.0),
                        ti.min(old[1] + g * w, 1.0),
                        ti.min(old[2] + b * w, 1.0),
                    ])

# ──────────────────────────────────────────────
# 主程序（使用 ti.GUI，兼容性好）
# ──────────────────────────────────────────────
gui = ti.GUI("Bezier & B-Spline", res=(WIDTH, HEIGHT), background_color=0x0D0D1A)

ctrl_pts = []
mode     = "bezier"   # "bezier" / "bspline"
aa_mode  = True

print("=== 操作说明 ===")
print("  鼠标左键 : 添加控制点")
print("  C        : 清空所有控制点")
print("  B        : 切换 贝塞尔 / B样条")
print("  A        : 切换反走样 开/关")
print("  ESC      : 退出")

clear_pixels()

while gui.running:
    # ── 事件处理 ──
    for e in gui.get_events(ti.GUI.PRESS):
        if e.key == ti.GUI.ESCAPE:
            gui.running = False
        elif e.key == ti.GUI.LMB:
            if len(ctrl_pts) < MAX_CONTROL_POINTS:
                ctrl_pts.append(list(e.pos))
        elif e.key == 'c':
            ctrl_pts.clear()
            clear_pixels()
        elif e.key == 'b':
            mode = "bspline" if mode == "bezier" else "bezier"
            clear_pixels()
        elif e.key == 'a':
            aa_mode = not aa_mode

    # ── 清屏 ──
    clear_pixels()

    # ── 计算并绘制曲线 ──
    curve_color = (0.2, 1.0, 0.3) if mode == "bezier" else (0.3, 0.7, 1.0)

    if mode == "bezier" and len(ctrl_pts) >= 2:
        arr = np.zeros((NUM_SEGMENTS + 1, 2), dtype=np.float32)
        for i in range(NUM_SEGMENTS + 1):
            arr[i] = de_casteljau(ctrl_pts, i / NUM_SEGMENTS)
        curve_points_field.from_numpy(arr)
        if aa_mode:
            draw_curve_aa(NUM_SEGMENTS + 1, *curve_color)
        else:
            draw_curve_basic(NUM_SEGMENTS + 1, *curve_color)

    elif mode == "bspline" and len(ctrl_pts) >= 4:
        bpts = compute_bspline(ctrl_pts, NUM_SEGMENTS)
        if bpts:
            n = min(len(bpts), NUM_SEGMENTS + 1)
            arr = np.zeros((NUM_SEGMENTS + 1, 2), dtype=np.float32)
            for i in range(n):
                arr[i] = bpts[i]
            curve_points_field.from_numpy(arr)
            if aa_mode:
                draw_curve_aa(n, *curve_color)
            else:
                draw_curve_basic(n, *curve_color)

    # ── 显示像素缓冲 ──
    gui.set_image(pixels)

    # ── 控制多边形（灰线） ──
    if len(ctrl_pts) >= 2:
        pts_arr = np.array(ctrl_pts, dtype=np.float32)
        for i in range(len(ctrl_pts) - 1):
            gui.line(pts_arr[i], pts_arr[i + 1], radius=1, color=0x888888)

    # ── 控制点（红色圆点） ──
    for p in ctrl_pts:
        gui.circle(p, radius=6, color=0xFF4444)

    # ── HUD 文字 ──
    mode_str = "Bezier" if mode == "bezier" else "B-Spline"
    aa_str   = "ON" if aa_mode else "OFF"
    gui.text(
        f"Mode: {mode_str} [B]   AA: {aa_str} [A]   Points: {len(ctrl_pts)}   C=clear  ESC=quit",
        pos=(0.01, 0.97),
        color=0xFFFFFF,
        font_size=18,
    )

    gui.show()