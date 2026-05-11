import taichi as ti
import numpy as np

ti.init(arch=ti.cpu)

# # 基础任务：三角形顶点

triangle_vertices = [
    [2.0,  0.0, -2.0],
    [0.0,  2.0, -2.0],
    [-2.0, 0.0, -2.0],
]
triangle_colors = [
    (1.0, 0.0, 0.0),   # 红
    (0.0, 1.0, 0.0),   # 绿
    (0.0, 0.0, 1.0),   # 蓝
]


# 选做：
cube_vertices = np.array([
    [-1, -1, -1],   # 0
    [ 1, -1, -1],   # 1
    [ 1,  1, -1],   # 2
    [-1,  1, -1],   # 3
    [-1, -1,  1],   # 4
    [ 1, -1,  1],   # 5
    [ 1,  1,  1],   # 6
    [-1,  1,  1],   # 7
], dtype=float)

cube_edges = [
    (0,1),(1,2),(2,3),(3,0),   # 后面
    (4,5),(5,6),(6,7),(7,4),   # 前面
    (0,4),(1,5),(2,6),(3,7),   # 连接边
]

# 变换矩阵函数

def get_model_matrix(angle_deg: float) -> np.ndarray:
    """绕 Z 轴旋转 angle_deg 度的模型变换矩阵"""
    rad = angle_deg * np.pi / 180.0
    c, s = np.cos(rad), np.sin(rad)
    return np.array([
        [ c, -s,  0,  0],
        [ s,  c,  0,  0],
        [ 0,  0,  1,  0],
        [ 0,  0,  0,  1],
    ], dtype=float)


def get_view_matrix(eye_pos: np.ndarray) -> np.ndarray:
    """将相机平移至原点的视图变换矩阵（相机朝向 -Z，无旋转）"""
    x, y, z = eye_pos
    return np.array([
        [1, 0, 0, -x],
        [0, 1, 0, -y],
        [0, 0, 1, -z],
        [0, 0, 0,  1],
    ], dtype=float)


def get_projection_matrix(eye_fov_deg: float, aspect_ratio: float,
                           zNear: float, zFar: float) -> np.ndarray:
    """
    透视投影矩阵
    eye_fov_deg : Y 方向视场角（角度制）
    aspect_ratio: 宽/高
    zNear, zFar : 正值距离；内部转为 n=-zNear, f=-zFar（右手坐标系）
    """
    fov = eye_fov_deg * np.pi / 180.0
    n = -zNear   # 近截面 z 坐标（负值）
    f = -zFar    # 远截面 z 坐标（负值）

    # 视锥体边界
    t = np.tan(fov / 2.0) * abs(n)
    b = -t
    r = aspect_ratio * t
    l = -r

    # 1. 透视 → 正交挤压矩阵
    M_persp2ortho = np.array([
        [n,   0,    0,      0  ],
        [0,   n,    0,      0  ],
        [0,   0,  n + f,  -n*f ],
        [0,   0,    1,      0  ],
    ], dtype=float)

    # 2. 正交投影矩阵（平移 + 缩放到 [-1,1]^3）
    M_ortho = np.array([
        [2/(r-l),    0,       0,      -(r+l)/(r-l)],
        [0,       2/(t-b),    0,      -(t+b)/(t-b)],
        [0,          0,    2/(n-f),   -(n+f)/(n-f)],
        [0,          0,       0,           1       ],
    ], dtype=float)

    return M_ortho @ M_persp2ortho


def apply_mvp(vertices: np.ndarray, mvp: np.ndarray) -> np.ndarray:
    """
    将 N×3 的顶点数组经 MVP 变换后做透视除法，返回 N×2 屏幕坐标（NDC 中 [-1,1]）。
    """
    n = vertices.shape[0]
    # 扩展为齐次坐标 N×4
    ones = np.ones((n, 1))
    v_h = np.hstack([vertices, ones])          # N×4

    # MVP 变换（列向量约定：MVP @ v^T，再转置回来）
    v_clip = (mvp @ v_h.T).T                   # N×4

    # 透视除法
    w = v_clip[:, 3:4]
    v_ndc = v_clip[:, :3] / w                  # N×3，x,y in [-1,1]

    return v_ndc[:, :2]                        # 只返回 x,y


def ndc_to_screen(ndc_pts: np.ndarray) -> np.ndarray:
    """NDC [-1,1] → 屏幕 [0,1]（Taichi GUI 用 0~1 坐标）"""
    return (ndc_pts + 1.0) / 2.0



# GUI 参数

WIDTH, HEIGHT = 700, 700
gui = ti.GUI("MVP 旋转与变换", res=(WIDTH, HEIGHT), background_color=0x1a1a2e)

eye_pos   = np.array([0.0, 0.0, 5.0])
eye_fov   = 45.0
aspect    = WIDTH / HEIGHT
zNear     = 0.1
zFar      = 50.0

angle     = 0.0          # 当前旋转角度（度）
DELTA     = 5.0          # 每次按键旋转量

# 是否显示立方体（选做）
show_cube = True

print("=== 操作说明 ===")
print("  A / D : 逆时针 / 顺时针旋转")
print("  T     : 切换 三角形 / 立方体 模式")
print("  ESC   : 退出")

while gui.running:
    # ── 事件处理 ──
    for e in gui.get_events(ti.GUI.PRESS):
        if e.key == ti.GUI.ESCAPE:
            gui.running = False
        elif e.key == 'a':
            angle += DELTA
        elif e.key == 'd':
            angle -= DELTA
        elif e.key == 't':
            show_cube = not show_cube

    # ── 构建 MVP ──
    M_model = get_model_matrix(angle)
    M_view  = get_view_matrix(eye_pos)
    M_proj  = get_projection_matrix(eye_fov, aspect, zNear, zFar)
    MVP     = M_proj @ M_view @ M_model

    # ── 绘制 ──
    if show_cube:
        # 选做：立方体线框
        pts_2d = apply_mvp(cube_vertices, MVP)
        pts_sc = ndc_to_screen(pts_2d)
        for i, j in cube_edges:
            gui.line(pts_sc[i], pts_sc[j], radius=1.5, color=0x00d4ff)
    else:
        # 基础：三角形线框（彩色边）
        verts = np.array(triangle_vertices, dtype=float)
        pts_2d = apply_mvp(verts, MVP)
        pts_sc = ndc_to_screen(pts_2d)
        edge_colors = [0xff4d4d, 0x4dff4d, 0x4d9fff]
        pairs = [(0,1),(1,2),(2,0)]
        for idx, (i, j) in enumerate(pairs):
            gui.line(pts_sc[i], pts_sc[j], radius=2.0, color=edge_colors[idx])
        for idx, p in enumerate(pts_sc):
            r, g, b = triangle_colors[idx]
            color = (int(r*255) << 16) | (int(g*255) << 8) | int(b*255)
            gui.circle(p, color=color, radius=5)

    # 显示当前模式与角度
    mode_str = "立方体 [T切换]" if show_cube else "三角形 [T切换]"
    gui.text(f"模式: {mode_str}   角度: {angle:.1f}°   [A/D 旋转]",
             pos=(0.02, 0.97), color=0xffffff, font_size=18)
    gui.show()