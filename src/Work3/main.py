import taichi as ti


ti.init(arch=ti.cpu)


res_x, res_y = 800, 600

pixels = ti.Vector.field(3, dtype=ti.f32, shape=(res_x, res_y))

Ka = ti.field(dtype=ti.f32, shape=())
Kd = ti.field(dtype=ti.f32, shape=())
Ks = ti.field(dtype=ti.f32, shape=())
shininess = ti.field(dtype=ti.f32, shape=())

use_blinn = ti.field(dtype=ti.i32, shape=())
use_shadow = ti.field(dtype=ti.i32, shape=())

Ka[None] = 0.2
Kd[None] = 0.7
Ks[None] = 0.5
shininess[None] = 32.0

use_blinn[None] = 0
use_shadow[None] = 1


@ti.func
def normalize(v):
    return v / (v.norm() + 1e-6)

@ti.func
def reflect(I, N):
    return I - 2.0 * I.dot(N) * N

@ti.func
def intersect_sphere(ro, rd, center, radius):

    t = -1.0
    normal = ti.Vector([0.0, 0.0, 0.0])

    oc = ro - center

    a = rd.dot(rd)
    b = 2.0 * oc.dot(rd)
    c = oc.dot(oc) - radius * radius

    delta = b * b - 4.0 * a * c

    if delta > 0:

        t1 = (-b - ti.sqrt(delta)) / (2.0 * a)

        if t1 > 0:

            t = t1

            hit_pos = ro + rd * t

            normal = normalize(hit_pos - center)

    return t, normal

# 圆锥求交

@ti.func
def intersect_cone(ro, rd, apex, base_y, radius):

    t = -1.0
    normal = ti.Vector([0.0, 0.0, 0.0])

    H = apex.y - base_y

    k = (radius / H) ** 2

    ro_local = ro - apex

    A = rd.x * rd.x + rd.z * rd.z - k * rd.y * rd.y

    B = 2.0 * (
        ro_local.x * rd.x
        + ro_local.z * rd.z
        - k * ro_local.y * rd.y
    )

    C = (
        ro_local.x * ro_local.x
        + ro_local.z * ro_local.z
        - k * ro_local.y * ro_local.y
    )

    if ti.abs(A) > 1e-6:

        delta = B * B - 4.0 * A * C

        if delta > 0:

            t1 = (-B - ti.sqrt(delta)) / (2.0 * A)
            t2 = (-B + ti.sqrt(delta)) / (2.0 * A)

            t_first = t1
            t_second = t2

            if t1 > t2:
                t_first = t2
                t_second = t1

            y1 = ro_local.y + t_first * rd.y

            if t_first > 0 and -H <= y1 <= 0:

                t = t_first

            else:

                y2 = ro_local.y + t_second * rd.y

                if t_second > 0 and -H <= y2 <= 0:

                    t = t_second

            if t > 0:

                p_local = ro_local + rd * t

                normal = normalize(
                    ti.Vector([
                        p_local.x,
                        -k * p_local.y,
                        p_local.z
                    ])
                )

    return t, normal


# 渲染

@ti.kernel
def render():

    for i, j in pixels:

        u = (i - res_x * 0.5) / res_y * 2.0
        v = (j - res_y * 0.5) / res_y * 2.0

        ro = ti.Vector([0.0, 0.0, 5.0])

        rd = normalize(
            ti.Vector([u, v, -1.0])
        )

        min_t = 1e10

        hit_normal = ti.Vector([0.0, 0.0, 0.0])

        hit_color = ti.Vector([0.0, 0.0, 0.0])

        t_sph, n_sph = intersect_sphere(
            ro,
            rd,
            ti.Vector([-1.2, -0.2, 0.0]),
            1.2
        )

        if 0 < t_sph < min_t:

            min_t = t_sph

            hit_normal = n_sph

            hit_color = ti.Vector([0.8, 0.1, 0.1])

        t_cone, n_cone = intersect_cone(
            ro,
            rd,
            ti.Vector([1.2, 1.2, 0.0]),
            -1.4,
            1.2
        )

        if 0 < t_cone < min_t:

            min_t = t_cone

            hit_normal = n_cone

            hit_color = ti.Vector([0.6, 0.2, 0.8])

        color = ti.Vector([0.05, 0.15, 0.15])

        if min_t < 1e9:

            p = ro + rd * min_t

            N = hit_normal

            light_pos = ti.Vector([2.0, 3.0, 4.0])

            light_color = ti.Vector([1.0, 1.0, 1.0])

            L = normalize(light_pos - p)

            V = normalize(ro - p)

            ambient = Ka[None] * light_color * hit_color

            in_shadow = 0

            if use_shadow[None] == 1:

                sro = p + N * 1e-3

                srd = normalize(light_pos - p)

                light_dist = (light_pos - p).norm()

                ts, _ = intersect_sphere(
                    sro,
                    srd,
                    ti.Vector([-1.2, -0.2, 0.0]),
                    1.2
                )

                tc, _ = intersect_cone(
                    sro,
                    srd,
                    ti.Vector([1.2, 1.2, 0.0]),
                    -1.4,
                    1.2
                )

                if (
                    (ts > 1e-4 and ts < light_dist)
                    or
                    (tc > 1e-4 and tc < light_dist)
                ):
                    in_shadow = 1

            if in_shadow == 1:

                color = ti.math.clamp(
                    ambient,
                    0.0,
                    1.0
                )

            else:

                diff = ti.max(0.0, N.dot(L))

                diffuse = (
                    Kd[None]
                    * diff
                    * light_color
                    * hit_color
                )

                spec = 0.0

                if use_blinn[None] == 1:

                    H = normalize(L + V)

                    spec = (
                        ti.max(0.0, N.dot(H))
                        ** shininess[None]
                    )

                else:

                    R = normalize(reflect(-L, N))

                    spec = (
                        ti.max(0.0, R.dot(V))
                        ** shininess[None]
                    )

                specular = (
                    Ks[None]
                    * spec
                    * light_color
                )

                color = ti.math.clamp(
                    ambient + diffuse + specular,
                    0.0,
                    1.0
                )

        pixels[i, j] = color

# GUI

gui = ti.GUI(
    "Phong Shading Demo",
    res=(res_x, res_y)
)

blinn_on = False
shadow_on = True

print("=================================================")
print("操作说明")
print("=================================================")
print("P : 切换 Phong / Blinn-Phong")
print("T : 开关 Shadow")
print("")
print("Q / A : Ka 增加 / 减少")
print("W / S : Kd 增加 / 减少")
print("E / D : Ks 增加 / 减少")
print("R / F : Shininess 增加 / 减少")
print("")
print("ESC : 退出")
print("=================================================")

# =========================================================
# 主循环
# =========================================================
while gui.running:

    if gui.get_event(ti.GUI.PRESS):

        if gui.event.key == ti.GUI.ESCAPE:
            break

        elif gui.event.key == 'p':
            blinn_on = not blinn_on
            use_blinn[None] = 1 if blinn_on else 0

        elif gui.event.key == 't':
            shadow_on = not shadow_on
            use_shadow[None] = 1 if shadow_on else 0

        elif gui.event.key == 'q':
            Ka[None] = min(1.0, Ka[None] + 0.05)
        elif gui.event.key == 'a':
            Ka[None] = max(0.0, Ka[None] - 0.05)

        elif gui.event.key == 'w':
            Kd[None] = min(1.0, Kd[None] + 0.05)
        elif gui.event.key == 's':
            Kd[None] = max(0.0, Kd[None] - 0.05)

        elif gui.event.key == 'e':
            Ks[None] = min(1.0, Ks[None] + 0.05)
        elif gui.event.key == 'd':
            Ks[None] = max(0.0, Ks[None] - 0.05)

        elif gui.event.key == 'r':
            shininess[None] = min(128.0, shininess[None] + 2.0)
        elif gui.event.key == 'f':
            shininess[None] = max(1.0, shininess[None] - 2.0)

    render()

    gui.set_image(pixels)

    gui.text(content=f"Ka : {Ka[None]:.2f}", pos=(0.02, 0.95), color=0xFFFFFF)
    gui.text(content=f"Kd : {Kd[None]:.2f}", pos=(0.02, 0.90), color=0xFFFFFF)
    gui.text(content=f"Ks : {Ks[None]:.2f}", pos=(0.02, 0.85), color=0xFFFFFF)
    gui.text(content=f"Shininess : {shininess[None]:.1f}", pos=(0.02, 0.80), color=0xFFFFFF)
    gui.text(content=f"Model : {'Blinn-Phong' if blinn_on else 'Phong'}", pos=(0.02, 0.75), color=0xFFFFFF)
    gui.text(content=f"Shadow : {'ON' if shadow_on else 'OFF'}", pos=(0.02, 0.70), color=0xFFFFFF)

    gui.show()