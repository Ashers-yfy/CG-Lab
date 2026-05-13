import taichi as ti

ti.init(arch=ti.cpu)

res_x, res_y = 800, 600
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(res_x, res_y))

# UI 参数
light_pos_x = ti.field(ti.f32, shape=())
light_pos_y = ti.field(ti.f32, shape=())
light_pos_z = ti.field(ti.f32, shape=())
max_bounces = ti.field(ti.i32, shape=())
samples_per_pixel = ti.field(ti.i32, shape=())

# 材质枚举
MAT_DIFFUSE = 0
MAT_MIRROR  = 1
MAT_GLASS   = 2
IOR_GLASS   = 1.5


@ti.func
def normalize(v):
    return v / v.norm(1e-5)


@ti.func
def reflect(I, N):
    return I - 2.0 * I.dot(N) * N


@ti.func
def refract(I, N, ior):
    cosi = ti.math.clamp(I.dot(N), -1.0, 1.0)
    etai = 1.0
    etat = ior
    n    = N
    if cosi > 0:
        temp = etai
        etai = etat
        etat = temp
        n    = -N
    cosi = ti.abs(cosi)
    eta  = etai / etat
    k    = 1.0 - eta * eta * (1.0 - cosi * cosi)
    refracted = ti.Vector([0.0, 0.0, 0.0])
    total_internal_reflection = False
    if k < 0.0:
        total_internal_reflection = True
    else:
        refracted = normalize(eta * I + (eta * cosi - ti.sqrt(k)) * n)
    return refracted, total_internal_reflection


@ti.func
def intersect_sphere(ro, rd, center, radius):
    t      = -1.0
    normal = ti.Vector([0.0, 0.0, 0.0])
    oc     = ro - center
    b      = 2.0 * oc.dot(rd)
    c      = oc.dot(oc) - radius * radius
    delta  = b * b - 4.0 * c
    if delta > 0:
        t1 = (-b - ti.sqrt(delta)) / 2.0
        if t1 > 0:
            t      = t1
            p      = ro + rd * t
            normal = normalize(p - center)
    return t, normal


@ti.func
def intersect_plane(ro, rd, plane_y):
    t      = -1.0
    normal = ti.Vector([0.0, 1.0, 0.0])
    if ti.abs(rd.y) > 1e-5:
        t1 = (plane_y - ro.y) / rd.y
        if t1 > 0:
            t = t1
    return t, normal


@ti.func
def scene_intersect(ro, rd):
    min_t   = 1e10
    hit_n   = ti.Vector([0.0, 0.0, 0.0])
    hit_c   = ti.Vector([0.0, 0.0, 0.0])
    hit_mat = MAT_DIFFUSE

    # 左侧玻璃球
    t, n = intersect_sphere(ro, rd, ti.Vector([-1.2, 0.0, 0.0]), 1.0)
    if 0 < t < min_t:
        min_t   = t
        hit_n   = n
        hit_c   = ti.Vector([0.9, 0.95, 1.0])
        hit_mat = MAT_GLASS

    # 右侧镜面球
    t, n = intersect_sphere(ro, rd, ti.Vector([1.2, 0.0, 0.0]), 1.0)
    if 0 < t < min_t:
        min_t   = t
        hit_n   = n
        hit_c   = ti.Vector([0.9, 0.9, 0.9])
        hit_mat = MAT_MIRROR

    # 地板
    t, n = intersect_plane(ro, rd, -1.0)
    if 0 < t < min_t:
        min_t   = t
        hit_n   = n
        hit_mat = MAT_DIFFUSE
        p        = ro + rd * t
        grid_scale = 2.0
        ix = ti.floor(p.x * grid_scale)
        iz = ti.floor(p.z * grid_scale)
        if (int(ix) + int(iz)) % 2 == 0:
            hit_c = ti.Vector([0.3, 0.3, 0.3])
        else:
            hit_c = ti.Vector([0.8, 0.8, 0.8])

    return min_t, hit_n, hit_c, hit_mat


@ti.kernel
def render():
    light_pos = ti.Vector([light_pos_x[None], light_pos_y[None], light_pos_z[None]])
    bg_color  = ti.Vector([0.05, 0.15, 0.2])

    for i, j in pixels:
        pixel_color = ti.Vector([0.0, 0.0, 0.0])

        for sample in range(16):
            if sample >= samples_per_pixel[None]:
                break

            u  = ((i + ti.random()) - res_x / 2.0) / res_y * 2.0
            v  = ((j + ti.random()) - res_y / 2.0) / res_y * 2.0
            ro = ti.Vector([0.0, 1.0, 5.0])
            rd = normalize(ti.Vector([u, v - 0.2, -1.0]))

            final_color = ti.Vector([0.0, 0.0, 0.0])
            throughput  = ti.Vector([1.0, 1.0, 1.0])

            for bounce in range(8):
                if bounce >= max_bounces[None]:
                    break

                t, N, obj_color, mat_id = scene_intersect(ro, rd)

                if t > 1e9:
                    final_color += throughput * bg_color
                    break

                p = ro + rd * t

                if mat_id == MAT_MIRROR:
                    ro         = p + N * 1e-4
                    rd         = normalize(reflect(rd, N))
                    throughput *= 0.8 * obj_color

                elif mat_id == MAT_GLASS:
                    refracted_dir, tir = refract(rd, N, IOR_GLASS)
                    if tir:
                        rd = normalize(reflect(rd, N))
                        ro = p + N * 1e-4
                    else:
                        rd = refracted_dir
                        ro = p - N * 1e-4
                    throughput *= 0.95 * obj_color

                else:  # MAT_DIFFUSE
                    L               = normalize(light_pos - p)
                    shadow_ray_orig = p + N * 1e-4
                    shadow_t, _, _, _ = scene_intersect(shadow_ray_orig, L)
                    dist_to_light   = (light_pos - p).norm()
                    in_shadow       = 0.0
                    if shadow_t < dist_to_light:
                        in_shadow = 1.0

                    ambient      = 0.2 * obj_color
                    direct_light = ambient

                    if in_shadow == 0.0:
                        diff     = ti.max(0.0, N.dot(L))
                        diffuse  = 0.8 * diff * obj_color
                        view_dir = normalize(-rd)
                        half_dir = normalize(L + view_dir)
                        spec     = ti.pow(ti.max(N.dot(half_dir), 0.0), 64.0)
                        specular = ti.Vector([1.0, 1.0, 1.0]) * spec * 0.3
                        direct_light += diffuse + specular

                    final_color += throughput * direct_light
                    break

            pixel_color += final_color

        pixels[i, j] = ti.math.clamp(pixel_color / samples_per_pixel[None], 0.0, 1.0)


# ── 主程序：ti.GUI ──
def main():
    gui = ti.GUI("Ray Tracing Demo", res=(res_x, res_y))

    light_pos_x[None]       = 2.0
    light_pos_y[None]       = 4.0
    light_pos_z[None]       = 3.0
    max_bounces[None]       = 3
    samples_per_pixel[None] = 1

    lx, ly, lz = 2.0, 4.0, 3.0
    mb, spp    = 3, 1
    selected   = 0
    names      = ["Light X", "Light Y", "Light Z", "Bounces", "Samples"]

    print("=== 操作说明 ===")
    print("  Tab    : 切换当前调节参数")
    print("  ↑ / ↓  : 增大 / 减小当前参数")
    print("  ESC    : 退出")

    while gui.running:
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key == ti.GUI.TAB:
                selected = (selected + 1) % 5
            elif e.key == ti.GUI.UP:
                if   selected == 0: lx += 0.2
                elif selected == 1: ly  = min(8.0, ly + 0.2)
                elif selected == 2: lz += 0.2
                elif selected == 3: mb  = min(5,   mb + 1)
                elif selected == 4: spp = min(16,  spp + 1)
            elif e.key == ti.GUI.DOWN:
                if   selected == 0: lx -= 0.2
                elif selected == 1: ly  = max(1.0, ly - 0.2)
                elif selected == 2: lz -= 0.2
                elif selected == 3: mb  = max(1,   mb - 1)
                elif selected == 4: spp = max(1,   spp - 1)

        light_pos_x[None]       = lx
        light_pos_y[None]       = ly
        light_pos_z[None]       = lz
        max_bounces[None]       = mb
        samples_per_pixel[None] = spp

        render()
        gui.set_image(pixels)

        gui.text(f"[Tab] >>>{names[selected]}<<<  [Up/Down] adjust",
                 pos=(0.01, 0.97), color=0xFFFFFF, font_size=16)
        gui.text(f"LX={lx:.1f}  LY={ly:.1f}  LZ={lz:.1f}  Bounces={mb}  Samples={spp}",
                 pos=(0.01, 0.93), color=0xAAFFAA, font_size=16)
        gui.text("Left=Glass  Right=Mirror  Ground=Checkerboard",
                 pos=(0.01, 0.89), color=0xFFFF88, font_size=16)

        gui.show()


if __name__ == '__main__':
    main()