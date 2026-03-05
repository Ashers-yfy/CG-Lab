import taichi as ti
from .physics import initialize, compute_gravity, update, pos

def main():
    initialize()
    gui = ti.GUI("Gravitational Particle System - Blue + Mouse", res=(800, 800))
    
    while gui.running:
        # 获取鼠标位置
        if gui.is_pressed(ti.GUI.LMB):
            mouse_pos = ti.Vector(gui.get_cursor_pos(), dt=ti.f32)
        else:
            mouse_pos = ti.Vector([0.5, 0.5], dt=ti.f32)  # 默认中心
        
        compute_gravity(mouse_pos)
        update()

        # 绘制蓝色粒子
        gui.circles(pos.to_numpy(), radius=3, color=0x3399FF)
        gui.show()

if __name__ == "__main__":
    main()