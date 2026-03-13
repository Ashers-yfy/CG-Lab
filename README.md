# CG-Lab: 交互式引力粒子系统可视化

本项目是一个基于 **Taichi** 实现的二维粒子系统可视化程序。  
系统通过模拟粒子在引力作用下的动态运动，展示粒子群在空间中的实时演化过程，并支持鼠标交互。

用户可以通过鼠标位置改变局部引力中心，从而观察粒子轨迹的变化，获得直观的物理仿真体验。

---

## 📌 项目特性

- 基于 **Taichi** 实现高性能粒子更新
- 支持二维粒子运动模拟
- 支持鼠标交互控制引力中心
- 粒子运动效果可实时可视化
- 项目结构简洁，适合学习基础仿真与图形可视化

---

## 🛠 项目结构

    CG-Lab/
    ├─ src/
    │  └─ Work0/
    │     ├─ main.py        # 程序入口，负责窗口创建与主循环
    │     ├─ physics.py     # 粒子运动与引力更新逻辑
    │     └─ config.py      # 参数配置文件
    ├─ .venv/               # Python 虚拟环境（已忽略）
    ├─ .gitignore
    ├─ pyproject.toml
    ├─ README.md
    └─ uv.lock

---

## 🧠 原理简介

程序中每个粒子都具有位置与速度。  
在每一帧中，系统会根据引力中心的位置计算粒子受到的引力作用，并更新其速度和位置，从而形成连续动态的粒子运动效果。

当按下鼠标左键时，鼠标位置会作为新的引力中心；  
未按下时，系统默认以窗口中心作为引力中心。

这使得用户能够通过交互方式实时影响粒子的整体运动趋势。

---

## ⚙️ 运行流程

程序主要流程如下：

1. 初始化粒子系统参数
2. 创建 Taichi GUI 窗口
3. 获取鼠标位置作为引力输入
4. 计算粒子受力并更新状态
5. 将所有粒子绘制到窗口中
6. 循环执行，形成连续动画

---

## 📺 效果演示

以下为粒子系统运行时的动态效果：

<p align="center">
  <img src="https://github.com/user-attachments/assets/6bdab4ef-f3fe-49a8-8468-8ed6a42d3f55" width="380" alt="交互式引力粒子系统演示 GIF">
</p>

---

## 🚀 快速开始

### 1. 克隆项目

    git clone <your-repo-url>
    cd CG-Lab

### 2. 创建并激活虚拟环境

Windows PowerShell:

    python -m venv .venv
    Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
    .\.venv\Scripts\Activate.ps1

### 3. 安装依赖

    pip install taichi

### 4. 运行程序

    cd src/Work0
    python main.py

---

## 🧩 环境要求

- Python 3.10 及以上
- Taichi 1.7.x
- Windows / macOS / Linux

---

## 📚 适用场景


- 学习粒子系统基础原理
- 入门 Taichi 编程

---

## 📄 说明

本项目仅用于北京师范大学人工智能学院计算机图形学实验一展示
