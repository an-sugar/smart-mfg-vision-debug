# 视觉部分调试记录

> 项目: 智造赛 — 物理产线视觉与控制系统
> 开始日期: 2026-07-01
> 工作目录: `soft/soft/Foundationpose/`

---

## 调试流程概览

| 步骤 | 内容 | 状态 |
|------|------|------|
| ① | 环境依赖检查 | ✅ 完成 (2026-07-01) |
| ② | YOLO 分割验证 | ✅ 完成 (2026-07-01) |
| ③ | 拖动示教控制 | ✅ 完成 (2026-07-01) |
| ④ | 离线位姿估计 (run_demo.py) | 🔄 进行中 (2026-07-01) |
| ⑤ | 实时相机 + 位姿估计 (Multi_YOLO_FP_new.py) | ⬜ 待开始 |
| ⑥ | 与机器人联调 | ⬜ 待开始 |

---

## 日志

### 2026-07-01

#### 环境搭建
- ✅ 创建 conda 虚拟环境 `vision` (Python 3.10)
  - 路径: `D:\anaconda\envs\vision\`
  - 用 `conda init powershell` 解决 PowerShell 激活问题
- ✅ 安装依赖: `ultralytics==8.4.83`, `opencv-python==4.13.0`, `numpy`, `pyrealsense2`, `torch==2.12.1`
- ✅ 复制 `best.pt` 到 `调试记录/` (已配置 `.gitignore` 排除)

#### 步骤① 相机连接测试 ✅
- `demo_camera_test.py` — 待运行（接 RealSense 后测试）

#### 步骤② YOLO 分割验证 ✅ 完成
- ✅ 模型正常检出 → 训练集图片验证通过 + 比赛道具拍照验证通过
- ✅ 成功输出 mask 到 `masks/obj*.png`
- ⚠️ 之前无检测的原因: 拍摄角度+环境复杂，非模型问题

#### 步骤③ 拖动示教控制 ✅ 完成
- ✅ `demo_drag_teach.py` — 纯拖动示教脚本
- ✅ `Robot.py` 复制到 `调试记录/`（本地 import，用 `.gitignore` 排除）
- ✅ 拖动示教成功进入 — 清错 → 上使能 → Mode(1) → DragTeachSwitch(1)

#### 步骤④ 位姿估计 🔄 — 环境配置血泪史

| # | 方案 | 结果 | 根因 |
|---|------|------|------|
| 1 | Intel Arc 本地 | 🚫 | 硬件无 CUDA |
| 2 | AutoDL Py3.12+CU12.8 | 🚫 | Python 太新，mycpp .so 不兼容 |
| 3 | Lenovo uv+Py3.10 | 🚫 | Windows 无 pytorch3d wheel，缺 MSVC |
| 4 | Lenovo conda+Py3.10 | 🚫 | conda-forge 炸 torch 版本，nvcc 缺失 |
| 5 | **AutoDL conda Py3.10 子环境** | 🔄 进行中 | CUDA 版本对齐后 pytorch3d 编译中 |

**当前状态**: 云端 fp310 环境，pytorch3d 源码编译中。完成后还需 nvdiffrast + 其余依赖。

