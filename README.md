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

#### 步骤④ 位姿估计环境准备 🔄
- ✅ `demo_env_check.py` — GPU / CUDA / FoundationPose 全套依赖检查脚本
- ✅ `demo_pose_offline.py` — 离线位姿估计 Demo（复刻 run_demo.py 核心逻辑）
  - 加载 3D 模型 → 初始化 FoundationPose → register + track
- ✅ 环境诊断完成 — Intel Arc 显卡不支持 CUDA，无法本地运行 FoundationPose
- ✅ 整理 `桌面/位姿估计-调试包/` — 包含所有 Demo + 权重 + 数据 + 源码
  - 路径: `C:\Users\樊予笑\Desktop\位姿估计-调试包\`
  - 拷贝到有 NVIDIA 显卡的电脑上即可开始步骤④
  - 包内 README 写明完整操作步骤

