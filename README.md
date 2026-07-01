# 视觉部分调试记录

> 项目: 智造赛 — 物理产线视觉与控制系统
> 开始日期: 2026-07-01
> 工作目录: `soft/soft/Foundationpose/`

---

## 调试流程概览

| 步骤 | 内容 | 状态 |
|------|------|------|
| ① | 环境依赖检查 | ✅ 完成 (2026-07-01) |
| ② | YOLO 分割验证 | 🔄 进行中 (2026-07-01) |
| ③ | 离线位姿估计 (run_demo.py) | ⬜ 待开始 |
| ④ | 实时相机 + 位姿估计 (Multi_YOLO_FP_new.py) | ⬜ 待开始 |
| ⑤ | 与机器人联调 | ⬜ 待开始 |

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

#### 步骤② YOLO 分割验证 🔄
- ✅ `demo_yolo_test.py` — 脚本重写为完全自包含版本
  - 所有输入输出均在 `调试记录/` 内
  - 支持相机拍照 / 已有图片 / 批量三种模式
- ✅ RealSense 拍照成功 — 保存到 `captures/capture_20260701_xxxx.jpg`
- ✅ YOLO 模型加载成功，推理耗时 ~0.4s
  - 模型: `train16/weights/best.pt`，3 类 (object1/2/3)
- ⚠️ **问题**: 模型返回 `(no detections)`，未检测到任何物体
  - **原因分析**: `best.pt` 是针对比赛道具（无人机、按钮等）训练的，当前相机场景中没有这些物体
  - **待验证**:
    1. 用 `origimgs/original_img.jpg` 旧图测试，确认模型本身是否正常
    2. 把比赛道具放到相机前，看能否正确检测
    3. 确认分辨率方向 — 模型输出 `480×640`，而 RealSense 配置为 `640×480`，需核实是否需要旋转

