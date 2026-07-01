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
- ⚠️ **问题1**: 模型返回 `(no detections)`，未检测到任何物体
  - **原因**: `best.pt` 是针对比赛道具（无人机、按钮等）训练的，当前相机场景中没有这些物体
  - **待验证**: 把比赛道具放到相机前；用旧图 `origimgs/original_img.jpg` 测试模型本身是否正常
- ⚠️ **问题2**: RealSense D435 间歇性断连 `RuntimeError: No device connected`
  - **现象**: 连续多次运行后相机无法打开，报 `Frame didn't arrive within 5000` 或 `No device connected`
  - **原因**: 上一个脚本退出时 pipeline 未完全释放，相机被锁定；或 Windows USB 省电策略挂起设备
  - **临时解决**: 拔插 RealSense USB 线
  - **根治方法**: 设备管理器 → Intel RealSense D435 → 属性 → 电源管理 → **取消勾选"允许计算机关闭此设备以节约电源"**

