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
| ④ | 离线位姿估计 | ✅ 完成 (2026-07-01) |
| ⑤ | 实时联调 (采集→云端→位姿) | ✅ 完成 (2026-07-01) |
| ⑥ | 手眼标定 + 机器人抓取 | ⬜ 待开始 |

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

**当前状态**: ✅ 全部跑通！

#### 🎉 突破: 全链条联调成功 (20:42)
- ✅ `demo_cloud_pose.py` — 云端位姿估计脚本
- ✅ `demo_capture_for_cloud.py` — 本地采集脚本
- ✅ **端到端跑通**: RealSense 拍照 → YOLO 分割 → 上传 AutoDL → FoundationPose 位姿估计 → 返回 4×4 矩阵
  - mustard0: 评分 135.3, 1.4s
  - plane2.obj (自建模型): 评分 92.4, 1.4s
- ✅ 全链条打通: ① 拍照 ② YOLO ③ 上传 ④ 位姿 ⑤ 返回 全部就绪
- ⬜ 下一步: 手眼标定 + IK 求解 + 机器人执行抓取

