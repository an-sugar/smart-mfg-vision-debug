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
| ⑥ | 机器人移动到位姿位置 | ✅ 完成 (2026-07-02) |
| ⑦ | 全体统调 + 两点 Pick-and-Place | 🔄 进行中 (2026-07-03) |
| ⑧ | 手眼标定 | ⬜ 待做 |

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

### 2026-07-02

#### 步骤⑥ 机器人移动到位姿位置 🎉 完成
- ✅ 全链条联调: RealSense拍照 → YOLO(drone 95.6%) → 上传 → 位姿1.4s → 下载 → 机器人移动
- ✅ 解决 FR3 "PTP关节指令超限" 错误 → 需在示教器上物理复位+上使能
- ✅ 解决 20004 实时数据口断连问题 → 用已知初始位推算, 不依赖读数
- ✅ `simple_move.py` — 一条命令从位姿到机器人就位
- ✅ `full_pipeline.py` — 最终全自动化脚本
- ✅ `demo_drag_capture_cloud.py` — 拖动示教版采集脚本（15秒自动定时）
- ✅ `demo_move_ik.py` — IK 求解 + MoveJ 绝对定位
- ✅ 整理初赛考题 → `智造赛初赛备考指南.md` (Obsidian)

### 2026-07-03

#### 步骤⑦ 全体统调 + 两点 Pick-and-Place 🔄

- ✅ 学习法奥官方 Python SDK 文档 (接口规范/参数格式/返回值)
- ✅ `demo_move_back_forth.py` — 两点往复移动 (MoveJ)
- ✅ `quick_capture.py` — 纯拍照+YOLO (不含拖动示教)
- ✅ `demo_two_point_move.py` — 拖动示教激活 + 两点移动
  - 从示教器获取抓取位和放置位关节角
  - DragTeach 进/出激活伺服 → MoveJ 成功执行
- ✅ `full_pipeline_v2.py` — 全流程 V2 (拍照→FP→移动)
- ✅ **全链条再次跑通**: 拍照(YOLO conf=0.36) → 上传 → FP 位姿(1.4s) → 下载
- 🔄 FP 位姿 → 机器人偏移量: `simple_move.py` 执行但精度待验证
  - 本次位姿: X=-13.6 Y=31.5 Z=404.8mm rz=-3.24°
- ⚠️ **核心问题**: 手眼标定缺失, FP 相机坐标 → 机器人基坐标偏差大
- ✅ 整理 Obsidian 备考笔记:
  - `初赛-Python编程考点.md` (14题原题+答案)
  - `初赛-Python基础知识复习笔记.md` (9考点+7预测+44练习题)
  - `初赛-FR3机器人SDK考点.md` (SDK接口+25练习题)
  - `智造赛初赛备考指南.md` (6大模块全解析)

