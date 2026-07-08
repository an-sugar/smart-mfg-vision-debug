# 视觉部分调试记录

> 项目: 智造赛 — 物理产线视觉与控制系统
> 更新: 2026-07-08

---

## 项目结构（按模块划分）

```
调试记录/
├── README.md                          # 本文件
├── Robot.py                           # FR3 XML-RPC SDK (本地引用)
├── best.pt                            # YOLOv8 分割模型权重
│
├── 📁 01_相机基础/
│   ├── demo_camera_test.py            # RealSense 连接 + 内参 + 实时预览
│   └── demo_env_check.py              # GPU/CUDA/依赖版本检查
│
├── 📁 02_YOLO分割/
│   ├── demo_yolo_test.py              # YOLO 分割验证 (单帧 + 可视化)
│   └── quick_capture.py               # 最小化拍照 + YOLO 检测
│
├── 📁 03_机器人基础/
│   ├── demo_drag_teach.py             # 拖动示教开关控制
│   ├── demo_two_point_move.py         # 两点拾放 MoveJ
│   ├── demo_move_back_forth.py        # 两点往复测试
│   ├── demo_move_ik.py                # IK 求解 + 绝对定位
│   ├── save_joints.py                 # 独立关节角保存 (20004 备用)
│   └── test_fk_convention.py          # FK 欧拉角约定测试 (Rz@Ry@Rx)
│
├── 📁 04_手眼标定/
│   ├── demo_handeye_calib.py          # 旧版标定 (Charuco)
│   ├── handeye_collect.py             # 交互式标定数据采集 (拖动示教)
│   ├── handeye_auto.py                # 自动标定采集 (MoveJ 到预设位姿)
│   ├── handeye_solve.py               # 标定求解 (Park/Horaud/Andreff)
│   ├── camera_to_robot.py             # 4×4 坐标转换验证
│   ├── T_cam_to_ee.txt                # 标定结果: 相机→法兰变换矩阵
│   ├── handeye_data_*.json            # 标定数据集
│   └── object_in_base.json            # 坐标转换结果示例
│
├── 📁 05_云端FP位姿/
│   ├── demo_capture_for_cloud.py      # 本地采集打包 (color+depth+K+YOLO mask)
│   ├── demo_cloud_pose.py             # 云端 FoundationPose 位姿估计
│   ├── demo_drag_capture_cloud.py     # 拖动示教 + 自动打包上传
│   ├── demo_pose_offline.py           # 离线 FP 测试
│   └── demo_pose_to_robot.py          # FP 结果 → 机器人坐标
│
├── 📁 06_全链条方案/
│   ├── demo_drag_with_preview.py      # 拖动示教 + 实时 YOLO 预览 + 空格拍照
│   ├── demo_senior_approach.py        # ⭐ 学长方案复刻 (两次定位+工具系平移)
│   ├── full_pipeline_final.py         # 两阶段全链条 (Phase1 拍照, Phase2 贴矩阵→移动)
│   └── full_pipeline_v2.py            # 一键全链条 V2
│
├── 📁 captured_cloud/                 # 拍照数据 (gitignored)
│   └── 2026MMDD_HHMMSS/
│       ├── color.png
│       ├── depth.npy
│       ├── mask_object1.png
│       ├── camera_K.txt
│       └── photo_joints.txt
│
└── 📁 错误尝试/ (仅供参考)
    ├── 错误尝试_mycpp_fallback.py     # C++ .so Python 替换
    ├── 错误尝试_pytorch3d_transforms_shim.py
    └── 错误尝试_setup_fake_pytorch3d.py
```

---

## 两种移动方案对比

| | 实测法 (V3) | 学长方案 |
|---|---|---|
| 脚本 | `simple_move_v3.py` (在 Foundationpose/) | `demo_senior_approach.py` |
| 定位次数 | 1 次 | 2 次 (粗调+精调) |
| 坐标系 | 基坐标系 (ref=2) | 工具坐标系 (ref=4) |
| 旋转对齐 | 无 (依赖 V3 经验映射) | MoveJ 转 J0 对齐物体 |
| 平移方式 | FK + 实测偏移 | 三角函数近似 (大角度) / FP 直移 (小角度) |
| 相机偏移 | X=0 Y=40 Z=-90 (卷尺测量) | 同左 |
| 优点 | 简单直接 | 不依赖基坐标绝对位置 |
| 精度 | 正常拍照姿态下可用 | 二次定位更精细 |

### 实测法 V3.1 转换公式

```python
# 相机→法兰映射 (相机绕光轴180°)
obj_f_x = CAM_X - cx          # 0 - cx
obj_f_y = CAM_Y - cz          # 40 - cz
obj_f_z = -CAM_Z - cy + cz*0.25  # 90 - cy + 0.25*cz

# 物体基坐标 = FK(拍照关节角) + 物体法兰偏移
obj_base = FK_photo.pos + obj_flange

# 移动量 = 物体基坐标 - 抓取位FK - 安全余量
delta = obj_base - FK_grasp.pos - [0, 0, 30]
```

### 工具坐标系方向 (ref=4) — 实测确认

| 轴 | dir=1 方向 |
|----|-----------|
| X+ | **上**（抬） |
| Y+ | **左** |
| Z+ | **前** |

---

## 核心链路

```
RealSense D435 (RGB-D)
    → YOLOv8 (best.pt, mask 分割)
    → AutoDL (FoundationPose, CUDA 12.8, RTX 5090)
    → 4×4 位姿矩阵
    → V3 实测法 或 学长工具系平移
    → FR3 StartJOG / MoveJ
```

---

## 已知问题

| 问题 | 影响 | 处理 |
|------|------|------|
| 20004 实时端口不稳定 | GetActualJointPosDegree 常返回全零 | 从示教器手动抄关节角 |
| 手眼标定旋转变化不足 | calibrateHandEye Tsai 失败 | Park 收敛但结果与实测不符，继续用实测值 |
| FR3 FK 欧拉角约定 | 4×4 矩阵链算不准 | 用 V3 简单加减法 (Rz@Ry@Rx 已确认) |
| 拍照姿态限制 | V3 在 J5 偏离 90° 太多时不准 | 尽量保持手腕朝下拍照 |

---

## 云端 FP 操作

```bash
# 1. 本机拍照
python demo_drag_with_preview.py  # 空格拍照, ESC退出

# 2. 上传到 AutoDL (拖拽 captured_cloud/最新文件夹)

# 3. AutoDL 终端
conda activate fp310
cd ~/Foundationpose
python demo_cloud_pose.py --data /root/202607XX_HHMMSS --mesh /root/obj_OBJ/mesh3/plane2.obj

# 4. 复制 4×4 矩阵 → 写 position1.txt → 跑移动
python simple_move_v3.py  # 实测法
# 或
python demo_senior_approach.py  # 学长方案
```
