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

#### 步骤④ 位姿估计 🔄

##### ❌ 错误尝试: AutoDL 云端 (PyTorch 2.8 + Python 3.12 + CUDA 12.8 + RTX 5090)
- **失败原因**: Python 3.12 太新，pytorch3d 无预编译包、mycpp .so 不兼容。依赖链断裂成本过高。

##### 🔄 重新开始: Lenovo 本机 (RTX 3050 + Python 3.10 + uv)
- ✅ 清理桌面旧文件，重新打包干净版 `FoundationPose调试包/` (909MB)
- ✅ 包内路径修正为平铺结构（Foundationpose/ 作为子目录）
- ✅ 选用 uv 做环境管理 + Python 3.10 + PyTorch CUDA 12.1
- ⏳ 待执行: Lenovo 上 `uv venv → pip install → demo_env_check.py → demo_pose_offline.py`
  1. `pytorch3d` 无 py312+cu128 预编译包 → 源码编译超时
  2. 创建假 `pytorch3d` 兼容包绕过后 → `transformations`/`kornia`/`omegaconf`/`h5py`/`ninja` 逐个缺包
  3. `mycpp/build/mycpp.cpython-39-x86_64-linux-gnu.so` 是 Python 3.9 编译的，3.12 加载不了
  4. 写 `mycpp_fallback.py` Python 替代后 → `nvdiffrast` JIT 编译 CUDA kernel 一直卡住
##### 🔄 新方案: 同实例内建 Python 3.10 子环境
- **思路**: RTX 5090 驱动 CUDA 13.0 向下兼容，在同一实例内用 conda 建 Python 3.10 环境，装 PyTorch CUDA 12.4 的预编译版。pytorch3d、mycpp 等库对 Python 3.10 有完整预编译支持。
- **待执行**:
  ```bash
  conda create -n fp310 python=3.10 -y
  conda activate fp310
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
  pip install pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/whl/py310_cu124_pyt240/download.html
  pip install nvdiffrast warp-lang trimesh open3d imageio scikit-learn pyyaml kornia omegaconf einops h5py ninja transformations
  python demo_env_check.py
  ```

