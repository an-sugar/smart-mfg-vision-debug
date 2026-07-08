# -*- coding: utf-8 -*-
"""
调试步骤 ④ — 环境依赖检查（FoundationPose 位姿估计）
=====================================================
逐一检查 FoundationPose 所需的所有依赖，输出 通过/失败/建议。

用法:
  python demo_env_check.py
"""

import sys
import subprocess
from importlib import import_module


def check_python():
    """Python 版本"""
    v = sys.version_info
    ok = (3, 8) <= v < (3, 12)
    print(f"  Python {v.major}.{v.minor}.{v.micro}  {'✅' if ok else '❌ 需要 3.8-3.11'}")
    return ok


def check_module(name, pip_name=None):
    """检查 Python 包是否可导入"""
    try:
        m = import_module(name)
        v = getattr(m, '__version__', '?')
        print(f"  {name:20s} ✅  v{v}")
        return True
    except ImportError:
        print(f"  {name:20s} ❌ 未安装  →  pip install {pip_name or name}")
        return False


def check_torch_cuda():
    """检查 PyTorch CUDA 支持"""
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if cuda_ok else "N/A"
        print(f"  {'torch CUDA':20s} {'✅' if cuda_ok else '❌'}  GPU: {gpu_name}")
        print(f"  {'CUDA version':20s}    {torch.version.cuda or 'N/A'}")
        return cuda_ok
    except ImportError:
        print(f"  {'torch':20s} ❌ 未安装")
        return False


def check_nvidia_driver():
    """检查 NVIDIA 驱动"""
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
            shell=True, stderr=subprocess.DEVNULL, text=True
        )
        driver = out.strip()
        print(f"  {'NVIDIA Driver':20s} ✅  v{driver}")
        return True
    except Exception:
        print(f"  {'NVIDIA Driver':20s} ❌ 未检测到 (nvidia-smi 失败)")
        return False


def check_nvdiffrast():
    """nvdiffrast 是纯 CUDA 包，需要 CUDA 环境"""
    try:
        import nvdiffrast
        print(f"  {'nvdiffrast':20s} ✅")
        return True
    except ImportError:
        print(f"  {'nvdiffrast':20s} ❌ 未安装")
        print(f"  {'':20s}    → pip install git+https://github.com/NVlabs/nvdiffrast.git")
        return False


def check_warp():
    """NVIDIA Warp"""
    try:
        import warp
        print(f"  {'warp-lang':20s} ✅  v{warp.__version__ if hasattr(warp, '__version__') else '?'}")
        return True
    except ImportError:
        print(f"  {'warp-lang':20s} ❌ 未安装")
        return False


def check_pytorch3d():
    """PyTorch3D 较难装，Windows 下需要特殊步骤"""
    try:
        import pytorch3d
        print(f"  {'pytorch3d':20s} ✅  v{pytorch3d.__version__ if hasattr(pytorch3d, '__version__') else '?'}")
        return True
    except ImportError:
        print(f"  {'pytorch3d':20s} ❌ 未安装")
        print(f"  {'':20s}    Windows 安装: https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md")
        return False


def check_kaolin():
    """Kaolin — FoundationPose 的 Octree 依赖"""
    try:
        import kaolin
        print(f"  {'kaolin':20s} ✅")
        return True
    except ImportError:
        print(f"  {'kaolin':20s} ⚠️  建议安装（Octree加速）")
        return False


def check_trimesh():
    try:
        import trimesh
        print(f"  {'trimesh':20s} ✅  v{trimesh.__version__}")
        return True
    except ImportError:
        print(f"  {'trimesh':20s} ❌")
        return False


def main():
    print("=" * 60)
    print("  FoundationPose 环境依赖检查")
    print("=" * 60)

    results = {}

    # CPU
    print("\n── 基础环境 ──")
    results['python'] = check_python()

    # GPU / CUDA
    print("\n── GPU / CUDA ──")
    results['driver'] = check_nvidia_driver()
    results['torch_cuda'] = check_torch_cuda()

    # FoundationPose 核心
    print("\n── FoundationPose 核心依赖 ──")
    results['numpy'] = check_module('numpy')
    results['scipy'] = check_module('scipy')
    results['opencv'] = check_module('cv2', 'opencv-python')
    results['imageio'] = check_module('imageio')
    results['trimesh'] = check_trimesh()
    results['open3d'] = check_module('open3d')
    results['nvdiffrast'] = check_nvdiffrast()
    results['warp'] = check_warp()
    results['pytorch3d'] = check_pytorch3d()
    results['kaolin'] = check_kaolin()

    # 辅助
    print("\n── 辅助依赖 ──")
    results['yaml'] = check_module('yaml', 'pyyaml')
    results['sklearn'] = check_module('sklearn', 'scikit-learn')

    # 权重检查
    print("\n── FoundationPose 权重 ──")
    import os
    weights_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "Foundationpose", "weights"
    )
    weight_dirs = []
    if os.path.isdir(weights_dir):
        weight_dirs = [d for d in os.listdir(weights_dir) if os.path.isdir(os.path.join(weights_dir, d))]
    if weight_dirs:
        print(f"  ✅ 找到 {len(weight_dirs)} 个权重目录: {weight_dirs}")
    else:
        print(f"  ❌ 未找到权重！请下载并放入 Foundationpose/weights/")

    # 总结
    print(f"\n{'=' * 60}")
    failures = [k for k, v in results.items() if not v]
    if failures:
        print(f"  ❌ {len(failures)} 项未通过: {', '.join(failures)}")
    else:
        print(f"  ✅ 全部通过！可以开始位姿估计调试。")
    print(f"{'=' * 60}")

    # 安装建议
    if failures:
        print("\n📋 安装命令参考 (在新的 conda 环境中):")
        print("  conda create -n foundationpose python=3.10 -y")
        print("  conda activate foundationpose")
        print("  # 先装 PyTorch CUDA")
        print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        print("  # 其余依赖")
        print("  pip install numpy scipy opencv-python imageio trimesh open3d")
        print("  pip install warp-lang")
        print("  pip install git+https://github.com/NVlabs/nvdiffrast.git")
        print("  # pytorch3d Windows 版需要手动安装")


if __name__ == "__main__":
    main()
