# -*- coding: utf-8 -*-
"""
调试步骤 ④ — FoundationPose 离线位姿估计 Demo
===============================================
用预录数据集验证 FoundationPose 位姿估计是否正常工作。

前置条件:
  1. demo_env_check.py 全部通过
  2. Foundationpose/weights/ 下有预训练权重
  3. 有可用的 3D 模型 (.obj) 和 RGB-D 数据

用法:
  # 用 mustard0 演示数据（项目自带）
  python demo_pose_offline.py

  # 用你自己的物体
  python demo_pose_offline.py --mesh ../Foundationpose/demo_data/obj_OBJ/mesh3/plane2.obj
"""

import argparse
import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np

# ══════════════════════════════════════════════════════════════
# 路径配置 — 一切从本目录出发
# ══════════════════════════════════════════════════════════════
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FP_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "Foundationpose")

# 把 Foundationpose 加入搜索路径
sys.path.insert(0, FP_DIR)

# ══════════════════════════════════════════════════════════════
# 检查环境
# ══════════════════════════════════════════════════════════════
def check_ready():
    """快速检查 FoundationPose 需要的包是否可导入"""
    checks = []
    try:
        import torch
        assert torch.cuda.is_available()
        checks.append(("PyTorch CUDA", True))
    except Exception:
        checks.append(("PyTorch CUDA", False))

    for name in ["nvdiffrast", "warp", "trimesh", "open3d"]:
        try:
            __import__(name)
            checks.append((name, True))
        except ImportError:
            checks.append((name, False))

    # 权重目录
    wd = os.path.join(FP_DIR, "weights")
    has_weights = os.path.isdir(wd) and len(os.listdir(wd)) > 0
    checks.append(("预训练权重", has_weights))

    print("── 环境速查 ──")
    all_ok = True
    for name, ok in checks:
        print(f"  {name:20s} {'✅' if ok else '❌'}")
        if not ok:
            all_ok = False
    print()

    if not all_ok:
        print("请先运行 demo_env_check.py 查看详细诊断。")
        return False
    return True


# ══════════════════════════════════════════════════════════════
# 离线位姿估计
# ══════════════════════════════════════════════════════════════
def run_offline_pose(mesh_file, data_dir=None, debug=True):
    """
    对单个 3D 模型 + RGB-D 帧序列做位姿估计。
    复刻 run_demo.py 的核心逻辑。
    """
    # ── 导入 FoundationPose 相关模块 ──
    from estimater import FoundationPose
    from datareader import YcbineoatReader
    from Utils import (
        set_logging_format, set_seed,
        draw_posed_3d_box, draw_xyz_axis,
        depth2xyzmap,
    )
    from learning.training.predict_score import ScorePredictor
    from learning.training.predict_pose_refine import PoseRefinePredictor
    import nvdiffrast.torch as dr
    import trimesh
    import logging

    set_logging_format()
    set_seed(0)

    # ── 加载 3D 模型 ──
    print(f"加载模型: {mesh_file}")
    mesh = trimesh.load(mesh_file, force='mesh')
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

    # ── 初始化估计器 ──
    print("初始化 FoundationPose ...")
    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    est = FoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        mesh=mesh,
        scorer=scorer,
        refiner=refiner,
        glctx=glctx,
        debug=1 if debug else 0,
        debug_dir=os.path.join(SCRIPT_DIR, "pose_debug"),
    )
    print("  初始化完成。")

    # ── 确定数据来源 ──
    if data_dir is None:
        data_dir = os.path.join(FP_DIR, "demo_data", "mustard0")

    print(f"数据目录: {data_dir}")
    reader = YcbineoatReader(video_dir=data_dir, shorter_side=None, zfar=np.inf)
    print(f"  共 {len(reader.color_files)} 帧")

    # ── 逐帧处理 ──
    print(f"\n{'=' * 50}")
    print("  位姿估计中... 按 ESC 退出, 按空格暂停")
    print(f"{'=' * 50}\n")

    poses = []

    for i in range(len(reader.color_files)):
        color = reader.get_color(i)
        depth = reader.get_depth(i)

        try:
            if i == 0:
                mask = reader.get_mask(0).astype(bool)
                pose = est.register(
                    K=reader.K, rgb=color, depth=depth,
                    ob_mask=mask, iteration=5,
                )
                print(f"[{i:04d}] register ✅")
            else:
                pose = est.track_one(
                    rgb=color, depth=depth,
                    K=reader.K, iteration=2,
                )
                print(f"[{i:04d}] track   ✅")
        except Exception as e:
            print(f"[{i:04d}] ❌ 失败: {e}")
            continue

        poses.append(pose)

        # ── 可视化 ──
        if debug:
            center_pose = pose @ np.linalg.inv(to_origin)
            vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
            vis = draw_xyz_axis(
                vis, ob_in_cam=center_pose, scale=0.1,
                K=reader.K, thickness=3, transparency=0, is_input_rgb=True,
            )
            cv2.imshow("FoundationPose (ESC=退出 空格=暂停)", vis[..., ::-1])

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == 32:  # 空格暂停
                cv2.waitKey(0)

    cv2.destroyAllWindows()

    print(f"\n✅ 完成。共处理 {len(poses)} 帧。")

    # 保存最后一帧位姿
    if poses:
        out = os.path.join(SCRIPT_DIR, "pose_debug", "last_pose.txt")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        np.savetxt(out, poses[-1].reshape(4, 4))
        print(f"  最后帧位姿保存到: {out}")

    return poses


# ══════════════════════════════════════════════════════════════
# 主入口
# ══════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  调试步骤 ④ — FoundationPose 离线位姿估计")
    print("=" * 60)
    print()

    if not check_ready():
        sys.exit(1)

    parser = argparse.ArgumentParser(description="FoundationPose 离线位姿估计 Demo")
    parser.add_argument(
        "--mesh", type=str,
        default=os.path.join(FP_DIR, "demo_data", "mustard0", "mesh", "textured_simple.obj"),
        help="3D 模型 .obj 路径",
    )
    parser.add_argument(
        "--data", type=str, default=None,
        help="RGB-D 数据目录（不传则用 mustard0 演示数据）",
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="不弹可视化窗口",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.mesh):
        print(f"❌ 模型文件不存在: {args.mesh}")
        sys.exit(1)

    run_offline_pose(
        mesh_file=args.mesh,
        data_dir=args.data,
        debug=not args.no_display,
    )


if __name__ == "__main__":
    main()
