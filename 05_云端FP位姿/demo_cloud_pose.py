# -*- coding: utf-8 -*-
"""
全链条联调 — Step 2: 云端位姿估计脚本
======================================
上传到 AutoDL 运行，吃本地采集的数据包，输出 4×4 位姿矩阵。

用法 (在 AutoDL 终端):
  conda activate fp310
  python demo_cloud_pose.py --data /root/20260701_205357 --mesh /root/obj_OBJ/mesh3/plane2.obj

输出:
  {data_dir}/pose_4x4.txt    ← 物体在相机坐标系的位姿
"""

import argparse
import os
import sys
import json
import time
import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "Foundationpose"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="采集数据包路径")
    parser.add_argument("--mesh", required=True, help="3D 模型 .obj")
    parser.add_argument("--est_refine_iter", default=5, type=int)
    parser.add_argument("--debug_dir", default=None)
    args = parser.parse_args()

    data_dir = args.data
    mesh_file = args.mesh

    print("=" * 55)
    print("  全链条联调 Step 2 — 云端位姿估计")
    print("=" * 55)
    print(f"  数据包: {data_dir}")
    print(f"  模型:   {mesh_file}")
    print()

    # ── 1. 读取数据 ──
    color = cv2.imread(os.path.join(data_dir, "color.png"))
    if color is None:
        print(f"❌ 找不到 color.png")
        sys.exit(1)
    print(f"✅ color.png ({color.shape[1]}×{color.shape[0]})")

    depth_m = np.load(os.path.join(data_dir, "depth.npy"))
    print(f"✅ depth.npy  ({depth_m.shape}, {depth_m.min():.3f}~{depth_m.max():.3f}m)")

    K = np.loadtxt(os.path.join(data_dir, "camera_K.txt"))
    print(f"✅ K: fx={K[0,0]:.1f} fy={K[1,1]:.1f} cx={K[0,2]:.1f} cy={K[1,2]:.1f}")

    # 找 mask
    mask_files = sorted([f for f in os.listdir(data_dir) if f.startswith("mask_")])
    if not mask_files:
        print("❌ 无 mask 文件")
        sys.exit(1)

    masks = []
    for mf in mask_files:
        mask = cv2.imread(os.path.join(data_dir, mf), cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            masks.append((mf, (mask > 127).astype(bool)))
            print(f"✅ {mf} ({mask.shape})")

    print()

    # ── 2. 初始化 FoundationPose ──
    print("加载 FoundationPose...")
    from estimater import FoundationPose
    from Utils import set_logging_format, set_seed
    from learning.training.predict_score import ScorePredictor
    from learning.training.predict_pose_refine import PoseRefinePredictor
    import nvdiffrast.torch as dr
    import trimesh

    set_logging_format()
    set_seed(0)

    mesh = trimesh.load(mesh_file, force='mesh')
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

    debug_dir = args.debug_dir or os.path.join(data_dir, "pose_debug")
    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    est = FoundationPose(
        model_pts=mesh.vertices, model_normals=mesh.vertex_normals,
        mesh=mesh, scorer=scorer, refiner=refiner,
        glctx=glctx, debug=1, debug_dir=debug_dir,
    )
    print("  初始化完成。\n")

    # ── 3. 位姿估计 ──
    print("=" * 55)
    print("  位姿估计中...")
    print("=" * 55)

    t0 = time.time()

    for idx, (mask_name, mask_bin) in enumerate(masks):
        try:
            pose = est.register(
                K=K, rgb=color, depth=depth_m,
                ob_mask=mask_bin, iteration=args.est_refine_iter,
            )
            elapsed = time.time() - t0
            print(f"\n🎯 [{mask_name}] 位姿估计成功！({elapsed:.1f}s)")

            # 保存位姿
            pose_file = os.path.join(data_dir, f"pose_{mask_name.replace('.png', '')}.txt")
            np.savetxt(pose_file, pose.reshape(4, 4), fmt="%.6f")
            print(f"   位姿保存到: {pose_file}")
            print(f"   位姿矩阵:\n{pose.reshape(4,4)}")

            # 提取平移分量 (相机坐标系下)
            tx, ty, tz = pose[0, 3], pose[1, 3], pose[2, 3]
            print(f"\n📐 物体在相机坐标系下的位置:")
            print(f"   X = {tx*1000:.1f} mm")
            print(f"   Y = {ty*1000:.1f} mm")
            print(f"   Z = {tz:.3f} m ({tz*1000:.1f} mm)")

        except Exception as e:
            print(f"❌ [{mask_name}] 失败: {e}")

    print(f"\n{'=' * 55}")
    print(f"  ✅ 完成！结果保存在: {data_dir}")
    print(f"{'=' * 55}")


if __name__ == "__main__":
    main()
