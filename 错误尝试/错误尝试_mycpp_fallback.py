# -*- coding: utf-8 -*-
"""
mycpp.cluster_poses 的纯 Python 替代实现
用于 Python 3.12（原 .so 是 3.9 编译的，加载不了）
"""

import numpy as np


def rotation_geodesic_distance(R1, R2):
    """计算两个旋转矩阵之间的测地距离 (弧度)"""
    R_diff = R1.T @ R2
    trace = np.trace(R_diff)
    cos_angle = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    return np.arccos(cos_angle)


def cluster_poses(angle_diff, dist_diff, poses_in, symmetry_tfs=None):
    """
    对位姿候选进行聚类去重

    参数:
        angle_diff: 角度阈值 (度)
        dist_diff: 平移阈值 (米)
        poses_in: 候选位姿列表, shape (N, 4, 4)
        symmetry_tfs: 对称变换列表, shape (S, 4, 4) 或 None

    返回:
        聚类后的位姿列表, shape (M, 4, 4)
    """
    poses_in = np.asarray(poses_in)
    if symmetry_tfs is not None:
        symmetry_tfs = np.asarray(symmetry_tfs)
    if symmetry_tfs is None or len(symmetry_tfs) == 0:
        symmetry_tfs = [np.eye(4)]

    radian_thres = np.deg2rad(angle_diff)

    print(f"num original candidates = {len(poses_in)}")
    poses_out = [poses_in[0]]

    for i in range(1, len(poses_in)):
        cur_pose = poses_in[i]
        cur_R = cur_pose[:3, :3]
        cur_t = cur_pose[:3, 3]

        is_new = True
        for cluster in poses_out:
            c_t = cluster[:3, 3]

            if np.linalg.norm(cur_t - c_t) >= dist_diff:
                continue

            for tf in symmetry_tfs:
                cur_pose_sym = cur_pose @ tf
                rot_diff = rotation_geodesic_distance(
                    cur_pose_sym[:3, :3], cluster[:3, :3]
                )
                if rot_diff < radian_thres:
                    is_new = False
                    break

            if not is_new:
                break

        if is_new:
            poses_out.append(cur_pose)

    print(f"num of pose after clustering: {len(poses_out)}")
    return np.array(poses_out)
