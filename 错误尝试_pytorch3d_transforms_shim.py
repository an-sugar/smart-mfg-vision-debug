# -*- coding: utf-8 -*-
"""
pytorch3d.transforms 兼容模块
==============================
只实现 FoundationPose 实际用到的函数，纯 PyTorch，无需 pytorch3d。

用这个文件替换 Utils.py 中的:
    from pytorch3d.transforms import so3_log_map, so3_exp_map, se3_exp_map, se3_log_map, matrix_to_axis_angle, matrix_to_euler_angles, euler_angles_to_matrix, rotation_6d_to_matrix

用法:
    把这个文件放到 Foundationpose/ 目录下, 然后:
    from pytorch3d_transforms_shim import so3_log_map, so3_exp_map, se3_exp_map, se3_log_map, matrix_to_axis_angle, rotation_6d_to_matrix
"""

import torch
import torch.nn.functional as F
import math


# ═══════════════════════════════════════════════════════════
# SO(3) 映射
# ═══════════════════════════════════════════════════════════

def so3_exp_map(log_rot, eps=1e-4):
    """SO(3) 指数映射: 旋转向量 → 旋转矩阵 (batch)"""
    if log_rot.dim() == 1:
        log_rot = log_rot.unsqueeze(0)
    N = log_rot.shape[0]

    angle = torch.norm(log_rot, dim=1)
    angle_valid = angle > eps
    angle_safe = torch.where(angle_valid, angle, torch.ones_like(angle))

    axis = log_rot / angle_safe.unsqueeze(-1)

    cos = torch.cos(angle)
    sin = torch.sin(angle)

    K = torch.zeros(N, 3, 3, device=log_rot.device, dtype=log_rot.dtype)
    K[:, 0, 1] = -axis[:, 2]
    K[:, 0, 2] = axis[:, 1]
    K[:, 1, 0] = axis[:, 2]
    K[:, 1, 2] = -axis[:, 0]
    K[:, 2, 0] = -axis[:, 1]
    K[:, 2, 1] = axis[:, 0]

    I = torch.eye(3, device=log_rot.device, dtype=log_rot.dtype).unsqueeze(0)
    R = I + sin.unsqueeze(-1).unsqueeze(-1) * K + (1 - cos).unsqueeze(-1).unsqueeze(-1) * K @ K

    # 小角度近似
    R[~angle_valid] = I[0]

    return R.squeeze(0) if log_rot.shape[0] == 1 and log_rot.dim() == 1 else R


def so3_log_map(R, eps=1e-4, cos_bound=1e-4):
    """SO(3) 对数映射: 旋转矩阵 → 旋转向量 (batch)"""
    if R.dim() == 2:
        R = R.unsqueeze(0)
    N = R.shape[0]

    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    cos_angle = ((trace - 1) / 2).clamp(-1 + cos_bound, 1 - cos_bound)
    angle = torch.acos(cos_angle)

    # 旋转轴
    rx = R[:, 2, 1] - R[:, 1, 2]
    ry = R[:, 0, 2] - R[:, 2, 0]
    rz = R[:, 1, 0] - R[:, 0, 1]
    axis = torch.stack([rx, ry, rz], dim=1)

    angle_valid = angle > eps
    angle_safe = torch.where(angle_valid, angle, torch.ones_like(angle))
    axis_norm = torch.norm(axis, dim=1, keepdim=True)
    axis = axis / (axis_norm + 1e-8)

    log_rot = axis * angle_safe.unsqueeze(-1)
    log_rot[~angle_valid] = 0

    return log_rot.squeeze(0) if R.shape[0] == 1 else log_rot


# ═══════════════════════════════════════════════════════════
# SE(3) 映射
# ═══════════════════════════════════════════════════════════

def se3_exp_map(log_transform, eps=1e-4):
    """SE(3) 指数映射: 6D twist → 4x4 变换矩阵 (batch)"""
    if log_transform.dim() == 1:
        log_transform = log_transform.unsqueeze(0)
    N = log_transform.shape[0]

    rot_vec = log_transform[:, :3]
    trans = log_transform[:, 3:]

    R = so3_exp_map(rot_vec, eps=eps)
    if R.dim() == 2:
        R = R.unsqueeze(0)

    angle = torch.norm(rot_vec, dim=1)
    angle_valid = angle > eps
    angle_safe = torch.where(angle_valid, angle, torch.ones_like(angle))

    rot_vec_normalized = rot_vec / angle_safe.unsqueeze(-1)
    K = torch.zeros(N, 3, 3, device=log_transform.device, dtype=log_transform.dtype)
    K[:, 0, 1] = -rot_vec_normalized[:, 2]
    K[:, 0, 2] = rot_vec_normalized[:, 1]
    K[:, 1, 0] = rot_vec_normalized[:, 2]
    K[:, 1, 2] = -rot_vec_normalized[:, 0]
    K[:, 2, 0] = -rot_vec_normalized[:, 1]
    K[:, 2, 1] = rot_vec_normalized[:, 0]

    I = torch.eye(3, device=log_transform.device, dtype=log_transform.dtype).unsqueeze(0)
    sin = torch.sin(angle)
    cos = torch.cos(angle)

    V1 = sin.unsqueeze(-1).unsqueeze(-1) / angle_safe.unsqueeze(-1).unsqueeze(-1)
    V2 = (1 - cos).unsqueeze(-1).unsqueeze(-1) / (angle_safe.unsqueeze(-1).unsqueeze(-1) ** 2)

    V = I + V1 * K + V2 * K @ K

    T = torch.zeros(N, 4, 4, device=log_transform.device, dtype=log_transform.dtype)
    T[:, :3, :3] = R
    T[:, :3, 3] = (V @ trans.unsqueeze(-1)).squeeze(-1)
    T[:, 3, 3] = 1

    T[~angle_valid, :3, :3] = I[0]
    T[~angle_valid, :3, 3] = trans[~angle_valid]

    return T.squeeze(0) if N == 1 else T


def se3_log_map(transform, eps=1e-4, cos_bound=1e-4):
    """SE(3) 对数映射: 4x4 变换矩阵 → 6D twist (batch)"""
    if transform.dim() == 2:
        transform = transform.unsqueeze(0)
    N = transform.shape[0]

    R = transform[:, :3, :3]
    t = transform[:, :3, 3]

    rot_vec = so3_log_map(R, eps=eps, cos_bound=cos_bound)
    if rot_vec.dim() == 1:
        rot_vec = rot_vec.unsqueeze(0)

    angle = torch.norm(rot_vec, dim=1)
    angle_valid = angle > eps
    angle_safe = torch.where(angle_valid, angle, torch.ones_like(angle))

    rot_vec_normalized = rot_vec / angle_safe.unsqueeze(-1)
    K = torch.zeros(N, 3, 3, device=transform.device, dtype=transform.dtype)
    K[:, 0, 1] = -rot_vec_normalized[:, 2]
    K[:, 0, 2] = rot_vec_normalized[:, 1]
    K[:, 1, 0] = rot_vec_normalized[:, 2]
    K[:, 1, 2] = -rot_vec_normalized[:, 0]
    K[:, 2, 0] = -rot_vec_normalized[:, 1]
    K[:, 2, 1] = rot_vec_normalized[:, 0]

    I = torch.eye(3, device=transform.device, dtype=transform.dtype).unsqueeze(0)
    cos = torch.cos(angle)
    sin = torch.sin(angle)

    A = sin / angle_safe
    B = (1 - cos) / (angle_safe ** 2)
    C = (1 - A / (2 * B + 1e-8)) / (angle_safe ** 2 + 1e-8)

    V_inv_00 = torch.where(angle_valid, 1 - B * angle_safe**2*C, torch.ones_like(angle))
    V_inv = I - 0.5 * K + C.unsqueeze(-1).unsqueeze(-1) * K @ K

    trans = (V_inv @ t.unsqueeze(-1)).squeeze(-1)
    trans[~angle_valid] = t[~angle_valid]

    log_transform = torch.cat([rot_vec, trans], dim=1)

    return log_transform.squeeze(0) if N == 1 else log_transform


# ═══════════════════════════════════════════════════════════
# 矩阵转换
# ═══════════════════════════════════════════════════════════

def matrix_to_axis_angle(matrix):
    """旋转矩阵 → 轴角 (x,y,z, angle)"""
    return torch.cat([so3_log_map(matrix[:, :3, :3]), so3_log_map(matrix[:, :3, :3]).norm(dim=-1, keepdim=True)], dim=-1)


def rotation_6d_to_matrix(d6):
    """6D 连续旋转表示 → 旋转矩阵 (来自 Zhou et al. CVPR 2019)"""
    if d6.dim() == 1:
        d6 = d6.unsqueeze(0)
    N = d6.shape[0]
    a1, a2 = d6[:, :3], d6[:, 3:6]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    R = torch.stack([b1, b2, b3], dim=-1)
    return R  # (N, 3, 3)


# ═══════════════════════════════════════════════════════════
# 附加: quaternion 相关（可能用到）
# ═══════════════════════════════════════════════════════════

def quaternion_to_matrix(quaternions):
    """四元数 → 旋转矩阵"""
    if quaternions.dim() == 1:
        quaternions = quaternions.unsqueeze(0)
    r, i, j, k = quaternions[:, 0], quaternions[:, 1], quaternions[:, 2], quaternions[:, 3]
    two_s = 2.0 / (quaternions * quaternions).sum(dim=1)
    R = torch.stack([
        torch.stack([1 - two_s * (j*j + k*k), two_s * (i*j - k*r), two_s * (i*k + j*r)], dim=1),
        torch.stack([two_s * (i*j + k*r), 1 - two_s * (i*i + k*k), two_s * (j*k - i*r)], dim=1),
        torch.stack([two_s * (i*k - j*r), two_s * (j*k + i*r), 1 - two_s * (i*i + j*j)], dim=1),
    ], dim=2)
    return R
