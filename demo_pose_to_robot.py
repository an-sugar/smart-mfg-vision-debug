# -*- coding: utf-8 -*-
"""
全链条联调 — Step 6: 位姿 → 机器人移动
========================================
1. 读取云端返回的 4×4 位姿矩阵
2. 提取物体位置 + 旋转角度
3. 先打印确认, 再控制机器人对准

用法:
  python demo_pose_to_robot.py          # 仅预览
  python demo_pose_to_robot.py --move   # 实际控制机器人
"""

import os
import sys
import math
import time
import argparse

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FP_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "Foundationpose")
sys.path.insert(0, FP_DIR)

# ── 读取位姿 ──
POSE_FILE = os.path.join(FP_DIR, "demo_data", "obj_OBJ", "ob_in_cam", "position1.txt")

def read_pose(file_path):
    """读取 4×4 位姿矩阵, 返回 (pose_4x4, camera_xyz_mm, rz_deg)"""
    pose = np.loadtxt(file_path)
    # 物体在相机坐标系下的位置 (mm)
    camera_xyz = pose[:3, 3] * 1000  # m → mm
    # 绕 Z 轴旋转角度
    r11 = pose[0, 0]
    r12 = pose[0, 1]
    rz = math.degrees(math.atan2(-r12, r11))
    return pose, camera_xyz, rz


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--move", action="store_true", help="实际控制机器人（不加只预览）")
    args = parser.parse_args()

    print("=" * 55)
    print("  全链条联调 Step 6 — 位姿 → 机器人")
    print("=" * 55)

    # ── 1. 读取位姿 ──
    if not os.path.exists(POSE_FILE):
        print(f" 找不到位姿文件: {POSE_FILE}")
        print("   请先运行云端位姿估计，把结果保存到此路径。")
        sys.exit(1)

    pose, camera_xyz, rz = read_pose(POSE_FILE)

    print(f"\n   相机坐标系下的物体:")
    print(f"   X = {camera_xyz[0]:.1f} mm  (相机右)")
    print(f"   Y = {camera_xyz[1]:.1f} mm  (相机下)")
    print(f"   Z = {camera_xyz[2]:.1f} mm  (相机前)")
    print(f"   绕 Z 轴旋转 = {rz:.2f} deg")

    print(f"\n   完整位姿矩阵:")
    for row in pose:
        print(f"   {row}")

    if not args.move:
        print(f"\n{'=' * 55}")
        print("   预览模式。确认数值合理后，运行:")
        print("     python demo_pose_to_robot.py --move")
        print(f"{'=' * 55}")
        return

    # ── 2. 连接机器人 ──
    print(f"\n 连接机器人...")
    try:
        from Robot import RPC
        robot = RPC("192.168.58.2")
        print("    已连接")
    except Exception as e:
        print(f"    连接失败: {e}")
        sys.exit(1)

    # ── 3. 检查机器人状态 ──
    print(f"\n 检查机器人状态...")
    robot.ResetAllError()
    time.sleep(1)

    joints = robot.GetActualJointPosDegree()
    if joints[0] == 0:
        current_joints = joints[1]
    else:
        print(f"    获取关节位置失败: {joints}")
        sys.exit(1)

    print(f"   当前关节: {[f'{j:.2f}' for j in current_joints]}")

    # ── 4. 简单的旋转对齐 + 位置显示 ──
    print(f"\n 转换到机器人坐标系 (近似手眼标定):")
    print(f"   物体绕 Z 旋转: {rz:.2f}°")
    print(f"   需要对准的距离:")
    print(f"     X = {camera_xyz[0]:.1f} mm")
    print(f"     Y = {camera_xyz[1]:.1f} mm")
    print(f"     Z = {camera_xyz[2]:.1f} mm")

    print(f"\n️  注意: 手眼标定精确参数待校准。")
    print(f"   当前使用学长代码中的近似方法。")
    print(f"   建议先用手动拖动示教验证一个点。")

    # ── 5. 可选: 旋转对齐 ──
    if abs(rz) > 90:
        rz = rz - 180 if rz > 90 else rz + 180
        print(f"   ️  旋转角过大，反向后: {rz:.2f}°")

    print(f"\n 建议操作:")
    print(f"   1. 先用 demo_drag_teach.py 手动拖到一个拍照位")
    print(f"   2. 拍照 → 云端拿到位姿")
    print(f"   3. 对比位姿给的坐标和实际机器人位置")
    print(f"   4. 确定手眼标定参数后再自动移动")

    print(f"\n{'=' * 55}")
    print(f"  位姿数据已就绪，等待标定后自动控制")
    print(f"{'=' * 55}")


if __name__ == "__main__":
    main()
