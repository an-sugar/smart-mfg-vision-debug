"""
全链条联调 — Step 7: 位姿 → IK → MoveJ 绝对定位
===============================================
从 FoundationPose 位姿直接算出目标关节角，MoveJ 过去，不再用 StartJOG 微调。

用法:
  python demo_move_ik.py
"""

import sys, os, time, math
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FP_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "Foundationpose")
sys.path.insert(0, FP_DIR)

from Robot import RPC
from robot_control import FR3RobotArm, move_along_x, move_along_y, move_along_z

POSE_FILE = os.path.join(FP_DIR, "demo_data/obj_OBJ/ob_in_cam/position1.txt")
INITIAL_JOINTS = [-85.820, -64.652, -122.536, -22.587, 88.657, -88.286]
BUCHANG = np.radians([0, -90, 0, -90, 0, -90])  # DH补偿
L = 150  # 相机到末端距离 (mm)


def main():
    # ── 1. 读位姿 ──
    pose = np.loadtxt(POSE_FILE)
    obj_x = float(pose[0, 3] * 1000)
    obj_y = float(pose[1, 3] * 1000)
    obj_z = float(pose[2, 3] * 1000)
    rz = math.degrees(math.atan2(-pose[0, 1], pose[0, 0]))
    if abs(rz) > 90:
        rz = rz - 180 if rz > 90 else rz + 180

    print("=" * 50)
    print(f"  物体在相机坐标系: X={obj_x:.1f} Y={obj_y:.1f} Z={obj_z:.1f} mm")
    print(f"  旋转角: rz={rz:.2f} deg")
    print("=" * 50)

    # ── 2. 相机坐标 → 机器人基坐标 (近似) ──
    # 相机装在末端，Z 向前，X 向右，Y 向下
    # 近似转换: 末端Z处前方L+obj_z处, 偏移obj_x/obj_y
    robot_target = np.array([obj_x, obj_y, obj_z + L])  # 简化近似

    print(f"\n  近似机器人目标坐标: X={robot_target[0]:.1f} Y={robot_target[1]:.1f} Z={robot_target[2]:.1f} mm")

    # ── 3. IK 求解 ──
    arm = FR3RobotArm()
    current_q = np.radians(INITIAL_JOINTS)

    # 从当前位姿出发，算移动到目标位置需要的关节角
    target_q = current_q.copy()

    print(f"\n  当前关节 (deg): {[f'{j:.1f}' for j in INITIAL_JOINTS]}")

    # 用 move_along 逐轴累加
    if abs(obj_x) > 0.5:
        target_q = move_along_x(arm, obj_x, target_q, BUCHANG)
    if abs(obj_y) > 0.5:
        target_q = move_along_y(arm, obj_y, target_q, BUCHANG)
    if abs(obj_z) > 0.5:
        target_q = move_along_z(arm, obj_z, target_q, BUCHANG)

    # 旋转对齐
    target_q[0] += math.radians(rz)

    target_deg = [f"{math.degrees(t):.1f}" for t in target_q]
    print(f"  目标关节 (deg): {target_deg}")

    # 验证正运动学
    fk = arm.forward_kinematics(target_q + BUCHANG)
    print(f"  FK 末端位置: X={fk[0]:.1f} Y={fk[1]:.1f} Z={fk[2]:.1f} mm")

    # ── 4. 连接机器人 ──
    print(f"\n  连接机器人...")
    robot = RPC("192.168.58.2")
    robot.ResetAllError(); time.sleep(0.5)
    robot.Mode(1); time.sleep(0.5)
    robot.RobotEnable(1); time.sleep(1)

    # ── 5. MoveJ 到目标 ──
    print(f"\n  MoveJ 到目标关节...")
    robot.MoveJ(list(target_deg), 0, 0)
    print(f"  等待移动完成...")
    time.sleep(6)

    print(f"\n{'=' * 50}")
    print(f"  到达目标位置")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
