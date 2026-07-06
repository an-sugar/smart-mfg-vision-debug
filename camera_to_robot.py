"""
FP位姿 → 机器人基坐标系 转换脚本
================================
相机偏移 (实测):
  X =   0 mm  (法兰中心)
  Y =  40 mm  (法兰前方)
  Z = -90 mm  (法兰下方)
"""

import sys, os, time, math
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FP_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "Foundationpose")
sys.path.insert(0, FP_DIR)

POSE_FILE = os.path.join(FP_DIR, "demo_data/obj_OBJ/ob_in_cam/position1.txt")

# 相机在末端法兰坐标系下的偏移 (mm)
CAM_X = 0
CAM_Y = 40   # 前方
CAM_Z = -90  # 下方

# 相机坐标系 → 法兰坐标系的旋转 (相机朝下, Z前→法兰Z下)
# 绕 X 轴旋转 -90°: 相机 Y↓ → 法兰 Z↓,  相机 Z前 → 法兰 Y←
RX_CAM_TO_FLANGE = math.radians(-90)  # 相机绕X转-90度后Z轴对齐法兰坐标系


def fp_to_robot_base(pose, ee_joints):
    """
    pose: 4×4 FP结果 (物体在相机系, 米)
    ee_joints: 6个关节角 (度)
    返回: 物体在机器人基坐标系下的位置 (x,y,z, mm) 和姿态
    """
    from Robot import RPC
    robot = RPC("192.168.58.2")

    # 1. 获取末端法兰在基坐标系下的位姿
    ret = robot.GetForwardKin(list(ee_joints))
    if ret[0] != 0:
        print(f"FK failed")
        return None
    fk = list(ret[1])  # [x,y,z,rx,ry,rz] mm, deg

    # 整个转换用 4×4 矩阵
    import cv2

    # T_ee_in_base: 法兰→基坐标
    R_ee, _ = cv2.Rodrigues(np.radians([fk[3], fk[4], fk[5]]))
    T_ee = np.eye(4)
    T_ee[:3, :3] = R_ee
    T_ee[:3, 3] = [fk[0], fk[1], fk[2]]

    # T_cam_in_ee: 相机→法兰 (固定偏移 + 旋转)
    # 相机坐标系: X右 Y↓ Z前
    # 法兰坐标系: 通常 Z↓/前 X右 Y前
    # 偏移在法兰系下: X=0 Y=40 Z=-90
    T_cam = np.eye(4)
    # 旋转: 绕X轴 -90度 把相机Z前转到下方
    Rx = np.array([[1, 0, 0],
                   [0, math.cos(RX_CAM_TO_FLANGE), -math.sin(RX_CAM_TO_FLANGE)],
                   [0, math.sin(RX_CAM_TO_FLANGE), math.cos(RX_CAM_TO_FLANGE)]])
    T_cam[:3, :3] = Rx
    T_cam[:3, 3] = [CAM_X, CAM_Y, CAM_Z]

    # 物体在相机下 (FP结果, 米 → 毫米)
    T_obj_in_cam = pose.copy()
    T_obj_in_cam[:3, 3] *= 1000  # m → mm

    # 物体在基坐标系下
    T_obj_in_base = T_ee @ T_cam @ T_obj_in_cam

    obj_x = T_obj_in_base[0, 3]
    obj_y = T_obj_in_base[1, 3]
    obj_z = T_obj_in_base[2, 3]

    return obj_x, obj_y, obj_z, T_obj_in_base, T_ee


def main():
    if not os.path.exists(POSE_FILE):
        print(f"No pose file: {POSE_FILE}")
        return

    pose = np.loadtxt(POSE_FILE)
    print(f"FP result loaded")

    # 物体在相机坐标系 (mm)
    x_cam = pose[0, 3] * 1000
    y_cam = pose[1, 3] * 1000
    z_cam = pose[2, 3] * 1000
    print(f"\nObject in camera frame: X={x_cam:.1f} Y={y_cam:.1f} Z={z_cam:.1f} mm")

    # 用已知关节角算
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--joints", type=float, nargs=6,
                       default=[-84.4, -129.1, -69.8, -21.6, 90.0, -97.1],
                       help="Current EE joint angles (degrees)")
    args = parser.parse_args()
    joints = list(args.joints)
    print(f"EE joints: {joints}")

    result = fp_to_robot_base(pose, joints)
    if result:
        obj_x, obj_y, obj_z, T_obj, T_ee = result
        print(f"\n{'='*50}")
        print(f"  Object in ROBOT BASE frame:")
        print(f"  X = {obj_x:.1f} mm")
        print(f"  Y = {obj_y:.1f} mm")
        print(f"  Z = {obj_z:.1f} mm")
        print(f"{'='*50}")

        # 保存
        import json
        data = {
            "obj_in_camera_mm": [x_cam, y_cam, z_cam],
            "obj_in_base_mm": [obj_x, obj_y, obj_z],
            "joints_used": joints,
            "cam_offset": [CAM_X, CAM_Y, CAM_Z],
            "T_obj_in_base": T_obj.tolist() if hasattr(T_obj, 'tolist') else str(T_obj),
        }
        out_file = os.path.join(SCRIPT_DIR, "object_in_base.json")
        json.dump(data, open(out_file, 'w'), indent=2)
        print(f"\nSaved: {out_file}")

        # 和示教器的抓取位对比
        print(f"\nCompare with teach pendant grasp point:")
        print(f"  Delta = object_base - grasp_base (compute manually)")
        print(f"  Use demo_two_point_move.py style to move robot")


if __name__ == "__main__":
    main()
