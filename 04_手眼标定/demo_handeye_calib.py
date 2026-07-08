"""
手眼标定数据采集 + 求解
=======================
眼在手 (eye-in-hand): 相机装在机器人末端

流程:
  1. 打印 charuco_board.png 并贴在桌面固定位置
  2. 运行本脚本, 拖动示教到不同角度拍照
  3. 每拍一次记录: 关节角 + 标定板在相机下的位姿
  4. 采集 5-10 组后自动求解

用法:
  python demo_handeye_calib.py          # 采集数据
  python demo_handeye_calib.py --solve  # 仅求解(已有数据)
"""

import sys, os, time, json, pickle
import numpy as np
import cv2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FP_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "Foundationpose")
sys.path.insert(0, FP_DIR)
DATA_FILE = os.path.join(SCRIPT_DIR, "handeye_data.pkl")

# ── ArUco 配置 (与打印的标定板对应) ──
ARUCO_DICT = cv2.aruco.DICT_6X6_250
SQUARE_LENGTH = 0.030   # 棋盘格方块边长 (m) — 请用尺子实测!
MARKER_LENGTH = 0.0225  # ArUco 标记边长 (m) — 请用尺子实测!


def capture_rgb():
    """RealSense 拍照, 返回 BGR 图像"""
    import pyrealsense2 as rs
    W, H = 640, 480
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30)
    pipe.start(cfg)
    align = rs.align(rs.stream.color)
    color = None
    for _ in range(60):
        frames = pipe.wait_for_frames()
        aligned = align.process(frames)
        cf = aligned.get_color_frame()
        if cf:
            color = np.asanyarray(cf.get_data())
            break
    pipe.stop()
    return color


def detect_board(img):
    """检测 Charuco 标定板, 返回 (rvec, tvec) 或 (None, None)"""
    aruco = cv2.aruco
    dict = aruco.getPredefinedDictionary(ARUCO_DICT)
    board = aruco.CharucoBoard((5, 7), SQUARE_LENGTH, MARKER_LENGTH, dict)
    params = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(dict, params)

    corners, ids, _ = detector.detectMarkers(img)
    if ids is None or len(ids) < 4:
        return None, None

    ret, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
        corners, ids, img, board)
    if charuco_ids is None or len(charuco_ids) < 4:
        return None, None

    valid, rvec, tvec = aruco.estimatePoseCharucoBoard(
        charuco_corners, charuco_ids, board,
        cameraMatrix=K, distCoeffs=None,
        rvec=None, tvec=None)

    if valid:
        # 画检测结果
        cv2.aruco.drawDetectedMarkers(img, corners, ids)
        cv2.aruco.drawDetectedCornersCharuco(img, charuco_corners, charuco_ids)
        cv2.drawFrameAxes(img, K, None, rvec, tvec, 0.05)
        return rvec, tvec
    return None, None


def get_ee_pose(robot):
    """获取机器人末端法兰在基坐标系下位姿 [x,y,z,rx,ry,rz] (mm, deg)"""
    joints = robot.GetActualJointPosDegree()
    if joints[0] != 0:
        return None
    j = joints[1]
    ret = robot.GetForwardKin(j)
    if ret[0] == 0:
        return [ret[1], ret[2], ret[3], ret[4], ret[5], ret[6]]
    return None


def capture_one(robot):
    """采集一组数据: 拍照 + 检测标定板 + 读取机器人位姿"""
    print("\n" + "-" * 40)

    # 拍照
    color = capture_rgb()
    if color is None:
        print("  Camera failed")
        return None

    # 检测标定板
    vis = color.copy()
    rvec, tvec = detect_board(vis)
    if rvec is None:
        print("  No board detected!")
        cv2.imwrite(os.path.join(SCRIPT_DIR, "calib_debug_no_board.png"), color)
        print("  Saved debug image: calib_debug_no_board.png")
        return None

    # 读取机器人位姿
    ee_pose = get_ee_pose(robot)
    if ee_pose is None:
        print("  Cannot read robot pose")
        return None

    print(f"  Board detected OK")
    print(f"  EE pose: X={ee_pose[0]:.1f} Y={ee_pose[1]:.1f} Z={ee_pose[2]:.1f} "
          f"RX={ee_pose[3]:.2f} RY={ee_pose[4]:.2f} RZ={ee_pose[5]:.2f}")

    # 保存预览图
    cv2.imwrite(os.path.join(SCRIPT_DIR, f"calib_{len(data)}_{time.strftime('%H%M%S')}.png"), vis)

    return {
        "ee_pose": ee_pose,        # 末端法兰位姿 (mm, deg)
        "board_rvec": rvec.flatten().tolist(),  # 标定板在相机下旋转向量
        "board_tvec": tvec.flatten().tolist(),  # 标定板在相机下平移 (m)
    }


def solve_handeye(data_list):
    """用采集的数据求解手眼标定矩阵"""
    import cv2

    R_gripper2base = []
    t_gripper2base = []
    R_target2cam = []
    t_target2cam = []

    for d in data_list:
        # 末端位姿 → 4×4 矩阵
        x, y, z, rx, ry, rz = d["ee_pose"]
        R_ee, _ = cv2.Rodrigues(np.array([np.radians(rx), np.radians(ry), np.radians(rz)]))
        t_ee = np.array([[x], [y], [z]])  # mm
        R_gripper2base.append(R_ee)
        t_gripper2base.append(t_ee)

        # 标定板在相机下 → 4×4
        R_board, _ = cv2.Rodrigues(np.array(d["board_rvec"]))
        t_board = np.array(d["board_tvec"]).reshape(3, 1) * 1000  # m→mm
        R_target2cam.append(R_board)
        t_target2cam.append(t_board)

    # eye-in-hand: AX=XB, A=ee_motion, B=board_motion
    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_gripper2base, t_gripper2base,
        R_target2cam, t_target2cam,
        method=cv2.CALIB_HAND_EYE_TSAI)

    T = np.eye(4)
    T[:3, :3] = R_cam2gripper
    T[:3, 3] = t_cam2gripper.flatten()  # mm

    print(f"\n{'=' * 50}")
    print(f"  Hand-Eye Calibration Result (T_cam_to_ee):")
    print(f"{'=' * 50}")
    for row in T:
        print(f"  {row[0]:8.3f} {row[1]:8.3f} {row[2]:8.3f} {row[3]:8.1f}")
    print(f"{'=' * 50}")

    # Eye-in-hand 用法:
    # 1. FP gives obj_in_camera (m)
    # 2. obj_in_ee = T_cam_to_ee @ obj_in_camera
    # 3. obj_in_base = T_ee_in_base @ obj_in_ee

    return T


# ══════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════
data = []

if os.path.exists(DATA_FILE):
    with open(DATA_FILE, 'rb') as f:
        data = pickle.load(f)
    print(f"Loaded {len(data)} existing data points")

if "--solve" in sys.argv:
    if len(data) < 3:
        print(f"Need at least 3 data points, have {len(data)}")
        sys.exit(1)
    solve_handeye(data)
    sys.exit(0)

# ── 连接机器人 ──
from Robot import RPC
robot = RPC("192.168.58.2")
robot.ResetAllError(); time.sleep(0.5)
robot.Mode(1); time.sleep(0.5)
robot.RobotEnable(1); time.sleep(1)

# ── 预加载相机内参 ──
import pyrealsense2 as rs
pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
prof = pipe.start(cfg)
intr = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
global K
K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]])
pipe.stop()
print(f"Camera K: fx={K[0,0]:.1f} fy={K[1,1]:.1f} cx={K[0,2]:.1f} cy={K[1,2]:.1f}")

# ── 循环采集 ──
print(f"\n{'=' * 50}")
print(f"  Hand-Eye Calibration Data Collection")
print(f"{'=' * 50}")
print(f"  Place the Charuco board on the table.")
print(f"  Each round: drag robot → photo → detect")
print(f"  Need 5-10 good captures.")
print(f"  Type 'q' to stop and solve.")
print(f"{'=' * 50}")

while True:
    cmd = input(f"\n[{len(data)} collected] Enter to capture, 'q' to solve: ").strip()
    if cmd == 'q':
        break

    # 拖动示教
    print("  Drag teach ON (drag robot to new pose, then wait)...")
    robot.DragTeachSwitch(1)
    time.sleep(10)
    robot.DragTeachSwitch(0)
    time.sleep(1)

    # 采集
    d = capture_one(robot)
    if d:
        data.append(d)
        with open(DATA_FILE, 'wb') as f:
            pickle.dump(data, f)
        print(f"  Saved. Total: {len(data)}")

if len(data) < 3:
    print(f"\nNeed at least 3 points, have {len(data)}. Exiting.")
    sys.exit(1)

T = solve_handeye(data)

# 保存标定矩阵
np.savetxt(os.path.join(SCRIPT_DIR, "T_cam_to_ee.txt"), T, fmt="%.6f",
           header="Camera to End-Effector transform (eye-in-hand)")
print(f"\nCalibration matrix saved to: T_cam_to_ee.txt")
print(f"Copy this to FoundationPose, use it in future scripts.")
