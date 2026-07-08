"""自动手眼标定采集 — MoveJ到预设位姿 → 拍照 → Charuco检测 → 保存"""
import cv2, numpy as np, sys, time, json, os, datetime
import pyrealsense2 as rs

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FP_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "Foundationpose")
sys.path.insert(0, FP_DIR)
from Robot import RPC

# ── 今日7组关节角 ──
JOINTS_SETS = [
    [-71.263, -67.591, -151.837, -19.719, 68.252, -100.126],
    [-101.717, -103.659, -94.486, -41.565, 112.206, -100.204],
    [-51.214, -35.819, -135.546, -53.425, 57.434, -50.869],
    [-57.223, -62.75, -121.858, -56.574, 51.325, -60.113],
    [-47.525, -68.025, -112.036, -66.384, 47.357, -72.569],
    [-84.789, -89.049, -94.529, -53.366, 73.677, -82.037],
    [-76.729, -28.789, -156.995, -24.538, 83.825, -82.014],
]

W, H = 640, 480
SQUARES_X, SQUARES_Y = 5, 7
SQUARE_LENGTH = 0.030
MARKER_LENGTH = 0.0225

# ── 相机 ──
print("Init camera...")
pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)
prof = pipe.start(cfg)
intr = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]])
for _ in range(30):
    pipe.wait_for_frames()

# ── ArUco ──
aruco = cv2.aruco
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
detector = aruco.ArucoDetector(aruco_dict, aruco.DetectorParameters())
board = aruco.CharucoBoard((SQUARES_X, SQUARES_Y), SQUARE_LENGTH, MARKER_LENGTH, aruco_dict)
charuco_detector = aruco.CharucoDetector(board)

# ── 机器人 ──
print("Connecting robot...")
robot = RPC("192.168.58.2")
robot.ResetAllError(); time.sleep(0.5)
robot.Mode(1); time.sleep(0.5)
robot.RobotEnable(1); time.sleep(1)

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
DATA_FILE = os.path.join(SCRIPT_DIR, f"handeye_data_{timestamp}.json")
collected = []

print(f"\n{'='*55}")
print(f"  Auto Hand-Eye Calibration {len(JOINTS_SETS)} poses")
print(f"  确认标定板固定不动！")
print(f"{'='*55}")

for i, joints in enumerate(JOINTS_SETS):
    print(f"\n[{'='*50}]")
    print(f"  Pose {i+1}/{len(JOINTS_SETS)}")
    print(f"  Joints: {[f'{j:.2f}' for j in joints]}")
    print(f"[{'='*50}]")

    # MoveJ
    print("  MoveJ...")
    robot.MoveJ(joints, 0, 0)
    time.sleep(5)

    # 拍照检测（拍10帧取最佳）
    best_rvec = best_tvec = None
    best_conf = 0
    for attempt in range(10):
        frames = pipe.wait_for_frames()
        color = np.asanyarray(frames.get_color_frame().get_data())
        marker_corners, ids, _ = detector.detectMarkers(color)
        if ids is not None and len(ids) >= 2:
            charuco_corners, charuco_ids, _, _ = charuco_detector.detectBoard(
                color, markerCorners=marker_corners, markerIds=ids)
            if charuco_corners is not None and len(charuco_corners) >= 6:
                obj_pts = board.getChessboardCorners()
                obj_matched = obj_pts[charuco_ids.flatten()]
                img_matched = charuco_corners.reshape(-1, 2)
                ret_pnp, rvec, tvec = cv2.solvePnP(obj_matched, img_matched, K, None)
                if ret_pnp:
                    if len(charuco_corners) > best_conf:
                        best_conf = len(charuco_corners)
                        best_rvec = rvec.copy()
                        best_tvec = tvec.copy()
                        best_color = color.copy()
        time.sleep(0.1)

    if best_rvec is None:
        print(f"  FAIL: Charuco not detected!")
        continue

    # FK
    ret = robot.GetForwardKin(joints)
    if ret[0] != 0:
        print(f"  FAIL: FK error")
        continue
    ee_pose = list(ret[1])

    # 保存
    data = {
        "joints": [round(j, 4) for j in joints],
        "ee_pose": [round(v, 4) for v in ee_pose],
        "board_rvec": best_rvec.flatten().tolist(),
        "board_tvec": best_tvec.flatten().tolist(),
    }
    collected.append(data)
    for f in [DATA_FILE, os.path.join(SCRIPT_DIR, "handeye_data_latest.json")]:
        with open(f, "w") as fp:
            json.dump(collected, fp, indent=2)

    dist = best_tvec[2, 0] * 1000
    print(f"  OK [{len(collected)}] EE: X={ee_pose[0]:.0f} Y={ee_pose[1]:.0f} Z={ee_pose[2]:.0f}  Board Z={dist:.0f}mm")

    # 保存预览
    display = best_color.copy()
    cv2.aruco.drawDetectedMarkers(display, marker_corners, ids)
    cv2.drawFrameAxes(display, K, None, best_rvec, best_tvec, 0.04)
    cv2.imwrite(os.path.join(SCRIPT_DIR, f"calib_{i}.png"), display)

pipe.stop()
cv2.destroyAllWindows()

print(f"\n{'='*55}")
print(f"  DONE! {len(collected)}/{len(JOINTS_SETS)} poses collected")
print(f"  File: {DATA_FILE}")
print(f"{'='*55}")

if len(collected) >= 3:
    print(f"\nRun: D:/anaconda/envs/vision/python.exe handeye_solve.py {DATA_FILE}")
