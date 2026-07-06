"""Hand-eye calibration data collector (OpenCV 4.13+)"""
import cv2, numpy as np, sys, time, json, os
import pyrealsense2 as rs

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(SCRIPT_DIR, "handeye_data.json")
FP_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "Foundationpose")
sys.path.insert(0, FP_DIR)

K = np.array([[611.8, 0, 318.9], [0, 611.1, 242.8], [0, 0, 1]])

# Camera
pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipe.start(cfg)
for _ in range(60):
    frames = pipe.wait_for_frames()
    cf = frames.get_color_frame()
    if cf: color = np.asanyarray(cf.get_data()); break
pipe.stop()

# ArUco
aruco = cv2.aruco
dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
board = aruco.CharucoBoard((5, 7), 0.030, 0.0225, dict)
detector = aruco.ArucoDetector(dict, aruco.DetectorParameters())
marker_corners, ids, _ = detector.detectMarkers(color)

if ids is None:
    print("FAIL: no markers")
    sys.exit(1)

print(f"Markers: {len(ids)}")

charuco_detector = aruco.CharucoDetector(board)
charuco_detector.setCharucoParameters(aruco.CharucoParameters())
charuco_corners, charuco_ids, _, _ = charuco_detector.detectBoard(color)

if charuco_corners is None or len(charuco_corners) < 4:
    print(f"FAIL: only {len(charuco_corners) if charuco_corners is not None else 0} charuco corners")
    sys.exit(1)

print(f"Charuco corners: {len(charuco_corners)}")

# SolvePnP with matched points
obj_pts = board.getChessboardCorners()
obj_matched = obj_pts[charuco_ids.flatten()]
img_matched = charuco_corners.reshape(-1, 2)
ret, rvec, tvec = cv2.solvePnP(obj_matched, img_matched, K, None)
if not ret:
    print("FAIL: PnP failed")
    sys.exit(1)

print(f"Board: Z={tvec[2,0]*1000:.0f}mm")

# Robot FK - use known joints since 20004 is dead
j = [-84.4, -129.1, -69.8, -21.6, 90.0, -97.1]
print(f"Using joints: {[round(x,2) for x in j]}")

from Robot import RPC
robot = RPC("192.168.58.2")
ret = robot.GetForwardKin(j)
if ret[0] == 0:
    ee_pose = list(ret[1])  # FK returns (0, [x,y,z,rx,ry,rz])
    print(f"EE: X={ee_pose[0]:.0f} Y={ee_pose[1]:.0f} Z={ee_pose[2]:.0f}")
else:
    print(f"FK failed: {ret}")
    ee_pose = [0, 0, 0, 0, 0, 0]

# Save
data = {
    "ee_pose": ee_pose,
    "board_rvec": rvec.flatten().tolist(),
    "board_tvec": tvec.flatten().tolist(),
    "joints": j,
}
existing = json.load(open(DATA_FILE)) if os.path.exists(DATA_FILE) else []
existing.append(data)
json.dump(existing, open(DATA_FILE, "w"), indent=2)
print(f"SAVED! Total: {len(existing)}")

# Debug
cv2.aruco.drawDetectedMarkers(color, marker_corners, ids)
cv2.aruco.drawDetectedCornersCharuco(color, charuco_corners, charuco_ids)
cv2.drawFrameAxes(color, K, None, rvec, tvec, 0.05)
cv2.imwrite(os.path.join(SCRIPT_DIR, f"calib_{len(existing)-1}.png"), color)
