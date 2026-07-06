"""Solve hand-eye calibration from collected data"""
import cv2, numpy as np, json, os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(SCRIPT_DIR, "handeye_data.json")

data = json.load(open(DATA_FILE))
print(f"Data points: {len(data)}")

R_ee = []; t_ee = []
R_cam = []; t_cam = []

for i, d in enumerate(data):
    # EE pose (x,y,z,rx,ry,rz in mm, deg) from FK
    x, y, z, rx, ry, rz = d["ee_pose"]
    R, _ = cv2.Rodrigues(np.radians([rx, ry, rz]))
    t = np.array([[x], [y], [z]])  # mm
    R_ee.append(R)
    t_ee.append(t)

    # Board in camera (rvec, tvec)
    R_b, _ = cv2.Rodrigues(np.array(d["board_rvec"]))
    t_b = np.array(d["board_tvec"]).reshape(3, 1) * 1000  # m -> mm
    R_cam.append(R_b)
    t_cam.append(t_b)

# Eye-in-hand: AX=XB
# A = robot EE motion, B = camera motion observed
try:
    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_ee, t_ee, R_cam, t_cam,
        method=cv2.CALIB_HAND_EYE_TSAI)
except Exception as e:
    print(f"Tsai failed: {e}, trying Park...")
    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_ee, t_ee, R_cam, t_cam,
        method=cv2.CALIB_HAND_EYE_PARK)

T = np.eye(4)
T[:3, :3] = R_cam2gripper
T[:3, 3] = t_cam2gripper.flatten()  # mm
T_inv = np.linalg.inv(T)

print(f"\n{'='*55}")
print(f"  T_cam_to_ee (Camera -> End-Effector, mm):")
print(f"{'='*55}")
for row in T:
    print(f"  [{row[0]:10.4f} {row[1]:10.4f} {row[2]:10.4f} {row[3]:10.1f}]")
print(f"{'='*55}")

# Usage guide
print(f"\n  Usage:")
print(f"    1. FP gives obj_in_cam (4x4, meters)")
print(f"    2. Convert to mm: obj_in_cam[:3,3] *= 1000")
print(f"    3. obj_in_ee = T_cam_to_ee @ obj_in_cam")
print(f"    4. obj_in_base = T_ee_in_base @ obj_in_ee")
print(f"       where T_ee_in_base comes from FK at the current pose")

# Save
np.savetxt(os.path.join(SCRIPT_DIR, "T_cam_to_ee.txt"), T, fmt="%.6f",
           header="Camera to End-Effector transform (mm), eye-in-hand")
print(f"\n  Saved: T_cam_to_ee.txt")
