"""手眼标定求解"""
import cv2, numpy as np, json, os, sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 数据文件
if len(sys.argv) > 1:
    data_file = sys.argv[1]
else:
    # 找最新的 handeye_data_*.json
    import glob
    files = sorted(glob.glob(os.path.join(SCRIPT_DIR, "handeye_data_*.json")), reverse=True)
    if not files:
        # fallback
        data_file = os.path.join(SCRIPT_DIR, "handeye_data.json")
    else:
        data_file = files[0]

print(f"Data file: {data_file}")
data = json.load(open(data_file))
print(f"Data points: {len(data)}")

R_ee = []; t_ee = []
R_cam = []; t_cam = []

for i, d in enumerate(data):
    # EE pose (x,y,z,rx,ry,rz in mm, deg) from FK
    ee = d["ee_pose"]
    x, y, z, rx, ry, rz = ee
    R, _ = cv2.Rodrigues(np.radians([rx, ry, rz]))
    t = np.array([[x], [y], [z]])  # mm
    R_ee.append(R)
    t_ee.append(t)

    # Board in camera (rvec, tvec)
    R_b, _ = cv2.Rodrigues(np.array(d["board_rvec"]))
    t_b = np.array(d["board_tvec"]).reshape(3, 1) * 1000  # m → mm
    R_cam.append(R_b)
    t_cam.append(t_b)

    print(f"  [{i}] EE: [{x:.1f}, {y:.1f}, {z:.1f}]mm  rz={rz:.1f}°")

# 求解 (eye-in-hand: AX=XB)
methods = [
    ("Park", cv2.CALIB_HAND_EYE_PARK),
    ("Horaud", cv2.CALIB_HAND_EYE_HORAUD),
    ("Andreff", cv2.CALIB_HAND_EYE_ANDREFF),
    ("Daniilidis", cv2.CALIB_HAND_EYE_DANIILIDIS),
    ("Tsai", cv2.CALIB_HAND_EYE_TSAI),
]

best_T = None
best_method = ""

for name, method in methods:
    try:
        R, t = cv2.calibrateHandEye(R_ee, t_ee, R_cam, t_cam, method=method)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t.flatten()

        # 计算平移分量
        tx, ty, tz = t.flatten()
        rx_r, ry_r, rz_r = cv2.Rodrigues(R)[0].flatten()
        print(f"\n  [{name}]  offset(mm): X={tx:.1f}  Y={ty:.1f}  Z={tz:.1f}")
        print(f"           rotation(rad): {rx_r:.4f} {ry_r:.4f} {rz_r:.4f}")

        # 跳过明显垃圾解 (平移全零)
        if abs(t[0]) < 0.01 and abs(t[1]) < 0.01 and abs(t[2]) < 0.01:
            continue
        if best_T is None:
            best_T = T
            best_method = name
    except Exception as e:
        print(f"  [{name}] FAILED: {e}")

if best_T is None:
    print("\nAll methods failed. Need more data or more rotation variation.")
    sys.exit(1)

T = best_T
print(f"\n{'='*60}")
print(f"  T_cam_to_ee (Camera → EE, mm)  [{best_method}]")
print(f"{'='*60}")
for row in T:
    print(f"  [{row[0]:10.4f} {row[1]:10.4f} {row[2]:10.4f} {row[3]:10.1f}]")
print(f"{'='*60}")

# 保存
np.savetxt(os.path.join(SCRIPT_DIR, "T_cam_to_ee.txt"), T, fmt="%.6f",
           header=f"Camera to End-Effector transform (mm), eye-in-hand [{best_method}]")
print(f"\nSaved: T_cam_to_ee.txt")

# ── 使用验证 ──
print(f"\n{'='*60}")
print(f"  使用方法:")
print(f"    T_cam_to_ee @ obj_in_cam(mm) = obj_in_ee")
print(f"    T_ee_in_base(FK) @ obj_in_ee = obj_in_base")
print(f"{'='*60}")

# 对比实测值
print(f"\n实测值: X=0mm  Y=40mm  Z=-90mm (法兰前方40, 下方90)")
print(f"标定值: X={T[0,3]:.0f}mm  Y={T[1,3]:.0f}mm  Z={T[2,3]:.0f}mm")
