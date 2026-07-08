"""
手眼标定 — 交互式数据采集
========================
操作:
  标定板固定放在桌面
  拖动示教移动机械臂到不同姿态（尤其要绕不同轴旋转！）
  每到一个位姿 → 空格键采集一组数据
  采集 10-15 组（越多越好，旋转变化要大）
  ESC 退出

每帧自动检测 Charuco 角点，绿色=检测成功，红色=失败。
"""
import cv2, numpy as np, sys, time, json, os, datetime
import pyrealsense2 as rs

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FP_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "Foundationpose")
sys.path.insert(0, FP_DIR)
from Robot import RPC

# ── 配置 ──
W, H = 640, 480
# Charuco 板参数（跟你的打印板对齐）
SQUARES_X, SQUARES_Y = 5, 7
SQUARE_LENGTH = 0.030    # 方格边长 (m)
MARKER_LENGTH = 0.0225   # ArUco 标记边长 (m)

# ── 初始化相机 ──
print("初始化相机...")
pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)
prof = pipe.start(cfg)

# 获取内参
intr = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]])
print(f"K: fx={K[0,0]:.1f} fy={K[1,1]:.1f}")

# 等流稳定
for _ in range(30):
    pipe.wait_for_frames()

# ── 初始化 ArUco ──
aruco = cv2.aruco
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
aruco_params = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, aruco_params)
board = aruco.CharucoBoard((SQUARES_X, SQUARES_Y), SQUARE_LENGTH, MARKER_LENGTH, aruco_dict)
charuco_detector = aruco.CharucoDetector(board)

# ── 连接机器人 ──
print("连接机器人...")
robot = RPC("192.168.58.2")
robot.ResetAllError(); time.sleep(0.5)
robot.Mode(1); time.sleep(0.5)
robot.RobotEnable(1); time.sleep(1)
robot.DragTeachSwitch(1)
print("DragTeach ON — 手动移动机械臂到不同姿态")

# ── 数据容器 ──
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
DATA_FILE = os.path.join(SCRIPT_DIR, f"handeye_data_{timestamp}.json")
collected = json.load(open(os.path.join(SCRIPT_DIR, "handeye_data.json"))) if os.path.exists(os.path.join(SCRIPT_DIR, "handeye_data.json")) else []

print(f"\n{'='*55}")
print(f"  手眼标定数据采集")
print(f"  标定板固定放桌面！不要移动！")
print(f"  用拖动示教改变姿态（多绕不同轴旋转）")
print(f"  空格=采集  ESC=退出")
print(f"{'='*55}\n")

try:
    while True:
        frames = pipe.wait_for_frames()
        color = np.asanyarray(frames.get_color_frame().get_data())
        display = color.copy()

        # 检测
        marker_corners, ids, _ = detector.detectMarkers(color)
        detected_ok = False
        rvec = tvec = None

        if ids is not None and len(ids) >= 2:
            charuco_corners, charuco_ids, _, _ = charuco_detector.detectBoard(color, markerCorners=marker_corners, markerIds=ids)

            if charuco_corners is not None and len(charuco_corners) >= 6:
                # PnP
                obj_pts = board.getChessboardCorners()
                obj_matched = obj_pts[charuco_ids.flatten()]
                img_matched = charuco_corners.reshape(-1, 2)
                ret_pnp, rvec, tvec = cv2.solvePnP(obj_matched, img_matched, K, None)
                if ret_pnp:
                    detected_ok = True
                    cv2.aruco.drawDetectedMarkers(display, marker_corners, ids)
                    cv2.drawFrameAxes(display, K, None, rvec, tvec, 0.04)
                    for pt in charuco_corners.reshape(-1, 2).astype(int):
                        cv2.circle(display, tuple(pt), 3, (0, 255, 0), -1)

        # 状态栏
        status = "OK" if detected_ok else "NO DETECT"
        color_status = (0, 255, 0) if detected_ok else (0, 0, 255)
        cv2.putText(display, f"Charuco: {status}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_status, 2)
        dist = tvec[2, 0] * 1000 if tvec is not None else 0
        cv2.putText(display, f"Dist: {dist:.0f}mm", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(display, f"Collected: {len(collected)}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(display, "SPACE=Capture  ESC=Save&Exit", (10, H-15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("Hand-Eye Calibration | SPACE=Capture ESC=Exit", display)
        key = cv2.waitKey(33) & 0xFF

        if key == 32 and detected_ok:
            # ── 退出拖动示教，用户从示教器读取关节角 ──
            robot.DragTeachSwitch(0)
            time.sleep(0.5)

            print(f"\n{'='*50}")
            print(f"  位姿 #{len(collected)+1} — 请从示教器读取关节角")
            print(f"{'='*50}")

            try:
                j_str = input("  输入 J1 J2 J3 J4 J5 J6: ")
                joints = [float(x) for x in j_str.strip().split()]
                if len(joints) != 6:
                    print("  需要6个值，跳过")
                    robot.DragTeachSwitch(1)
                    continue
            except:
                print("  输入无效，跳过")
                robot.DragTeachSwitch(1)
                continue

            ret = robot.GetForwardKin(joints)
            if ret[0] != 0:
                print(f"  FK计算失败!")
                robot.DragTeachSwitch(1)
                continue
            ee_pose = list(ret[1])

            # 保存
            data = {
                "joints": [round(j, 4) for j in joints],
                "ee_pose": [round(v, 4) for v in ee_pose],
                "board_rvec": rvec.flatten().tolist(),
                "board_tvec": tvec.flatten().tolist(),
            }
            collected.append(data)
            # 双写: 时间戳文件 + 固定备份
            for f in [DATA_FILE, os.path.join(SCRIPT_DIR, "handeye_data_latest.json")]:
                with open(f, "w") as fp:
                    json.dump(collected, fp, indent=2)

            # 保存预览图
            cv2.imwrite(os.path.join(SCRIPT_DIR, f"calib_{len(collected)-1}.png"), display)
            print(f"  [{len(collected)}] SAVED | EE: X={ee_pose[0]:.0f} Y={ee_pose[1]:.0f} Z={ee_pose[2]:.0f} | Board Z={dist:.0f}mm\n")

            # ── 重新进入拖动示教 ──
            print("  拖动示教重新开启，移到下一个姿态")
            robot.DragTeachSwitch(1)

        elif key == 27:
            break

finally:
    robot.DragTeachSwitch(0)
    pipe.stop()
    cv2.destroyAllWindows()
    print(f"\nDone. {len(collected)} data points saved to {DATA_FILE}")
    if len(collected) >= 3:
        print(f"Run: python handeye_solve.py {DATA_FILE}")
    else:
        print("Need at least 3 points with rotation variation!")
