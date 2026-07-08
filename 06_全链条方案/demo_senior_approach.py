"""
学长方案复刻 — 两次定位 + 旋转对齐 + 工具坐标系平移
====================================================
内置拍照(YOLO+深度+K矩阵保存)，只差云端FP需要手动粘贴。

流程:
  ① MoveJ → 初始拍照位
  ② 自动拍照 (YOLO mask + depth + K) → 暂停让你上传AutoDL跑FP → 粘贴矩阵
  ③ rotate_to_align_and_move (旋转对齐+工具系平移)
  ④ 自动拍照 → 暂停让你上传FP → 粘贴矩阵
  ⑤ rotate_to_align_and_move (精确定位)
  ⑥ 到达抓取位
"""
import sys, os, time, math, glob, datetime
import cv2, numpy as np
import pyrealsense2 as rs

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
POSE_FILE = os.path.join(SCRIPT_DIR, "..", "Foundationpose", "demo_data", "obj_OBJ", "ob_in_cam", "position1.txt")
POSE_FILE = os.path.normpath(POSE_FILE)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "Foundationpose"))
from Robot import RPC
from ultralytics import YOLO

# ── 初始拍照位 (学长的硬编码) ──
INITIAL_JOINTS = [-85.820, -64.652, -122.536, -22.587, 88.657, -88.286]

# ── 机械臂长度 ──
L = 150  # mm

W, H = 640, 480
OUTPUT_BASE = os.path.join(SCRIPT_DIR, "captured_cloud")


def check_robot(robot):
    for _ in range(30):
        ret = robot.GetActualJointPosDegree()
        if ret[0] == 0 and not all(abs(j) < 0.001 for j in ret[1]):
            return True
        time.sleep(0.5)
    print("WARNING: 机器人状态异常，继续尝试...")
    return False


def initial_position(robot):
    print(f"\n[1] MoveJ → 初始拍照位")
    print(f"    Joints: {[f'{j:.2f}' for j in INITIAL_JOINTS]}")
    robot.MoveJ(INITIAL_JOINTS, 0, 0)
    time.sleep(5)
    print("    到位。")


def auto_capture(label=""):
    """自动拍照: RealSense + YOLO → 保存 color/depth/K/mask/photo_joints"""
    print(f"\n{'='*50}")
    print(f"  📷 {label}")
    print(f"{'='*50}")

    # 相机
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30)
    prof = pipe.start(cfg)
    align = rs.align(rs.stream.color)
    scale = prof.get_device().first_depth_sensor().get_depth_scale()

    # 等流稳定
    for _ in range(30):
        pipe.wait_for_frames()

    # YOLO
    model = YOLO(os.path.join(SCRIPT_DIR, "best.pt"))

    # 机器人关节角
    robot = RPC("192.168.58.2")

    # 输出目录
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(OUTPUT_BASE, ts)
    os.makedirs(out_dir, exist_ok=True)

    best_conf = 0
    best_mask = None
    best_color = None
    best_depth = None
    best_depth_raw = None

    # 拍 10 帧取最佳 YOLO 检测
    print("   拍照中 (10帧取最佳)...")
    for frame_idx in range(10):
        frames = pipe.wait_for_frames()
        aligned = align.process(frames)
        cf = aligned.get_color_frame()
        df = aligned.get_depth_frame()
        if not cf or not df:
            continue

        color = np.asanyarray(cf.get_data())
        depth_raw = np.asanyarray(df.get_data())

        results = model(color, conf=0.25, iou=0.45, retina_masks=True, verbose=False)
        for r in results:
            if r.masks is not None and len(r.masks) > 0:
                for i, mt in enumerate(r.masks.data):
                    mask = (mt.cpu().numpy() > 0.5).astype(np.uint8)
                    conf_val = r.boxes[i].conf.item() if r.boxes else 1.0
                    if conf_val > best_conf:
                        best_conf = conf_val
                        best_mask = mask
                        best_color = color.copy()
                        best_depth = depth_raw.copy()
                        best_depth_raw = depth_raw.copy()

    pipe.stop()

    if best_mask is None:
        print("    ❌ YOLO 未检测到物体!")
        return None

    print(f"    ✅ 检测到物体, conf={best_conf:.3f}")

    # 保存
    cv2.imwrite(os.path.join(out_dir, "color.png"), best_color)
    depth_m = best_depth_raw.astype(np.float32) * scale
    np.save(os.path.join(out_dir, "depth.npy"), depth_m)
    cv2.imwrite(os.path.join(out_dir, "mask_object1.png"), (best_mask * 255).astype(np.uint8))

    intr = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]])
    np.savetxt(os.path.join(out_dir, "camera_K.txt"), K, fmt="%.6f")

    # 关节角
    for attempt in range(5):
        try:
            ret_j = robot.GetActualJointPosDegree()
            if ret_j[0] == 0 and not all(abs(j) < 0.001 for j in ret_j[1]):
                joints = list(ret_j[1])
                np.savetxt(os.path.join(out_dir, "photo_joints.txt"), joints, fmt="%.4f")
                print(f"    Joints: {[f'{j:.2f}' for j in joints]}")
                break
        except:
            pass
        time.sleep(0.5)
    else:
        print("    ⚠️ 关节角读取失败，请在示教器上手动记录")

    print(f"    Saved: {out_dir}")
    return out_dir


def manual_input_pose():
    """手动粘贴4x4矩阵"""
    print("\n    把 AutoDL 返回的 4x4 矩阵粘贴到这里 (4行):")
    lines = []
    for i in range(4):
        try:
            line = input(f"    行{i+1}: ")
            lines.append(line)
        except EOFError:
            break
    data = []
    for line in lines:
        nums = [float(x) for x in line.strip().split()]
        if len(nums) >= 4:
            data.append(nums[:4])
    if len(data) < 4:
        print("    矩阵解析失败!")
        return None
    pose = np.array(data)
    os.makedirs(os.path.dirname(POSE_FILE), exist_ok=True)
    np.savetxt(POSE_FILE, pose, fmt="%.6f")
    return pose


def get_pose_from_matrix(pose):
    xyz = pose[:3, 3] * 1000
    r11, r12 = pose[0, 0], pose[0, 1]
    rz = math.degrees(math.atan2(-r12, r11))
    return xyz, rz


def rotate_to_align_and_move(robot):
    """学长原版: 旋转对齐 J0 + 工具坐标系(ref=4)平移"""
    pose = np.loadtxt(POSE_FILE)
    camera_xyz, target_rz = get_pose_from_matrix(pose)
    print(f"\n    FP结果: X={camera_xyz[0]:.1f} Y={camera_xyz[1]:.1f} Z={camera_xyz[2]:.1f} mm  rz={target_rz:.2f}")

    # 旋转对齐
    if abs(target_rz) > 90:
        target_rz = target_rz - 180 if target_rz > 90 else target_rz + 180
        print(f"    旋转角翻转: rz={target_rz:.2f}")

    check_robot(robot)
    robot_pos = robot.GetActualJointPosDegree()[1]
    target_pose = list(robot_pos)
    target_pose[0] = target_pose[0] + target_rz
    print(f"    MoveJ 旋转: J0 {robot_pos[0]:.2f} → {target_pose[0]:.2f} (d={target_rz:.1f})")
    robot.MoveJ(target_pose, 0, 0)
    time.sleep(6)

    # 工具坐标系平移
    rz_rad = target_rz * math.pi / 180

    if abs(target_rz) < 5:
        # rz很小 = 已正对物体, 直接用FP距离做工具系平移
        # 实测映射: camX→-toolY, camY→-toolX, camZ→toolZ
        tx = -camera_xyz[1] * 0.85   # camY(mm下图) → tool X+(上)
        ty = -camera_xyz[0] * 0.85   # camX(mm右图) → tool Y+(左)
        tz =  camera_xyz[2] * 0.85   # camZ(mm深) → tool Z+(前)
        print(f"    rz<5° 直接平移: X={tx:.0f}mm Y={ty:.0f}mm Z={tz:.0f}mm")
        if abs(tx) > 3:
            d = 1 if tx > 0 else 0
            print(f"      X: {abs(tx):.0f}mm dir={d}")
            check_robot(robot)
            robot.StartJOG(4, 1, d, min(abs(tx), 300), 20.0, 30.0)
            time.sleep(6)
        if abs(ty) > 3:
            d = 1 if ty > 0 else 0
            print(f"      Y: {abs(ty):.0f}mm dir={d}")
            check_robot(robot)
            robot.StartJOG(4, 2, d, min(abs(ty), 300), 20.0, 30.0)
            time.sleep(6)
        if abs(tz) > 3:
            d = 1 if tz > 0 else 0
            print(f"      Z: {abs(tz):.0f}mm dir={d}")
            check_robot(robot)
            robot.StartJOG(4, 3, d, min(abs(tz), 300), 20.0, 20.0)
            time.sleep(6)
        return target_rz

    if rz_rad > 0:
        dir_y = 1
        y = abs(math.sin(abs(rz_rad)) / math.cos(abs(rz_rad)/2) * (L + camera_xyz[2])) - rz_rad
    else:
        dir_y = 0
        y = abs(math.sin(abs(rz_rad)) / math.cos(abs(rz_rad)/2) * (L + camera_xyz[2]))

    print(f"    工具系平移: Y={y:.1f}mm dir={dir_y}")

    if y > 200:
        z_dis = 30 * (y / 125)
        x_dis = 18 * (y / 125)
        if y > 300:
            check_robot(robot)
            robot.StartJOG(4, 2, dir_y, 280, 20.0, 30.0); time.sleep(6)
            check_robot(robot)
            robot.StartJOG(4, 3, 0, z_dis, 20.0, 20.0); time.sleep(6)
            check_robot(robot)
            robot.StartJOG(4, 1, 0, x_dis, 20.0, 30.0); time.sleep(6)
            remaining = y - 280
            if abs(remaining) > 1:
                check_robot(robot)
                robot.StartJOG(4, 2, dir_y, remaining, 20.0, 10.0); time.sleep(6)
        else:
            check_robot(robot)
            robot.StartJOG(4, 2, dir_y, y, 20.0, 30.0); time.sleep(6)
            check_robot(robot)
            robot.StartJOG(4, 3, 0, z_dis, 20.0, 20.0); time.sleep(6)
            check_robot(robot)
            robot.StartJOG(4, 1, 0, x_dis, 20.0, 30.0); time.sleep(6)
    else:
        check_robot(robot)
        robot.StartJOG(4, 2, dir_y, y, 20.0, 30.0); time.sleep(6)
        check_robot(robot)
        robot.StartJOG(4, 3, 0, 30, 20.0, 20.0); time.sleep(6)
        check_robot(robot)
        robot.StartJOG(4, 1, 0, 18, 20.0, 30.0); time.sleep(6)

    return target_rz


# ═══════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  学长方案: MoveJ初始位 → 两次定位 → 工具系平移")
    print("=" * 60)

    # ── 连接 ──
    print("\n连接机器人...")
    robot = RPC("192.168.58.2")
    robot.ResetAllError(); time.sleep(0.5)
    robot.Mode(1); time.sleep(0.5)
    robot.RobotEnable(1); time.sleep(1)
    print("连接成功。")

    # ═══ ① MoveJ → 初始拍照位 ═══
    initial_position(robot)

    # ═══ ② 第一次拍照 (自动) ═══
    data_dir = auto_capture("第一次拍照 (初始位)")
    if data_dir is None: return

    print(f"\n{'='*50}")
    print(f"  上传到 AutoDL:")
    print(f"    scp -r {data_dir} root@<ip>:/root/")
    print(f"  AutoDL:")
    print(f"    conda activate fp310")
    print(f"    python demo_cloud_pose.py --data /root/{os.path.basename(data_dir)} --mesh /root/obj_OBJ/mesh3/plane2.obj")
    print(f"{'='*50}")

    pose1 = manual_input_pose()
    if pose1 is None: return

    # ═══ ③ 旋转对齐 + 平移 ═══
    rz = rotate_to_align_and_move(robot)

    # ═══ ④ 第二次拍照 (自动) ═══
    data_dir2 = auto_capture("第二次拍照 (靠近后)")
    if data_dir2 is None: return

    print(f"\n{'='*50}")
    print(f"  上传到 AutoDL:")
    print(f"    scp -r {data_dir2} root@<ip>:/root/")
    print(f"  AutoDL:")
    print(f"    conda activate fp310")
    print(f"    python demo_cloud_pose.py --data /root/{os.path.basename(data_dir2)} --mesh /root/obj_OBJ/mesh3/plane2.obj")
    print(f"{'='*50}")

    pose2 = manual_input_pose()
    if pose2 is None: return

    # ═══ ⑤ 精确移动 ═══
    rz2 = rotate_to_align_and_move(robot)

    print(f"\n{'='*60}")
    print(f"  DONE! 机械臂应在抓取位置 (rz={rz2:.1f})")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
