"""
全链条一键脚本 (最终版)
========================
① 拖动示教 + 实时预览 → 空格拍照
② 提示上传 → 等FP结果
③ 粘贴位姿矩阵 → 自动转换 + 机器人移动

用法:
  python full_pipeline_final.py
"""

import os, sys, time, math
import cv2, numpy as np
import pyrealsense2 as rs

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FP_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "Foundationpose")
sys.path.insert(0, FP_DIR)

# ── 配置 ──
W, H = 640, 480
PHOTO_JOINTS = [-85.820, -64.652, -122.536, -22.587, 88.657, -88.286]
GRASP_JOINTS = [-84.438, -129.095, -69.760, -21.553, 90.009, -97.098]
CAM_X, CAM_Y, CAM_Z = 0, 40, -90
POSE_FILE = os.path.join(FP_DIR, "demo_data/obj_OBJ/ob_in_cam/position1.txt")

OUTPUT_BASE = os.path.join(SCRIPT_DIR, "captured_cloud")


# ═══════════════════════════════════════════════════════════
# Phase 1: 拖动示教 + 实时预览 + 空格拍照
# ═══════════════════════════════════════════════════════════
def phase1_capture():
    from Robot import RPC
    from ultralytics import YOLO

    print("=" * 60)
    print("  Phase 1: 拖动示教 + 实时预览 + 拍照")
    print("=" * 60)

    # 相机
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30)
    prof = pipe.start(cfg)
    align = rs.align(rs.stream.color)
    scale = prof.get_device().first_depth_sensor().get_depth_scale()
    intr = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]])

    # YOLO
    model = YOLO(os.path.join(SCRIPT_DIR, "best.pt"))

    # 机器人拖动示教
    robot = RPC("192.168.58.2")
    robot.ResetAllError(); time.sleep(0.5)
    robot.Mode(1); time.sleep(0.5)
    robot.RobotEnable(1); time.sleep(1)
    robot.DragTeachSwitch(1)

    print("\n操作:")
    print("  拖动示教中 — 实时画面")
    print("  空格键 = 拍照并保存")
    print("  ESC    = 退出\n")

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(OUTPUT_BASE, ts)
    saved = False

    try:
        while True:
            frames = pipe.wait_for_frames()
            aligned = align.process(frames)
            cf = aligned.get_color_frame()
            df = aligned.get_depth_frame()
            if not cf or not df: continue

            color = np.asanyarray(cf.get_data())
            depth_raw = np.asanyarray(df.get_data())

            # YOLO
            results = model(color, conf=0.25, iou=0.45, retina_masks=True,
                            show_boxes=False, show_conf=False, verbose=False)

            display = color.copy()
            detected = False
            best_conf = 0
            masks_data = []

            for r in results:
                if r.masks is not None and len(r.masks) > 0:
                    detected = True
                    for i, mt in enumerate(r.masks.data):
                        mask = (mt.cpu().numpy() > 0.5).astype(np.uint8)
                        conf_val = r.boxes[i].conf.item() if r.boxes else 1.0
                        best_conf = max(best_conf, conf_val)
                        masks_data.append((mask, int(r.boxes[i].cls.item()) if r.boxes else 0, conf_val))

                        # 叠加
                        overlay = np.zeros_like(display)
                        overlay[:, :, 1] = mask * 255
                        display = np.where(mask[:, :, None] > 0,
                                          (display * 0.6 + overlay * 0.4).astype(np.uint8), display)
                        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(display, contours, -1, (0, 255, 0), 2)

            # 状态
            status = "DRAG TEACHING"
            if detected:
                status += f" | DRONE {best_conf:.2f}!"
            cv2.putText(display, status, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0) if detected else (0, 0, 255), 2)
            cv2.putText(display, "SPACE=Capture  ESC=Exit", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            cv2.imshow("Live | Drag Teach | SPACE=Capture ESC=Exit", display)
            key = cv2.waitKey(33) & 0xFF

            if key == 32 and detected:  # 拍照!
                print(f"\n  DRONE detected! conf={best_conf:.3f}")

                os.makedirs(out_dir, exist_ok=True)

                # 保存彩色图
                cv2.imwrite(os.path.join(out_dir, "color.png"), color)

                # 保存深度图
                depth_m = depth_raw.astype(np.float32) * scale
                np.save(os.path.join(out_dir, "depth.npy"), depth_m)

                # 保存K矩阵
                np.savetxt(os.path.join(out_dir, "camera_K.txt"), K, fmt="%.6f")

                # 保存mask
                for i, (mask, cid, cf) in enumerate(masks_data):
                    cv2.imwrite(os.path.join(out_dir, f"mask_object1.png"),
                               (mask * 255).astype(np.uint8))
                    print(f"  mask_object1.png | drine | conf={cf:.3f}")

                saved = True
                print(f"  Saved: {out_dir}")

                # 预览
                depth_vis = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_raw, alpha=0.03), cv2.COLORMAP_JET)
                depth_vis = cv2.resize(depth_vis, (display.shape[1], display.shape[0]))
                cv2.putText(depth_vis, "DEPTH", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                preview = np.hstack((display, depth_vis))
                cv2.putText(preview, "YOLO+MASK", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("CAPTURED | Press any key", preview)
                cv2.waitKey(0)
                cv2.destroyWindow("CAPTURED | Press any key")

            elif key == 27:  # ESC
                break

    finally:
        robot.DragTeachSwitch(0)
        pipe.stop()
        cv2.destroyAllWindows()

    print("\nDrag teach OFF.")
    return out_dir if saved else None


# ═══════════════════════════════════════════════════════════
# Phase 2: 解析FP结果 + 机器人移动
# ═══════════════════════════════════════════════════════════
def phase2_move():
    from Robot import RPC

    print(f"\n{'=' * 60}")
    print(f"  Phase 2: FP -> Robot Move")
    print(f"{'=' * 60}")

    print("\n请把 AutoDL 返回的 4x4 位姿矩阵粘贴到这里 (每行4个数字, 共4行):")
    print("  Ctrl+Z 结束输入\n")

    lines = []
    try:
        while len(lines) < 4:
            line = input()
            lines.append(line)
    except EOFError:
        pass

    # 解析矩阵
    data = []
    for line in lines:
        nums = [float(x) for x in line.strip().split()]
        if len(nums) >= 4:
            data.append(nums[:4])

    if len(data) < 4:
        print("Invalid matrix. Using last saved pose.")
        if not os.path.exists(POSE_FILE):
            print("No pose file either. Exiting.")
            return
        pose = np.loadtxt(POSE_FILE)
    else:
        pose = np.array(data)
        np.savetxt(POSE_FILE, pose, fmt="%.6f")
        print(f"Pose saved.")

    cx, cy, cz = pose[0, 3] * 1000, pose[1, 3] * 1000, pose[2, 3] * 1000
    print(f"\nFP: 物体距相机 X={cx:.1f} Y={cy:.1f} Z={cz:.1f} mm")

    # FK + 转换
    robot = RPC("192.168.58.2")
    robot.ResetAllError(); time.sleep(0.5)
    robot.Mode(1); time.sleep(0.5)
    robot.RobotEnable(1); time.sleep(1)

    ret = robot.GetForwardKin(list(PHOTO_JOINTS))
    if ret[0] != 0:
        print("FK failed")
        return
    fk = list(ret[1])
    print(f"拍照位 FK: X={fk[0]:.0f} Y={fk[1]:.0f} Z={fk[2]:.0f}")

    # 物体在法兰系
    obj_f_x = CAM_X + cx
    obj_f_y = CAM_Y - cz
    obj_f_z = -CAM_Z - cy

    # 物体在基坐标
    obj_base_x = fk[0] + obj_f_x
    obj_base_y = fk[1] + obj_f_y
    obj_base_z = fk[2] + obj_f_z
    print(f"物体基坐标: X={obj_base_x:.0f} Y={obj_base_y:.0f} Z={obj_base_z:.0f}")

    # 抓取位FK
    ret2 = robot.GetForwardKin(list(GRASP_JOINTS))
    if ret2[0] != 0:
        print("Grasp FK failed")
        return
    gk = list(ret2[1])
    print(f"抓取位 FK: X={gk[0]:.0f} Y={gk[1]:.0f} Z={gk[2]:.0f}")

    dx = obj_base_x - gk[0]
    dy = obj_base_y - gk[1]
    dz = obj_base_z - gk[2]
    print(f"\n需要移动: dX={dx:.0f} dY={dy:.0f} dZ={dz:.0f} mm")

    # 执行移动
    for axis, dist, nb in [("X", dx, 1), ("Y", dy, 2)]:
        if abs(dist) > 2:
            d = 1 if dist > 0 else 0
            print(f"  {axis}: {abs(dist):.0f}mm dir={d}")
            robot.StartJOG(2, nb, d, min(abs(dist), 300), 20.0, 30.0)
            time.sleep(4)

    if abs(dz) > 5:
        d = 0 if dz > 0 else 1
        print(f"  Z: {abs(dz):.0f}mm dir={d}")
        robot.StartJOG(2, 3, d, min(abs(dz), 200), 20.0, 20.0)
        time.sleep(4)

    print(f"\n{'=' * 60}")
    print(f"  DONE! Robot should be at grasp position")
    print(f"{'=' * 60}")


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  FULL PIPELINE FINAL")
    print("  拖动示教 → 实时预览 → 拍照 → FP → 移动")
    print("=" * 60)

    # Phase 1
    data_dir = phase1_capture()
    if not data_dir:
        print("No capture. Exiting.")
        return

    # Upload hint
    print(f"\n{'=' * 60}")
    print(f"  上传到 AutoDL:")
    print(f"    {data_dir}")
    print(f"  AutoDL 终端:")
    print(f"    conda activate fp310")
    print(f"    python demo_cloud_pose.py --data /root/{os.path.basename(data_dir)} --mesh /root/obj_OBJ/mesh3/plane2.obj")
    print(f"{'=' * 60}")

    # Phase 2
    phase2_move()


if __name__ == "__main__":
    main()
