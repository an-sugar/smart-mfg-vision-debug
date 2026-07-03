"""
全链条联调 - 拖动示教版
======================
用拖动示教代替自动 MoveJ，绕过 FR3 安全锁。

流程:
  1. 进入拖动示教 -> 你手拖机器人到拍照位置
  2. 退出拖动示教
  3. 拍照 + YOLO 分割
  4. 打包数据 -> 等会上传 AutoDL
  5. 循环

用法:
  python demo_drag_capture_cloud.py
"""

import os, sys, time, json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FP_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "Foundationpose")
sys.path.insert(0, FP_DIR)

import cv2
import numpy as np

# --- Config ---
ROBOT_IP = "192.168.58.2"
MODEL_PATH = os.path.join(SCRIPT_DIR, "best.pt")
OUTPUT_BASE = os.path.join(SCRIPT_DIR, "captured_cloud")

# --- YOLO ---
CLASS_NAMES = {0: "object1", 1: "object2", 2: "object3"}


def robot_drag(enable=True):
    """Enter/exit drag teach mode"""
    from Robot import RPC
    r = RPC(ROBOT_IP)
    r.ResetAllError()
    time.sleep(0.5)
    r.Mode(1)
    time.sleep(0.5)
    r.RobotEnable(1)
    time.sleep(0.5)
    err = r.DragTeachSwitch(1 if enable else 0)
    state = r.IsInDragTeach()
    drag_ok = (state[0] == 0 and state[1] == 1)
    return r, err == 0 and drag_ok


def capture_rgbd():
    """Take one RGB-D frame, return (color, depth_m, K)"""
    import pyrealsense2 as rs
    W, H = 640, 480
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30)
    prof = pipe.start(cfg)
    align = rs.align(rs.stream.color)
    scale = prof.get_device().first_depth_sensor().get_depth_scale()
    intr = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]])

    for _ in range(30):
        pipe.wait_for_frames()

    frames = align.process(pipe.wait_for_frames())
    color = np.asanyarray(frames.get_color_frame().get_data())
    depth = np.asanyarray(frames.get_depth_frame().get_data()).astype(np.float32) * scale
    pipe.stop()
    return color, depth, K


def yolo_segment(model, color):
    """Run YOLO, return list of (name, index, conf, mask_bin)"""
    results = model(color, retina_masks=True, conf=0.25, iou=0.45,
                    show_boxes=False, show_conf=False)

    masks = []
    for result in results:
        if result.masks is None:
            continue
        for i, mt in enumerate(result.masks.data):
            m = (mt.cpu().numpy() > 0.5).astype(np.uint8) * 255
            cid = int(result.boxes[i].cls.item()) if result.boxes else 0
            cf = result.boxes[i].conf.item() if result.boxes else 1.0
            masks.append((CLASS_NAMES.get(cid, f"cls{cid}"), i + 1, cf, m))
    return masks


def save_pkg(task, color, depth, K, masks):
    """Save capture package"""
    d = os.path.join(OUTPUT_BASE, task)
    os.makedirs(d, exist_ok=True)
    cv2.imwrite(os.path.join(d, "color.png"), color)
    np.save(os.path.join(d, "depth.npy"), depth)
    np.savetxt(os.path.join(d, "camera_K.txt"), K, fmt="%.6f",
               header=f"fx={K[0,0]:.3f} fy={K[1,1]:.3f} cx={K[0,2]:.3f} cy={K[1,2]:.3f}")
    for name, idx, conf, mask in masks:
        cv2.imwrite(os.path.join(d, f"mask_{name}.png"), mask)
    with open(os.path.join(d, "info.json"), "w") as f:
        json.dump({
            "task": task,
            "objects": [{"name": n, "conf": float(c)} for n, _, c, _ in masks],
            "K": K.tolist(),
        }, f, indent=2)
    print(f"  Saved: {d}")
    return d


def main():
    print("=" * 55)
    print("  Full Pipeline - Drag Teach Version")
    print("=" * 55)

    from ultralytics import YOLO
    model = YOLO(MODEL_PATH)
    print(f"YOLO loaded: {MODEL_PATH}")

    round_num = 1
    while True:
        print(f"\n{'=' * 55}")
        print(f"  Round {round_num}")
        print(f"{'=' * 55}")

        # Step 1: Drag teach ON
        print("\n[1] Entering drag teach mode...")
        robot, ok = robot_drag(enable=True)
        if not ok:
            print("  ERROR: Cannot enter drag teach. Exiting.")
            break
        print("  Drag teach ON - drag the robot to your desired pose")
        print("  You have 15 seconds to position the robot...")
        time.sleep(15)

        # Step 2: Drag teach OFF
        print("\n[2] Exiting drag teach...")
        robot.DragTeachSwitch(0)
        time.sleep(1)
        print("  Drag teach OFF")

        # Step 3: Capture
        print("\n[3] Capturing RGB-D...")
        color, depth, K = capture_rgbd()
        print(f"  Color: {color.shape}, Depth: {depth.shape}")

        # Step 4: YOLO
        print("\n[4] YOLO segmentation...")
        masks = yolo_segment(model, color)
        if not masks:
            print("  WARNING: No objects detected!")
        for n, _, c, _ in masks:
            print(f"  {n}: conf={c:.3f}")

        # Step 5: Save
        task_name = time.strftime(f"round{round_num}_%H%M%S")
        print(f"\n[5] Saving package: {task_name}")
        pkg_dir = save_pkg(task_name, color, depth, K, masks)

        # Step 6: Preview
        vis = color.copy()
        for name, _, _, mask in masks:
            color_map = {0: (0, 255, 0), 1: (255, 0, 0), 2: (0, 0, 255)}
            cid = [k for k, v in CLASS_NAMES.items() if v == name]
            cid = cid[0] if cid else 0
            clr = color_map.get(cid, (255, 255, 255))
            overlay = np.zeros_like(vis)
            overlay[:, :, 0] = clr[0]; overlay[:, :, 1] = clr[1]; overlay[:, :, 2] = clr[2]
            vis = np.where(mask[:, :, None] > 127,
                          (vis * 0.6 + overlay * 0.4).astype(np.uint8), vis)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, contours, -1, clr, 2)

        preview_path = os.path.join(pkg_dir, "preview.png")
        cv2.imwrite(preview_path, vis)
        print(f"\n  Preview saved: {preview_path}")

        print("  Done this round.")
        break  # Just one round then exit

    print("\n" + "=" * 55)
    print("  Done.")
    print(f"  Packages in: {OUTPUT_BASE}")
    print(f"  Upload to AutoDL, then run:")
    print(f"    python demo_cloud_pose.py --data /root/.../ --mesh /root/obj_OBJ/mesh3/plane2.obj")
    print("=" * 55)


if __name__ == "__main__":
    main()
