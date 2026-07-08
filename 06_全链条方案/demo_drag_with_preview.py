"""
拖动示教 + 实时预览 + 空格拍照验证
==================================
拖动示教过程中, 相机实时画面, 按空格拍照+YOLO+深度预览。
确认检测到物体后, 再退出拖动示教保存数据。

用法: python demo_drag_with_preview.py
"""

import os, sys, time, cv2, numpy as np
import pyrealsense2 as rs

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FP_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "Foundationpose")
sys.path.insert(0, FP_DIR)

from Robot import RPC

W, H = 640, 480

# ── 1. 初始化 ──
print("=" * 50)
print("  拖动示教 + 实时预览 + 空格拍照")
print("=" * 50)

# 相机
pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)
cfg.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30)
prof = pipe.start(cfg)
align = rs.align(rs.stream.color)
scale = prof.get_device().first_depth_sensor().get_depth_scale()

from ultralytics import YOLO
model = YOLO(os.path.join(SCRIPT_DIR, "best.pt"))

# 机器人
print("连接机器人...")
robot = RPC("192.168.58.2")
robot.ResetAllError(); time.sleep(0.5)
robot.Mode(1); time.sleep(0.5)
robot.RobotEnable(1); time.sleep(1)
robot.DragTeachSwitch(1)
print("Drag teach ON — 拖动中可实时查看画面")

# 创建输出目录
import datetime
ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
out_dir = os.path.join(SCRIPT_DIR, "captured_cloud", ts)
os.makedirs(out_dir, exist_ok=True)

print("\n操作:")
print("  拖动示教已开启 — 用手移动机器人")
print("  画面实时显示 — 彩色图 + YOLO检测框")
print("  空格键 = 拍照 + 显示 Mask + 深度图")
print("  ESC    = 退出拖动示教并保存")
print()

try:
    saved = False
    while True:
        # 获取对齐帧
        frames = pipe.wait_for_frames()
        aligned = align.process(frames)
        cf = aligned.get_color_frame()
        df = aligned.get_depth_frame()
        if not cf or not df: continue

        color = np.asanyarray(cf.get_data())
        depth_raw = np.asanyarray(df.get_data())
        depth_m = depth_raw.astype(np.float32) * scale

        # YOLO 实时检测
        results = model(color, conf=0.25, iou=0.45, retina_masks=True,
                        show_boxes=True, show_conf=True, show_labels=True,
                        verbose=False)

        # 叠加检测框
        display = color.copy()
        detected = False
        for r in results:
            if r.masks is not None and len(r.masks) > 0:
                detected = True
                for i, mt in enumerate(r.masks.data):
                    mask = (mt.cpu().numpy() > 0.5).astype(np.uint8)
                    cid = int(r.boxes[i].cls.item()) if r.boxes else 0
                    cf_val = r.boxes[i].conf.item() if r.boxes else 1.0
                    name = {0: "drone", 1: "obj2", 2: "obj3"}.get(cid, f"cls{cid}")

                    # 绿色mask叠加
                    overlay = np.zeros_like(display)
                    overlay[:,:,1] = mask * 255
                    display = np.where(mask[:,:,None] > 0,
                                      (display * 0.6 + overlay * 0.4).astype(np.uint8),
                                      display)
                    # 轮廓
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(display, contours, -1, (0,255,0), 2)
                    # 标签
                    if contours:
                        M = cv2.moments(max(contours, key=cv2.contourArea))
                        if M["m00"] > 0:
                            cx, cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
                            cv2.putText(display, f"{name} {cf_val:.2f}",
                                        (cx-30, cy-10), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.6, (0,255,0), 2)

        # 状态栏
        status = "DRAG TEACHING"
        if detected:
            status += " | DRONE DETECTED!"
        cv2.putText(display, status, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0) if detected else (0, 0, 255), 2)
        cv2.putText(display, "SPACE=Capture  ESC=Save&Exit", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.imshow("Live Preview", display)

        key = cv2.waitKey(33) & 0xFF  # ~30fps

        if key == 32 and detected:  # 空格 + 检测到了 → 预览
            print(f"\n  DRONE detected! conf={cf_val:.3f}")

            # 深度可视化
            depth_vis = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_raw, alpha=0.03), cv2.COLORMAP_JET)

            # 保存
            cv2.imwrite(os.path.join(out_dir, "color.png"), color)
            np.save(os.path.join(out_dir, "depth.npy"), depth_m)
            for i, mt in enumerate(r.masks.data):
                m = (mt.cpu().numpy() > 0.5).astype(np.uint8) * 255
                cv2.imwrite(os.path.join(out_dir, f"mask_obj{i+1}.png"), m)
                print(f"  Saved mask_obj{i+1}.png")

            # K 矩阵
            intr = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
            K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]])
            np.savetxt(os.path.join(out_dir, "camera_K.txt"), K, fmt="%.6f")
            print(f"  K: fx={K[0,0]:.1f} fy={K[1,1]:.1f}")

            saved = True
            print(f"  Saved to: {out_dir}")

            # 显示预览 (左侧原图, 右侧mask叠加+深度)
            preview = np.hstack((color, display))
            depth_small = cv2.resize(depth_vis, (display.shape[1], display.shape[0]))
            row2 = np.hstack((np.zeros_like(display), depth_small))
            cv2.putText(row2, "DEPTH MAP (JET colormap)", (display.shape[1]//2-100, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            full = np.vstack((preview, row2))
            cv2.imshow("CAPTURED - Press any key to continue", full)
            cv2.waitKey(0)
            cv2.destroyWindow("CAPTURED - Press any key to continue")

        elif key == 27:  # ESC → 退出
            break

finally:
    robot.DragTeachSwitch(0)
    time.sleep(0.5)
    pipe.stop()
    cv2.destroyAllWindows()
    print("\nDrag teach OFF. Camera closed.")

    # 退出拖动示教后再尝试读关节角（实时端口更稳定）
    if saved:
        print("Reading joints after drag teach...")
        for attempt in range(5):
            try:
                ret_j = robot.GetActualJointPosDegree()
                if ret_j[0] == 0 and not all(abs(j) < 0.001 for j in ret_j[1]):
                    joints = list(ret_j[1])
                    np.savetxt(os.path.join(out_dir, "photo_joints.txt"), joints, fmt="%.4f")
                    print(f"  Joints saved: {[f'{j:.2f}' for j in joints]}")
                    break
                else:
                    print(f"  Attempt {attempt+1}: all zeros, retrying...")
                    time.sleep(0.5)
            except Exception as e:
                print(f"  Attempt {attempt+1}: {e}")
                time.sleep(0.5)
        else:
            print("  WARNING: Could not read joints after 5 attempts.")

if saved:
    print(f"\nData saved to: {out_dir}")
    print(f"\nUpload to AutoDL, then run:")
    print(f"  python demo_cloud_pose.py --data /root/{ts} --mesh /root/obj_OBJ/mesh3/plane2.obj")
    print(f"\nThen send me the 4x4 matrix.")
else:
    print("\nNo capture made.")
