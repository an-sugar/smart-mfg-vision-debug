"""快速拍照+YOLO — 修复版"""
import os, sys, time
import cv2, numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

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

# 多等几帧让深度流稳定
color, depth_m = None, None
for _ in range(60):
    frames = pipe.wait_for_frames()
    aligned = align.process(frames)
    cf = aligned.get_color_frame()
    df = aligned.get_depth_frame()
    if cf and df:
        color = np.asanyarray(cf.get_data())
        depth_m = np.asanyarray(df.get_data()).astype(np.float32) * scale
        break
pipe.stop()

if color is None:
    print("ERROR: cant get frame")
    exit(1)

print(f"Color: {color.shape}  Depth: {depth_m.shape}  K: fx={K[0,0]:.1f}")

# YOLO
from ultralytics import YOLO
model = YOLO(os.path.join(SCRIPT_DIR, "best.pt"))
results = model(color, retina_masks=True, conf=0.25, iou=0.45,
                show_boxes=False, show_conf=False, verbose=False)

ts = time.strftime("%Y%m%d_%H%M%S")
d = os.path.join(SCRIPT_DIR, "captured_cloud", ts)
os.makedirs(d, exist_ok=True)
cv2.imwrite(os.path.join(d, "color.png"), color)
np.save(os.path.join(d, "depth.npy"), depth_m)
np.savetxt(os.path.join(d, "camera_K.txt"), K, fmt="%.6f")

detected = False
for r in results:
    if r.masks is not None:
        for i, mt in enumerate(r.masks.data):
            m = (mt.cpu().numpy() > 0.5).astype(np.uint8) * 255
            cid = int(r.boxes[i].cls.item()) if r.boxes else 0
            cf = r.boxes[i].conf.item() if r.boxes else 1.0
            name = {0: "object1", 1: "object2", 2: "object3"}.get(cid, f"cls{cid}")
            cv2.imwrite(os.path.join(d, f"mask_{name}.png"), m)
            print(f"  {name}  conf={cf:.3f}")
            detected = True

if not detected:
    print("  No objects detected")

print(f"\nSaved: {d}")
print(f"\nUpload to AutoDL, then run:")
print(f"  python demo_cloud_pose.py --data /root/{ts} --mesh /root/obj_OBJ/mesh3/plane2.obj")
