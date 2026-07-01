# -*- coding: utf-8 -*-
"""
全链条联调 — Step 1: 本地采集脚本
==================================
拍照 + YOLO 分割 + 打包 → 上传 AutoDL → 位姿估计 → 返回结果

产出:
  captured/任务名/
  ├── color.png        RGB 彩色图
  ├── depth.npy        深度图(米)
  ├── mask_obj1.png    YOLO 分割 mask
  ├── mask_obj2.png    (如果有多个物体)
  ├── camera_K.txt     相机内参 (3×3)
  └── README.txt       采集信息

用法:
  # 相机拍照
  python demo_capture_for_cloud.py

  # 已有图片
  python demo_capture_for_cloud.py --image ./captures/xxx.jpg

依赖:
  conda activate vision  (你的 YOLO 环境)
"""

import argparse
import os
import sys
import time
import json
import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_BASE = os.path.join(SCRIPT_DIR, "captured_cloud")
os.makedirs(OUTPUT_BASE, exist_ok=True)

# ── YOLO 配置 ──
MODEL_PATH = os.path.join(SCRIPT_DIR, "best.pt")
CLASS_NAMES = {0: "object1", 1: "object2", 2: "object3"}


def capture_with_realsense():
    """RealSense 拍一张 RGB + Depth，返回 (color, depth_m, K_matrix)"""
    try:
        import pyrealsense2 as rs
    except ImportError:
        print("❌ pyrealsense2 未安装")
        sys.exit(1)

    W, H, FPS = 640, 480, 30
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)
    config.enable_stream(rs.stream.depth, W, H, rs.format.z16, FPS)
    profile = pipeline.start(config)

    align = rs.align(rs.stream.color)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

    # 内参
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]])

    print("\n📷 预热中...")
    for _ in range(30):
        pipeline.wait_for_frames()

    print("   按 空格键 拍照, ESC 退出\n")
    color, depth_m = None, None

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            cf = aligned.get_color_frame()
            df = aligned.get_depth_frame()
            if not cf or not df:
                continue
            preview = np.asanyarray(cf.get_data())
            cv2.putText(preview, "SPACE=Capture  ESC=Quit", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Capture for Cloud", preview)
            key = cv2.waitKey(1) & 0xFF
            if key == 32:
                color = np.asanyarray(cf.get_data())
                depth_raw = np.asanyarray(df.get_data())
                depth_m = depth_raw.astype(np.float32) * depth_scale
                print("📷 已拍照")
                break
            elif key == 27:
                print("❌ 取消")
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

    return color, depth_m, K


def run_yolo(model, color_image):
    """YOLO 分割 → 返回 mask 列表 [(name, mask_binary), ...]"""
    results = model(color_image, retina_masks=True, conf=0.25, iou=0.45,
                    show_boxes=False, show_conf=False, show_labels=True)

    masks_out = []
    for result in results:
        if result.masks is None or len(result.masks) == 0:
            print("  ⚠️ YOLO 未检测到物体")
            return []

        for i, mask_t in enumerate(result.masks.data):
            mask_bin = (mask_t.cpu().numpy() > 0.5).astype(np.uint8) * 255
            cls_id = int(result.boxes[i].cls.item()) if result.boxes is not None else 0
            conf = result.boxes[i].conf.item() if result.boxes is not None else 1.0
            name = CLASS_NAMES.get(cls_id, f"class_{cls_id}")
            masks_out.append((name, i + 1, conf, mask_bin))
            print(f"  ✅ {name} (obj{i+1}) | conf={conf:.3f}")
    return masks_out


def save_package(task_name, color, depth_m, K, masks):
    """保存一帧完整数据包"""
    task_dir = os.path.join(OUTPUT_BASE, task_name)
    os.makedirs(task_dir, exist_ok=True)

    # 保存 RGB
    cv2.imwrite(os.path.join(task_dir, "color.png"), color)
    print(f"  ✅ color.png ({color.shape[1]}×{color.shape[0]})")

    # 保存深度 (npy, 米为单位)
    np.save(os.path.join(task_dir, "depth.npy"), depth_m)
    print(f"  ✅ depth.npy ({depth_m.shape}, range: {depth_m.min():.3f}~{depth_m.max():.3f}m)")

    # 保存相机内参
    np.savetxt(os.path.join(task_dir, "camera_K.txt"), K, fmt="%.6f",
               header=f"fx fy cx cy\n{K[0,0]:.3f} {K[1,1]:.3f} {K[0,2]:.3f} {K[1,2]:.3f}")
    print(f"  ✅ camera_K.txt (fx={K[0,0]:.1f}, fy={K[1,1]:.1f})")

    # 保存 mask
    for name, idx, conf, mask in masks:
        p = os.path.join(task_dir, f"mask_{name}.png")
        cv2.imwrite(p, mask)
        print(f"  ✅ mask_{name}.png ({mask.shape}, conf={conf:.3f})")

    # 保存信息文件
    info = {
        "task": task_name,
        "num_objects": len(masks),
        "objects": [{"index": i, "name": n, "conf": float(c)} for n, i, c, _ in masks],
        "resolution": f"{color.shape[1]}×{color.shape[0]}",
        "K": K.tolist(),
    }
    with open(os.path.join(task_dir, "info.json"), "w") as f:
        json.dump(info, f, indent=2)

    # README
    readme = f"""# {task_name}

| 项目 | 值 |
|------|-----|
| 时间 | {task_name} |
| 分辨率 | {color.shape[1]}×{color.shape[0]} |
| 内参 K | fx={K[0,0]:.3f}, fy={K[1,1]:.3f}, cx={K[0,2]:.3f}, cy={K[1,2]:.3f} |
| 物体数 | {len(masks)} |
| 物体 | {', '.join(n for n,_,_,_ in masks)} |

## 在 AutoDL 上运行

```bash
# 把这个文件夹上传到 /root/autodl-tmp/
# 然后:
conda activate fp310
cd /root/autodl-tmp/{task_name}
python /root/demo_pose_offline.py \\
    --mesh /root/obj_OBJ/mesh_X/物体名.obj \\
    --data 此文件夹路径
```
"""
    with open(os.path.join(task_dir, "README.txt"), "w", encoding="utf-8") as f:
        f.write(readme)

    print(f"\n📦 数据包: {task_dir}")
    print(f"   ├── color.png")
    print(f"   ├── depth.npy")
    print(f"   ├── camera_K.txt")
    for n, _, _, _ in masks:
        print(f"   ├── mask_{n}.png")
    print(f"   ├── info.json")
    print(f"   └── README.txt")
    print(f"\n👉 把这个文件夹打包上传到 AutoDL 的 /root/autodl-tmp/")

    return task_dir


def main():
    print("=" * 55)
    print("  全链条联调 Step 1 — 本地采集")
    print("=" * 55)

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=None, help="用已有图片，不调相机")
    args = parser.parse_args()

    # ── 1. 获取图像 ──
    if args.image:
        color = cv2.imread(args.image)
        if color is None:
            print(f"❌ 无法读取: {args.image}")
            sys.exit(1)
        depth_m = np.zeros_like(color[:, :, 0], dtype=np.float32)  # 占位
        K = np.eye(3)  # 占位
        print(f"\n📷 已读取: {args.image}")
        print("   ⚠️ 无深度图 + 内参，仅供 YOLO 测试")
    else:
        color, depth_m, K = capture_with_realsense()
        if color is None:
            print("未拍照，退出。")
            sys.exit(0)

    # ── 2. YOLO 分割 ──
    print("\n🔍 YOLO 分割...")
    from ultralytics import YOLO
    model = YOLO(MODEL_PATH)
    masks = run_yolo(model, color)

    if not masks:
        print("❌ 未检测到物体！请确保比赛道具在相机视野内。")
        print("   仍会保存数据包（含 color + depth），可以手动标 mask。")
        # 继续保存，只是没有 mask

    # ── 3. 保存数据包 ──
    task_name = time.strftime("%Y%m%d_%H%M%S")
    print(f"\n💾 保存数据包: {task_name}")
    task_dir = save_package(task_name, color, depth_m, K, masks)

    # ── 4. 预览 ──
    vis = color.copy()
    for name, _, conf, mask in masks:
        cls_id = [k for k, v in CLASS_NAMES.items() if v == name][0]
        color_map = {0: (0, 255, 0), 1: (255, 0, 0), 2: (0, 0, 255)}
        c = color_map.get(cls_id, (255, 255, 255))
        overlay = np.zeros_like(vis)
        overlay[:, :, 0] = c[0]; overlay[:, :, 1] = c[1]; overlay[:, :, 2] = c[2]
        vis = np.where(mask[:, :, None] > 127, (vis * 0.6 + overlay * 0.4).astype(np.uint8), vis)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, c, 2)

    cv2.imshow("Preview | ESC=Close", vis)
    print("\n按任意键关闭预览...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(f"\n✅ 完成！数据包在: {task_dir}")
    print(f"   上传到 AutoDL 后即可进行位姿估计。")


if __name__ == "__main__":
    main()
