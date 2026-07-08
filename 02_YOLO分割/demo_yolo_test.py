# -*- coding: utf-8 -*-
"""
调试步骤 ② — YOLO 分割验证 Demo
=================================
完全自包含脚本 — 所有文件读写都在本文件夹（调试记录/）内完成。

依赖:
  pip install ultralytics opencv-python numpy pyrealsense2

用法:
  # 先把 best.pt 复制到本文件夹
  # cp ../Foundationpose/YOLOv8seg/ultralytics/runs/segment/train16/weights/best.pt .

  # 方式 A: 从 RealSense 拍照后分割（默认）
  python demo_yolo_test.py

  # 方式 B: 对已有图片做分割
  python demo_yolo_test.py --image ./captures/20250701_test.jpg

  # 方式 C: 对某个文件夹批量推理
  python demo_yolo_test.py --source ./captures

  # 指定模型
  python demo_yolo_test.py --model ./best.pt
"""

import argparse
import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np

# ═══════════════════════════════════════════════════════════════════
# 一切以脚本所在目录为基准 — 绝不依赖外部路径
# ═══════════════════════════════════════════════════════════════════
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 子文件夹（自动创建）
MASK_DIR   = os.path.join(SCRIPT_DIR, "masks")       # 输出的黑白 mask
CAPTURE_DIR = os.path.join(SCRIPT_DIR, "captures")    # 拍的照片
RESULT_DIR = os.path.join(SCRIPT_DIR, "results")      # 叠加可视化结果

# 模型默认位置：先看本文件夹，再看 Foundationpose 目录
MODEL_CANDIDATES = [
    os.path.join(SCRIPT_DIR, "best.pt"),               # ① 本文件夹
    os.path.join(SCRIPT_DIR, "..", "Foundationpose",
        "YOLOv8seg", "ultralytics", "runs",
        "segment", "train16", "weights", "best.pt"),   # ② 原始训练输出
]

# 类别名 + 颜色（按你的 object-seg.yaml）
CLASS_NAMES  = {0: "object1", 1: "object2", 2: "object3"}
CLASS_COLORS = {0: (0, 255, 0), 1: (255, 0, 0), 2: (0, 0, 255)}  # BGR


# ═══════════════════════════════════════════════════════════════════
# 1. 模型查找
# ═══════════════════════════════════════════════════════════════════
def find_model(explicit=None):
    """查找 best.pt：优先用命令行指定的，其次搜索候选路径"""
    if explicit and os.path.isfile(explicit):
        return os.path.abspath(explicit)

    for cand in MODEL_CANDIDATES:
        if os.path.isfile(cand):
            return os.path.abspath(cand)

    # 回退：扫描 Foundationpose 下所有 train 目录
    runs_dir = os.path.join(
        SCRIPT_DIR, "..", "Foundationpose",
        "YOLOv8seg", "ultralytics", "runs", "segment",
    )
    runs_dir = os.path.abspath(runs_dir)
    if os.path.isdir(runs_dir):
        for name in sorted(os.listdir(runs_dir), reverse=True):
            pt = os.path.join(runs_dir, name, "weights", "best.pt")
            if os.path.isfile(pt):
                return pt

    return None


# ═══════════════════════════════════════════════════════════════════
# 2. 相机拍照
# ═══════════════════════════════════════════════════════════════════
def capture_from_camera():
    """用 RealSense 拍一张 640×480 照片，返回 (image_array, save_path)"""
    try:
        import pyrealsense2 as rs
    except ImportError:
        print("❌ 未安装 pyrealsense2，请先: pip install pyrealsense2")
        sys.exit(1)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    print("\n📷 RealSense 相机已启动")
    print("   空格键 = 拍照并进入分割")
    print("   ESC    = 取消\n")

    for _ in range(30):
        pipeline.wait_for_frames()

    image = None
    save_path = None
    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            frame = np.asanyarray(color_frame.get_data())

            cv2.putText(frame, "SPACE=Capture  ESC=Cancel", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("YOLO Seg — 拍照", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 32:
                os.makedirs(CAPTURE_DIR, exist_ok=True)
                ts = time.strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(CAPTURE_DIR, f"capture_{ts}.jpg")
                cv2.imwrite(save_path, frame)
                print(f"📷 已保存: {save_path}")
                image = frame
                break
            elif key == 27:
                print("❌ 取消。")
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

    return image, save_path


# ═══════════════════════════════════════════════════════════════════
# 3. 分割推理
# ═══════════════════════════════════════════════════════════════════
def run_segmentation(model, source):
    """运行 YOLO 分割，返回 ultralytics Results 列表"""
    print(f"\n🔍 推理中...  来源: {source}")
    t0 = time.time()
    results = model(
        source=source,
        save=False,
        retina_masks=True,
        conf=0.25,
        iou=0.45,
        show_boxes=False,
        show_conf=False,
        show_labels=True,
    )
    print(f"⏱  推理耗时: {time.time() - t0:.3f}s")
    return results


# ═══════════════════════════════════════════════════════════════════
# 4. 可视化 + 保存 mask
# ═══════════════════════════════════════════════════════════════════
def visualize_and_save(results, original_image=None):
    """保存 mask → masks/ + 叠加图 → results/，返回 (vis_image, mask_count)"""
    os.makedirs(MASK_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)

    total_masks = 0
    vis = original_image.copy() if original_image is not None else None

    for result in results:
        img_name = os.path.splitext(os.path.basename(result.path))[0]

        if vis is None:
            vis = result.orig_img.copy()

        if result.masks is None or len(result.masks) == 0:
            print(f"⚠️  [{img_name}] 未检测到任何物体！")
            cv2.imwrite(
                os.path.join(RESULT_DIR, f"{img_name}_no_detect.png"), vis
            )
            continue

        masks_data = result.masks.data  # (N, H, W) tensor
        boxes = result.boxes

        print(f"\n─── [{img_name}] ───")
        obj_count = 0

        for i, mask_tensor in enumerate(masks_data):
            mask_np = mask_tensor.cpu().numpy()
            mask_bin = (mask_np > 0.5).astype(np.uint8)

            if boxes is not None and i < len(boxes):
                cls_id = int(boxes[i].cls.item())
                conf = boxes[i].conf.item()
            else:
                cls_id = 0
                conf = 1.0

            cls_name = CLASS_NAMES.get(cls_id, f"class_{cls_id}")
            color = CLASS_COLORS.get(cls_id, (255, 255, 255))
            obj_count += 1

            # ── 保存黑白 mask ──
            mask_path = os.path.join(MASK_DIR, f"obj{obj_count}.png")
            cv2.imwrite(mask_path, mask_bin * 255)
            print(f"  ✅ obj{obj_count} | {cls_name} | conf={conf:.3f} → {mask_path}")

            # ── 半透明颜色叠加 ──
            overlay = np.zeros_like(vis)
            overlay[:, :, 0] = color[0]
            overlay[:, :, 1] = color[1]
            overlay[:, :, 2] = color[2]
            vis = np.where(
                mask_bin[:, :, None] > 0,
                (vis * 0.6 + overlay * 0.4).astype(np.uint8),
                vis,
            )

            # ── 轮廓 ──
            contours, _ = cv2.findContours(
                mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(vis, contours, -1, color, 2)

            # ── 标签 ──
            if contours:
                M = cv2.moments(max(contours, key=cv2.contourArea))
                if M["m00"] > 0:
                    cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                    cv2.putText(vis, f"{cls_name} ({conf:.2f})",
                                (cx - 30, cy - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, color, 2)

            total_masks += 1

    # 保存叠加图
    ts = time.strftime("%Y%m%d_%H%M%S")
    vis_path = os.path.join(RESULT_DIR, f"overlay_{ts}.png")
    cv2.imwrite(vis_path, vis)
    print(f"\n📊 叠加图: {vis_path}  |  共 {total_masks} 个 mask")

    return vis, total_masks


# ═══════════════════════════════════════════════════════════════════
# 5. 主入口
# ═══════════════════════════════════════════════════════════════════
def main():
    print("=" * 55)
    print("  调试步骤 ② — YOLO 分割验证")
    print(f"  工作目录: {SCRIPT_DIR}")
    print("=" * 55)

    # ── 参数 ──
    parser = argparse.ArgumentParser(description="YOLO 分割验证 (独立运行)")
    parser.add_argument("--model", type=str, default=None,
                        help="best.pt 路径 (默认自动查找)")
    parser.add_argument("--image", type=str, default=None,
                        help="单张图片路径")
    parser.add_argument("--source", type=str, default=None,
                        help="图片文件夹路径 (批量)")
    args = parser.parse_args()

    # ── 找模型 ──
    model_path = find_model(explicit=args.model)
    if model_path is None:
        print("❌ 找不到 best.pt！\n"
              "   请把 best.pt 复制到本文件夹，或用 --model 指定路径。\n"
              f"   脚本所在目录: {SCRIPT_DIR}")
        sys.exit(1)

    print(f"\n✅ 模型: {model_path}")
    print(f"   类别: {CLASS_NAMES}")

    from ultralytics import YOLO
    model = YOLO(model_path)
    print("   加载完成。")

    # ── 图片来源 ──
    if args.source:
        source = args.source
        original_image = None
        print(f"\n📂 批量: {source}")
    elif args.image:
        source = args.image
        original_image = cv2.imread(source)
        if original_image is None:
            print(f"❌ 无法读取: {source}")
            sys.exit(1)
        print(f"\n🖼  图片: {source}")
    else:
        original_image, source = capture_from_camera()
        if source is None:
            print("未拍照，退出。"); sys.exit(0)

    # ── 推理 + 保存 ──
    results = run_segmentation(model, source)
    vis, count = visualize_and_save(results, original_image)

    # ── 屏幕显示对比 ──
    if count > 0 and original_image is not None:
        h, w = original_image.shape[:2]
        scale = min(900 / w, 600 / h, 1.0)
        o = cv2.resize(original_image, None, fx=scale, fy=scale) if scale < 1.0 else original_image
        v = cv2.resize(vis, None, fx=scale, fy=scale) if scale < 1.0 else vis

        side = np.hstack((o, v))
        cv2.putText(side, "Original", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(side, "Segmentation", (w + 10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("YOLO | Left=Original  Right=+Masks", side)
        print("\n按任意键关闭窗口...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # ── 总结 ──
    print(f"\n{'=' * 55}")
    if count:
        print(f"  ✅ 第②步验证通过！")
        print(f"  Mask:     {MASK_DIR}")
        print(f"  叠加图:   {RESULT_DIR}")
        print(f"  照片:     {CAPTURE_DIR}")
        print(f"\n  👉 下一步: 运行 demo_yolo_test.py 拿这些 mask 给")
        print(f"             Multi_YOLO_FP_new.py 做位姿估计")
    else:
        print(f"  ⚠️ 未检测到物体！")
    print(f"{'=' * 55}")


if __name__ == "__main__":
    main()
