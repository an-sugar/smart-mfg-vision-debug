# -*- coding: utf-8 -*-
"""
调试步骤 ② — YOLO 分割验证 Demo
=================================
功能:
  1. 加载已训练的 YOLO 分割模型 best.pt
  2. 从 RealSense 拍照 或 使用已有图片
  3. 运行分割推理，输出每个物体的 mask
  4. 可视化：原图 + 半透明 mask 叠加 + 标签
  5. 保存 mask 到 demo_data/mask/obj*.png

用法:
  # 方式 A: 从相机拍照后分割（默认）
  python demo_yolo_test.py

  # 方式 B: 对已有图片做分割
  python demo_yolo_test.py --image ./origimgs/original_img.jpg

  # 方式 C: 对某个文件夹批量推理
  python demo_yolo_test.py --source ./origimgs

注意:
  本脚本需要放在 Foundationpose/ 同级目录运行，或修改 FOUNDATIONPOSE_DIR 变量。
  模型路径: YOLOv8seg/ultralytics/runs/segment/train16/weights/best.pt
"""

import argparse
import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np

# ── 路径配置 ──────────────────────────────────────────────
# 脚本所在目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Foundationpose 目录 (脚本在 调试记录/ 下，Foundationpose 在同级)
FOUNDATIONPOSE_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "Foundationpose")
# 模型路径
MODEL_PATH = os.path.join(
    FOUNDATIONPOSE_DIR,
    "YOLOv8seg/ultralytics/runs/segment/train16/weights/best.pt",
)
# 输出目录
MASK_OUT_DIR = os.path.join(FOUNDATIONPOSE_DIR, "demo_data", "mask")
CAPTURE_DIR = os.path.join(FOUNDATIONPOSE_DIR, "origimgs")
RESULT_DIR = os.path.join(SCRIPT_DIR, "segmentation_results")

# 3 类标签 → 按你的数据集配置
CLASS_NAMES = {0: "object1", 1: "object2", 2: "object3"}
# 每个类别的显示颜色 (BGR)
CLASS_COLORS = {
    0: (0, 255, 0),    # object1 绿色
    1: (255, 0, 0),    # object2 蓝色
    2: (0, 0, 255),    # object3 红色
}


def find_model():
    """自动查找 best.pt，优先用 train16，否则搜所有 train 目录"""
    if os.path.exists(MODEL_PATH):
        return MODEL_PATH

    # 回退：扫描所有 train 目录
    runs_dir = os.path.join(
        FOUNDATIONPOSE_DIR,
        "YOLOv8seg/ultralytics/runs/segment",
    )
    if os.path.isdir(runs_dir):
        for name in sorted(os.listdir(runs_dir), reverse=True):
            pt = os.path.join(runs_dir, name, "weights", "best.pt")
            if os.path.exists(pt):
                return pt
    return None


def capture_from_camera(save_dir):
    """用 RealSense 拍一张照片，返回图片路径"""
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

    # 预热几帧
    for _ in range(30):
        pipeline.wait_for_frames()

    image = None
    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            frame = np.asanyarray(color_frame.get_data())

            cv2.putText(frame, "SPACE=Capture  ESC=Cancel", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("YOLO Seg Test — 拍照", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 32:
                os.makedirs(save_dir, exist_ok=True)
                img_path = os.path.join(save_dir, "original_img.jpg")
                cv2.imwrite(img_path, frame)
                print(f"📷 已保存: {img_path}")
                image = frame
                break
            elif key == 27:
                print("❌ 取消拍照。")
                break
    finally:
        pipeline.stop()
        cv2.destroyWindow("YOLO Seg Test — 拍照")

    return img_path if image is not None else None


def run_segmentation(model, source):
    """运行 YOLO 分割推理"""
    print(f"\n🔍 推理中... 来源: {source}")
    t0 = time.time()
    results = model(
        source=source,
        save=False,          # 不用自动保存，我们手动处理
        retina_masks=True,   # 高质量 mask（与输入分辨率一致）
        conf=0.25,           # 置信度阈值
        iou=0.45,            # NMS IoU 阈值
        show_boxes=False,
        show_conf=False,
        show_labels=True,
    )
    elapsed = time.time() - t0
    print(f"⏱  推理耗时: {elapsed:.3f}s")
    return results


def visualize_and_save(results, original_image=None, save_vis=True):
    """
    处理推理结果:
      - 保存 mask 到 demo_data/mask/obj*.png
      - 生成叠加可视化图
      - 打印检测摘要
    """
    os.makedirs(MASK_OUT_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)

    total_masks = 0

    for img_idx, result in enumerate(results):
        img_name = os.path.splitext(os.path.basename(result.path))[0]

        if original_image is not None:
            vis = original_image.copy()
        else:
            vis = result.orig_img.copy()

        if result.masks is None or len(result.masks) == 0:
            print(f"⚠️  [{img_name}] 未检测到任何物体！")
            cv2.imwrite(os.path.join(RESULT_DIR, f"{img_name}_no_detect.png"), vis)
            continue

        masks_data = result.masks.data  # (N, H, W) tensor
        boxes = result.boxes

        print(f"\n─── [{img_name}] ───")
        obj_count = 0

        for i, mask_tensor in enumerate(masks_data):
            mask_np = mask_tensor.cpu().numpy()
            mask_bin = (mask_np > 0.5).astype(np.uint8)

            # 获取类别
            if boxes is not None and i < len(boxes):
                cls_id = int(boxes[i].cls.item())
                conf = boxes[i].conf.item()
            else:
                cls_id = 0
                conf = 1.0

            cls_name = CLASS_NAMES.get(cls_id, f"class_{cls_id}")
            color = CLASS_COLORS.get(cls_id, (255, 255, 255))

            # ── 保存 mask ──
            obj_count += 1
            mask_save_path = os.path.join(MASK_OUT_DIR, f"obj{obj_count}.png")
            cv2.imwrite(mask_save_path, mask_bin * 255)
            print(f"  ✅ obj{obj_count} | {cls_name} | conf={conf:.3f} → {mask_save_path}")

            # ── 叠加到可视化图 ──
            mask_colored = np.zeros_like(vis)
            mask_colored[:, :, 0] = color[0]  # B
            mask_colored[:, :, 1] = color[1]  # G
            mask_colored[:, :, 2] = color[2]  # R
            # 半透明叠加
            alpha = 0.4
            vis = np.where(
                mask_bin[:, :, None] > 0,
                (vis * (1 - alpha) + mask_colored * alpha).astype(np.uint8),
                vis,
            )

            # 画轮廓
            contours, _ = cv2.findContours(
                mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(vis, contours, -1, color, 2)

            # 写标签
            if len(contours) > 0:
                c = max(contours, key=cv2.contourArea)
                M = cv2.moments(c)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    label = f"{cls_name} ({conf:.2f})"
                    cv2.putText(
                        vis, label, (cx - 30, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
                    )

            total_masks += 1

    # ── 保存可视化结果 ──
    vis_path = os.path.join(RESULT_DIR, f"{img_name}_seg_overlay.png")
    cv2.imwrite(vis_path, vis)
    print(f"\n📊 可视化已保存: {vis_path}")
    print(f"  共检测到 {total_masks} 个物体 mask")

    return vis, total_masks


def main():
    print("=" * 55)
    print("  调试步骤 ② — YOLO 分割验证")
    print("=" * 55)

    # ── 1. 找模型 ──
    model_path = find_model()
    if model_path is None:
        print(f"❌ 找不到 best.pt，请检查路径:")
        print(f"   预期位置: {MODEL_PATH}")
        sys.exit(1)

    print(f"\n✅ 模型: {model_path}")
    print(f"   类别: {CLASS_NAMES}")

    # ── 2. 加载模型 ──
    from ultralytics import YOLO
    print("   加载中...")
    model = YOLO(model_path)
    print("   加载完成。")

    # ── 3. 确定图片来源 ──
    parser = argparse.ArgumentParser(description="YOLO 分割验证 Demo")
    parser.add_argument(
        "--image", type=str, default=None,
        help="单张图片路径（不传则打开相机拍照）",
    )
    parser.add_argument(
        "--source", type=str, default=None,
        help="图片文件夹路径（批量推理）",
    )
    args = parser.parse_args()

    if args.source:
        # 批量模式
        source = args.source
        original_image = None
        print(f"\n📂 批量模式: {source}")
    elif args.image:
        # 单图模式
        source = args.image
        original_image = cv2.imread(source)
        if original_image is None:
            print(f"❌ 无法读取图片: {source}")
            sys.exit(1)
        print(f"\n🖼  图片模式: {source}")
    else:
        # 相机模式
        os.makedirs(CAPTURE_DIR, exist_ok=True)
        img_path = capture_from_camera(CAPTURE_DIR)
        if img_path is None:
            print("未拍摄照片，退出。")
            sys.exit(0)
        source = img_path
        original_image = cv2.imread(img_path)

    # ── 4. 推理 ──
    results = run_segmentation(model, source)

    # ── 5. 可视化 + 保存 ──
    vis, count = visualize_and_save(results, original_image)

    if count > 0 and original_image is not None:
        # ── 6. 显示对比 ──
        h, w = original_image.shape[:2]
        scale = min(900 / w, 600 / h, 1.0)
        if scale < 1.0:
            original_image = cv2.resize(original_image, None, fx=scale, fy=scale)
            vis = cv2.resize(vis, None, fx=scale, fy=scale)

        side_by_side = np.hstack((original_image, vis))
        cv2.putText(side_by_side, "Original", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(side_by_side, "Segmentation", (w + 10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("YOLO Segmentation Result | Left=Original  Right=+Masks", side_by_side)
        print("\n按任意键关闭窗口...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print(f"\n{'=' * 55}")
    if count > 0:
        print(f"  ✅ 第②步验证通过！")
        print(f"  Mask 已保存到: {MASK_OUT_DIR}")
        print(f"  可视化保存在: {RESULT_DIR}")
        print(f"\n  👉 下一步: 运行 Multi_YOLO_FP_new.py 进行位姿估计")
    else:
        print(f"  ⚠️ 未检测到物体！请检查:")
        print(f"    1. 相机/图片中是否有待检测物体")
        print(f"    2. 模型类别是否匹配 (当前: {CLASS_NAMES})")
        print(f"    3. 光照条件是否充足")
    print(f"{'=' * 55}")


if __name__ == "__main__":
    main()
