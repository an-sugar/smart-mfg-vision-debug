"""
全流程 V2: 拍照 + YOLO + 上传 + FP 位姿 + 下载 + 机器人两点移动
=============================================================
前提: AutoDL 已开机, fp310 环境就绪, 机器人已使能
"""

import sys, os, time, math, json
import numpy as np
import cv2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FP_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "Foundationpose")
sys.path.insert(0, FP_DIR)

OUTPUT_BASE = os.path.join(SCRIPT_DIR, "captured_cloud")

# ── 已知抓取位/放置位 (来自示教器) ──
POINT_GRASP = [-84.438, -129.095, -69.760, -21.553, 90.009, -97.098]
POINT_PLACE  = [-48.887, -128.738, -50.073, -65.008, 74.073, -96.688]

# ── 机器人 IP ──
ROBOT_IP = "192.168.58.2"


def step1_capture():
    """RealSense 拍照 + YOLO 分割 + 保存"""
    import pyrealsense2 as rs
    from ultralytics import YOLO

    print("=" * 50)
    print("  Step 1: 拍照 + YOLO 分割")
    print("=" * 50)

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

    for _ in range(30): pipe.wait_for_frames()
    frames = align.process(pipe.wait_for_frames())
    color = np.asanyarray(frames.get_color_frame().get_data())
    depth_m = np.asanyarray(frames.get_depth_frame().get_data()).astype(np.float32) * scale
    pipe.stop()

    print(f"  Color: {color.shape}  Depth: {depth_m.shape}")
    print(f"  K: fx={K[0,0]:.1f} fy={K[1,1]:.1f}")

    # YOLO
    model = YOLO(os.path.join(SCRIPT_DIR, "best.pt"))
    results = model(color, retina_masks=True, conf=0.25, iou=0.45,
                    show_boxes=False, show_conf=False)

    # 保存
    ts = time.strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(OUTPUT_BASE, ts)
    os.makedirs(save_dir, exist_ok=True)
    cv2.imwrite(os.path.join(save_dir, "color.png"), color)
    np.save(os.path.join(save_dir, "depth.npy"), depth_m)
    np.savetxt(os.path.join(save_dir, "camera_K.txt"), K, fmt="%.6f")

    detected = False
    for r in results:
        if r.masks is not None:
            for i, mt in enumerate(r.masks.data):
                m = (mt.cpu().numpy() > 0.5).astype(np.uint8) * 255
                cid = int(r.boxes[i].cls.item()) if r.boxes else 0
                cf = r.boxes[i].conf.item() if r.boxes else 1.0
                name = {0: "object1", 1: "object2", 2: "object3"}.get(cid, "unknown")
                cv2.imwrite(os.path.join(save_dir, f"mask_{name}.png"), m)
                print(f"  {name}  conf={cf:.3f}  -> mask_{name}.png")
                detected = True

    with open(os.path.join(save_dir, "info.json"), "w") as f:
        json.dump({"ts": ts, "detected": detected, "K": K.tolist()}, f)

    print(f"\n  Saved: {save_dir}")
    return save_dir, detected


def step2_upload_to_cloud(local_dir):
    """提示用户上传到 AutoDL"""
    ts = os.path.basename(local_dir)
    print(f"\n{'=' * 50}")
    print(f"  Step 2: 上传到 AutoDL")
    print(f"{'=' * 50}")
    print(f"\n  请在 JupyterLab 中上传此文件夹:")
    print(f"    {local_dir}")
    print(f"\n  然后在 AutoDL 终端运行:")
    print(f"    conda activate fp310")
    print(f"    python demo_cloud_pose.py --data /root/{ts} --mesh /root/obj_OBJ/mesh3/plane2.obj")
    print(f"\n  下载结果文件 pose_mask_object1.txt 到本机后, 按 Enter 继续...")
    input()


def step3_move_to_points():
    """机器人: 抓取位 → 放置位"""
    from Robot import RPC

    print(f"\n{'=' * 50}")
    print(f"  Step 3: 机器人移动 (抓取位 → 放置位)")
    print(f"{'=' * 50}")

    robot = RPC(ROBOT_IP)

    # 拖动示教激活
    print("  激活伺服...")
    robot.ResetAllError(); time.sleep(0.5)
    robot.Mode(1); time.sleep(0.5)
    robot.RobotEnable(1); time.sleep(1)
    robot.DragTeachSwitch(1); time.sleep(2)
    robot.DragTeachSwitch(0); time.sleep(1)
    robot.RobotEnable(1); time.sleep(1)
    print("  伺服已激活")

    # 移动到抓取位
    print(f"  MoveJ → 抓取位 {[f'{j:.1f}' for j in POINT_GRASP]}")
    robot.MoveJ(POINT_GRASP, 0, 0)
    time.sleep(3)

    # 抓取 (此处由 PLC 控制气动夹爪)
    print(f"  >>> PLC 控制夹爪闭合 <<<")
    time.sleep(1)

    # 移动到放置位
    print(f"  MoveJ → 放置位 {[f'{j:.1f}' for j in POINT_PLACE]}")
    robot.MoveJ(POINT_PLACE, 0, 0)
    time.sleep(3)

    # 释放
    print(f"  >>> PLC 控制夹爪打开 <<<")
    time.sleep(1)

    print(f"\n  Done. 一个完整抓放周期完成。")
    return robot


def main():
    print("=" * 55)
    print("  FULL PIPELINE V2: Capture -> Move")
    print("=" * 55)

    # Step 1: 拍照 + YOLO
    save_dir, detected = step1_capture()

    if not detected:
        print(f"\n WARNING: YOLO 未检测到物体! 数据仍保存在 {save_dir}")

    # Step 2: 提示上传到云端 (人工操作)
    step2_upload_to_cloud(save_dir)

    # Step 3: 下载位姿后 (人工操作), 机器人移动
    step3_move_to_points()

    print(f"\n{'=' * 55}")
    print(f"  Pipeline Complete!")
    print(f"{'=' * 55}")


if __name__ == "__main__":
    main()
