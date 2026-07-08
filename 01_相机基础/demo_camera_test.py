# -*- coding: utf-8 -*-
"""
调试步骤 ① — 相机连接测试 Demo
=================================
功能:
  1. 检测 RealSense 相机是否连接
  2. 打印相机基本信息 (序列号、固件、内参)
  3. 实时显示彩色图 + 深度图
  4. 按 空格键 保存当前帧到 ./captured/
  5. 按 ESC 退出

用法:
  python demo_camera_test.py
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import os
import datetime


def check_camera():
    """检查相机连接并打印设备信息"""
    ctx = rs.context()
    devices = ctx.query_devices()

    if len(devices) == 0:
        print("❌ 未检测到 RealSense 相机！请检查 USB 连接。")
        return None

    print(f"✅ 检测到 {len(devices)} 个 RealSense 设备\n")

    for i, dev in enumerate(devices):
        print(f"─── 设备 {i+1} ───")
        print(f"  名称:     {dev.get_info(rs.camera_info.name)}")
        print(f"  序列号:   {dev.get_info(rs.camera_info.serial_number)}")
        print(f"  固件版本: {dev.get_info(rs.camera_info.firmware_version)}")
        print(f"  USB 类型: {dev.get_info(rs.camera_info.usb_type_descriptor)}")
        print()

    return devices


def get_depth_scale(profile):
    """获取深度比例因子"""
    sensor = profile.get_device().first_depth_sensor()
    scale = sensor.get_depth_scale()
    print(f"  深度缩放因子: {scale:.6f} (即深度值 × {scale:.6f} = 米)")
    return scale


def print_intrinsics(profile):
    """打印彩色相机内参"""
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    print(f"\n─── 彩色相机内参 (对齐后) ───")
    print(f"  分辨率: {intr.width} × {intr.height}")
    print(f"  fx={intr.fx:.3f}, fy={intr.fy:.3f}")
    print(f"  cx={intr.ppx:.3f}, cy={intr.ppy:.3f}")
    print(f"  畸变模型: {intr.model}")
    print(f"  畸变系数: {intr.coeffs}")

    K = np.array([[intr.fx, 0, intr.ppx],
                  [0, intr.fy, intr.ppy],
                  [0, 0, 1]])
    return K


def main():
    print("=" * 50)
    print("  调试步骤 ① — RealSense 相机连接测试")
    print("=" * 50)
    print()

    # ── 1. 检查相机 ──
    devices = check_camera()
    if devices is None:
        return

    # ── 2. 启动流 ──
    W, H, FPS = 640, 480, 30

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)
    config.enable_stream(rs.stream.depth, W, H, rs.format.z16, FPS)

    print(f"─── 启动视频流 ───")
    print(f"  分辨率: {W} × {H} @ {FPS} FPS")
    profile = pipeline.start(config)

    depth_scale = get_depth_scale(profile)
    K = print_intrinsics(profile)

    # 深度对齐到彩色
    align = rs.align(rs.stream.color)

    # 准备保存目录
    save_dir = "./captured"
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'=' * 50}")
    print("  实时显示中...")
    print("    空格键 = 保存当前帧")
    print("    ESC    = 退出")
    print(f"{'=' * 50}\n")

    # ── 3. 实时循环 ──
    try:
        for _ in range(30):  # 先丢 30 帧让自动曝光稳定
            pipeline.wait_for_frames()

        while True:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)

            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            color = np.asanyarray(color_frame.get_data())
            depth_raw = np.asanyarray(depth_frame.get_data())

            # 深度转伪彩色用于显示
            depth_m = depth_raw.astype(float) * depth_scale
            depth_vis = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_raw, alpha=0.03), cv2.COLORMAP_JET
            )

            # 深度叠加到彩色图（半透明）
            depth_overlay = cv2.addWeighted(color, 0.3, depth_vis, 0.7, 0)

            # 添加文字
            cv2.putText(color, "Color (Space=Save, ESC=Exit)", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(depth_overlay, "Depth (colormap)", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # 拼在一起显示
            display = np.hstack((color, depth_overlay))
            cv2.imshow("Camera Test | Left=Color  Right=Depth", display)

            key = cv2.waitKey(1) & 0xFF

            if key == 32:  # 空格 → 保存
                timestamp = datetime.datetime.now().strftime("%H%M%S")
                color_path = os.path.join(save_dir, f"color_{timestamp}.png")
                depth_path = os.path.join(save_dir, f"depth_{timestamp}.png")
                cv2.imwrite(color_path, color)
                np.save(depth_path.replace(".png", ".npy"), depth_m)
                print(f"📷 已保存: {color_path}")
                print(f"📷 已保存: {depth_path.replace('.png', '.npy')} (米为单位)")

            elif key == 27:  # ESC → 退出
                print("\n👋 退出。")
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
