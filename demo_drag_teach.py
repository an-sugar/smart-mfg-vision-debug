# -*- coding: utf-8 -*-
"""
调试步骤 ③ — FR3 拖动示教控制
===============================
独立的机器人拖动示教开关脚本。
功能纯粹：进入拖动示教 → 手动摆好机器人 → 退出

用法:
  python demo_drag_teach.py

操作:
  输入 d + 回车  = 切换拖动示教 开/关
  输入 q + 回车  = 退出
"""

import time

# Robot.py 已复制到本目录，直接 import
from Robot import RPC

ROBOT_IP = "192.168.58.2"


def main():
    print("=" * 50)
    print("  FR3 拖动示教控制")
    print("=" * 50)

    # ── 连接 ──
    print(f"\n连接机器人 {ROBOT_IP} ...")
    try:
        robot = RPC(ROBOT_IP)
        print("✅ 连接成功")
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        return

    # ── 初始化：清错 + 上使能 ──
    dragging = False
    print(f"\n初始化机器人...")
    print("  清除错误...")
    robot.ResetAllError()
    time.sleep(0.5)
    print("  上使能...")
    err = robot.RobotEnable(1)
    if err == 0:
        print("  ✅ 机器人已上使能")
    else:
        print(f"  ⚠️  上使能返回错误码: {err}（可能已经上过使能了）")

    print(f"\n当前状态: ⏸ 空闲（机器人锁定）")
    print("指令: [d] 切换拖动示教  [q] 退出\n")

    try:
        while True:
            cmd = input("> ").strip().lower()

            if cmd == 'd':
                if not dragging:
                    # ── 进入拖动示教 ──
                    print("  切换到手动模式...")
                    robot.Mode(1)  # 必须先切手动模式
                    time.sleep(1)
                    err = robot.DragTeachSwitch(1)
                    if err == 0:
                        print("  ✅ 拖动示教已开启 — 可以拖机器人了")
                        dragging = True
                    else:
                        print(f"  ❌ DragTeachSwitch(1) 失败，错误码: {err}")

                else:
                    # ── 退出拖动示教 ──
                    err = robot.DragTeachSwitch(0)
                    if err == 0:
                        print("  ✅ 已退出拖动示教，机器人锁定")
                        dragging = False
                    else:
                        print(f"  ❌ DragTeachSwitch(0) 失败，错误码: {err}")

            elif cmd == 'q':
                print("  退出...")
                break

            else:
                print("  未知指令。 [d] 拖动示教  [q] 退出")

    finally:
        if dragging:
            print("清理：退出拖动示教...")
            robot.DragTeachSwitch(0)
        print("完成。\n")


if __name__ == "__main__":
    main()
