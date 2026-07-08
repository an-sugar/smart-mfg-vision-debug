"""
两点往复移动 — 拖动示教激活版
==============================
拖动示教进/出激活伺服 → 立即跑 A↔B
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "..", "Foundationpose"))
from Robot import RPC

ROBOT_IP = "192.168.58.2"

POINT_A = [-84.438, -129.095, -69.760, -21.553, 90.009, -97.098]   # 抓取位
POINT_B = [-48.887, -128.738, -50.073, -65.008, 74.073, -96.688]  # 放置位


def main():
    loops = 1
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--loop", type=int, default=1)
    args = parser.parse_args()
    loops = args.loop

    print("=" * 50)
    print("  FR3 两点移动 (拖动示教激活)")
    print(f"  A: {[f'{j:.1f}' for j in POINT_A]}")
    print(f"  B: {[f'{j:.1f}' for j in POINT_B]}")
    print("=" * 50)

    robot = RPC(ROBOT_IP)

    # ── Step 0: 拖动示教激活伺服 ──
    print("\n[0] 拖动示教激活伺服...")
    robot.ResetAllError()
    time.sleep(0.5)
    robot.Mode(1)
    time.sleep(0.5)
    robot.RobotEnable(1)
    time.sleep(1)

    # 进拖动示教
    robot.DragTeachSwitch(1)
    print("    拖动示教 ON — 3秒...")
    time.sleep(3)

    # 出拖动示教
    robot.DragTeachSwitch(0)
    print("    拖动示教 OFF")
    time.sleep(1)

    # 重新上使能
    robot.RobotEnable(1)
    print("    重新上使能完成")
    time.sleep(1)

    # ── 两点移动 ──
    for rnd in range(loops):
        print(f"\n--- Round {rnd+1}/{loops} ---")
        print(f"  MoveJ → A (抓取位)")
        robot.MoveJ(POINT_A, 0, 0)
        time.sleep(3)

        print(f"  MoveJ → B (放置位)")
        robot.MoveJ(POINT_B, 0, 0)
        time.sleep(3)

    print(f"\nDone.")


if __name__ == "__main__":
    main()
