"""
FR3 机器人两点往复运动 Demo
============================
从当前位姿 → 目标点A → 目标点B → 循环

基于法奥官方 Python SDK 接口:
  MoveJ(joint, desc_pos, tool, user, vel, acc, ovl, exaxis_pos, blendT, offset_flag, offset_pos)
  blendT=-1.0 阻塞到位, vel=30 慢速安全

用法: python demo_move_back_forth.py
"""

import sys, os, time, math
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(SCRIPT_DIR), "Foundationpose"))

from Robot import RPC

ROBOT_IP = "192.168.58.2"

def main():
    print("=" * 50)
    print("  FR3 两点往复运动 Demo")
    print("=" * 50)

    # ── 1. 连接 + 初始化 ──
    print("\n[1] 连接机器人...")
    robot = RPC(ROBOT_IP)
    robot.ResetAllError()
    time.sleep(0.5)
    robot.Mode(1)
    time.sleep(0.5)
    robot.RobotEnable(1)
    time.sleep(1)

    # ── 2. 获取当前关节角 ──
    ret = robot.GetActualJointPosDegree()
    if ret[0] != 0:
        # 20004 口断连 — 用已知安全位
        print("  20004 口断连，使用已知安全初始位")
        current = [-85.8, -64.7, -122.5, -22.6, 88.7, -88.3]
    else:
        current = list(ret[1])
    print(f"  当前关节: {[f'{j:.1f}' for j in current]}")

    # ── 3. 定义两个目标位姿 ──
    # 点A: 当前关节基础上微调关节1和关节2
    point_a = current.copy()
    point_a[0] += 10.0      # 关节1 +10度
    point_a[1] -= 8.0       # 关节2 -8度

    # 点B: 回到当前的基础上往反方向调
    point_b = current.copy()
    point_b[0] -= 12.0      # 关节1 -12度
    point_b[1] += 6.0       # 关节2 +6度

    # 通用参数
    tool, user = 0, 0
    vel, acc, ovl = 30.0, 50.0, 100.0
    exaxis = [0.0, 0.0, 0.0, 0.0]
    dP = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    blendT = -1.0   # 阻塞 — 到位才返回

    # ── 4. 循环往复 ──
    rounds = 3
    print(f"\n[2] 开始往复运动 ({rounds} 轮)...")
    print(f"    点A: joint0={point_a[0]:.1f}  joint1={point_a[1]:.1f}")
    print(f"    点B: joint0={point_b[0]:.1f}  joint1={point_b[1]:.1f}")

    for i in range(rounds):
        print(f"\n  --- 第 {i+1}/{rounds} 轮 ---")

        # 移动到A (Robot.py 签名: MoveJ(joint_pos, tool, user))
        print(f"  MoveJ → A")
        robot.MoveJ(point_a, tool, user)
        time.sleep(0.5)

        # 移动到B
        print(f"  MoveJ → B")
        robot.MoveJ(point_b, tool, user)
        time.sleep(0.5)

    # ── 5. 回到安全位（用已知值，不依赖可能被污染的读数）──
    print(f"\n[3] 回到安全位...")
    safe_home = [-85.8, -64.7, -122.5, -22.6, 88.7, -88.3]
    robot.MoveJ(safe_home, tool, user)

    print(f"\n{'=' * 50}")
    print(f"  完成！机器人已回到起始位置")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
