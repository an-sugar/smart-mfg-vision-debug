"""测试 FR3 FK 返回的欧拉角约定"""
import sys, os, math, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Foundationpose"))
from Robot import RPC

robot = RPC("192.168.58.2")
robot.ResetAllError(); time.sleep(0.5)
robot.Mode(1); time.sleep(0.5)
robot.RobotEnable(1); time.sleep(1)

# 用抓取位测试 (J4≈-21, J5≈90, J6≈-97, 手腕朝下)
GRASP_JOINTS = [-84.438, -129.095, -69.760, -21.553, 90.009, -97.098]
ret = robot.GetForwardKin(GRASP_JOINTS)
if ret[0] != 0:
    print("FK failed"); sys.exit(1)
fk = list(ret[1])
print(f"FK result: pos=[{fk[0]:.1f}, {fk[1]:.1f}, {fk[2]:.1f}]  rpy=[{fk[3]:.1f}, {fk[4]:.1f}, {fk[5]:.1f}]")

rx, ry, rz = math.radians(fk[3]), math.radians(fk[4]), math.radians(fk[5])
cx, sx = math.cos(rx), math.sin(rx)
cy, sy = math.cos(ry), math.sin(ry)
cz, sz = math.cos(rz), math.sin(rz)

Rx = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])
Ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])
Rz = np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])

# 测试各种组合
tests = [
    ("Rz@Ry@Rx", Rz @ Ry @ Rx),
    ("Rx@Ry@Rz", Rx @ Ry @ Rz),
    ("Ry@Rx@Rz", Ry @ Rx @ Rz),
    ("Rz@Rx@Ry", Rz @ Rx @ Ry),
    ("Ry@Rz@Rx", Ry @ Rz @ Rx),
    ("Rx@Rz@Ry", Rx @ Rz @ Ry),
]

print("\n在抓取位(J4≈-21,J5≈90), 手腕朝下拍摄:")
print("  Z轴应朝下: R第3列 ≈ [0, 0, ±1]")
print("  Y轴应朝前: R第2列 ≈ [0, ±1, 0]")
print()
for name, R in tests:
    print(f"{name:12s}: Z=[{R[0,2]:.3f}, {R[1,2]:.3f}, {R[2,2]:.3f}]  Y=[{R[0,1]:.3f}, {R[1,1]:.3f}, {R[2,1]:.3f}]")

# 用当前极端姿态测试
PHOTO_JOINTS = [-148.97, -89.56, -88.47, -129.49, 180.12, -179.67]
ret2 = robot.GetForwardKin(PHOTO_JOINTS)
if ret2[0] == 0:
    fk2 = list(ret2[1])
    rx2, ry2, rz2 = math.radians(fk2[3]), math.radians(fk2[4]), math.radians(fk2[5])
    cx2, sx2 = math.cos(rx2), math.sin(rx2)
    cy2, sy2 = math.cos(ry2), math.sin(ry2)
    cz2, sz2 = math.cos(rz2), math.sin(rz2)
    Rx2 = np.array([[1,0,0],[0,cx2,-sx2],[0,sx2,cx2]])
    Ry2 = np.array([[cy2,0,sy2],[0,1,0],[-sy2,0,cy2]])
    Rz2 = np.array([[cz2,-sz2,0],[sz2,cz2,0],[0,0,1]])

    print(f"\n当前极端姿态(J4=-129,J5=180):")
    print(f"  FK rpy=[{fk2[3]:.1f}, {fk2[4]:.1f}, {fk2[5]:.1f}]")
    for name, R in [(n, f(Rx2, Ry2, Rz2)) for n, f in [
        ("Rz@Ry@Rx", lambda x,y,z: z@y@x),
        ("Rx@Ry@Rz", lambda x,y,z: x@y@z),
        ("Ry@Rx@Rz", lambda x,y,z: y@x@z),
        ("Rz@Rx@Ry", lambda x,y,z: z@x@y),
    ]]:
        # 也验证: 物体在基坐标 = FK_pos + R @ obj_flange
        # V3说obj_base ≈ [-396, -604, 175]
        print(f"  {name}: Z=[{R[0,2]:.3f}, {R[1,2]:.3f}, {R[2,2]:.3f}]")
