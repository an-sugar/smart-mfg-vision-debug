"""快速读取当前关节角并写入最新 captured_cloud 目录"""
import sys, os, time, glob
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(SCRIPT_DIR), "Foundationpose"))
from Robot import RPC

robot = RPC("192.168.58.2")
robot.ResetAllError(); time.sleep(0.5)

for attempt in range(10):
    try:
        ret = robot.GetActualJointPosDegree()
        if ret[0] == 0 and not all(abs(j) < 0.001 for j in ret[1]):
            joints = list(ret[1])
            print(f"Joints: {[f'{j:.2f}' for j in joints]}")

            # 写到最新的 captured_cloud
            CAPTURED_DIR = os.path.join(SCRIPT_DIR, "captured_cloud")
            dirs = sorted(glob.glob(os.path.join(CAPTURED_DIR, "2*")), reverse=True)
            if dirs:
                jf = os.path.join(dirs[0], "photo_joints.txt")
                import numpy as np
                np.savetxt(jf, joints, fmt="%.4f")
                print(f"Saved to {dirs[0]}/photo_joints.txt")
            break
        else:
            print(f"Attempt {attempt+1}: zeros, retry...")
    except Exception as e:
        print(f"Attempt {attempt+1}: {e}")
    time.sleep(0.5)
else:
    print("Failed after 10 attempts.")
