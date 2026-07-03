# 20260702_112058

| 项目 | 值 |
|------|-----|
| 时间 | 20260702_112058 |
| 分辨率 | 640×480 |
| 内参 K | fx=611.772, fy=611.052, cx=318.941, cy=242.821 |
| 物体数 | 1 |
| 物体 | object1 |

## 在 AutoDL 上运行

```bash
# 把这个文件夹上传到 /root/autodl-tmp/
# 然后:
conda activate fp310
cd /root/autodl-tmp/20260702_112058
python /root/demo_pose_offline.py \
    --mesh /root/obj_OBJ/mesh_X/物体名.obj \
    --data 此文件夹路径
```
