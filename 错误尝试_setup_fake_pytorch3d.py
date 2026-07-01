# -*- coding: utf-8 -*-
"""
创建假的 pytorch3d 包（绕过编译问题）
运行一次即可，以后 FoundationPose 就能正常 import 了。
用法: python setup_fake_pytorch3d.py
"""

import os
import sys
import textwrap

# 目标目录
site_packages = os.path.dirname(__file__) if "__file__" in dir() else os.getcwd()
PKG = os.path.join(site_packages, "pytorch3d")


def write_file(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(textwrap.dedent(content))
    print(f"  ✅ {path}")


def main():
    print("创建 pytorch3d 兼容包...")

    # ── pytorch3d/__init__.py ──
    write_file(f"{PKG}/__init__.py", """\
        pass
    """)

    # ── pytorch3d/transforms.py ──
    write_file(f"{PKG}/transforms.py", """\
        import torch
        import torch.nn.functional as F

        def so3_exp_map(log_rot, eps=1e-4):
            if log_rot.dim() == 1:
                log_rot = log_rot.unsqueeze(0)
            N = log_rot.shape[0]
            angle = torch.norm(log_rot, dim=1)
            valid = angle > eps
            angle_s = torch.where(valid, angle, torch.ones_like(angle))
            axis = log_rot / angle_s.unsqueeze(-1)
            cos, sin = torch.cos(angle), torch.sin(angle)
            K = torch.zeros(N,3,3,device=log_rot.device,dtype=log_rot.dtype)
            K[:,0,1]=-axis[:,2]; K[:,0,2]=axis[:,1]; K[:,1,0]=axis[:,2]
            K[:,1,2]=-axis[:,0]; K[:,2,0]=-axis[:,1]; K[:,2,1]=axis[:,0]
            I = torch.eye(3,device=log_rot.device).unsqueeze(0)
            R = I + sin.view(-1,1,1)*K + (1-cos).view(-1,1,1)*K@K
            R[~valid] = I[0]
            return R.squeeze(0) if N==1 else R

        def so3_log_map(R, eps=1e-4, cos_bound=1e-4):
            if R.dim()==2: R=R.unsqueeze(0)
            N=R.shape[0]
            trace=(R[:,0,0]+R[:,1,1]+R[:,2,2]).clamp(-1+cos_bound,1-cos_bound)
            angle=torch.acos((trace-1)/2)
            rx=R[:,2,1]-R[:,1,2]; ry=R[:,0,2]-R[:,2,0]; rz=R[:,1,0]-R[:,0,1]
            axis=torch.stack([rx,ry,rz],dim=1)
            valid=angle>eps
            angle_s=torch.where(valid,angle,torch.ones_like(angle))
            log=axis/(torch.norm(axis,dim=1,keepdim=True)+1e-8)*angle_s.unsqueeze(-1)
            log[~valid]=0
            return log.squeeze(0) if N==1 else log

        def se3_exp_map(log, eps=1e-4):
            if log.dim()==1: log=log.unsqueeze(0)
            N=log.shape[0]; rv=log[:,:3]; t=log[:,3:]
            R=so3_exp_map(rv,eps)
            if R.dim()==2: R=R.unsqueeze(0)
            angle=torch.norm(rv,dim=1); valid=angle>eps
            angle_s=torch.where(valid,angle,torch.ones_like(angle))
            rn=rv/angle_s.unsqueeze(-1)
            K=torch.zeros(N,3,3,device=log.device,dtype=log.dtype)
            K[:,0,1]=-rn[:,2];K[:,0,2]=rn[:,1];K[:,1,0]=rn[:,2]
            K[:,1,2]=-rn[:,0];K[:,2,0]=-rn[:,1];K[:,2,1]=rn[:,0]
            I=torch.eye(3,device=log.device).unsqueeze(0)
            s,c=torch.sin(angle),torch.cos(angle)
            V=I + (s/angle_s).view(-1,1,1)*K + ((1-c)/(angle_s**2)).view(-1,1,1)*K@K
            T=torch.zeros(N,4,4,device=log.device,dtype=log.dtype)
            T[:,:3,:3]=R; T[:,:3,3]=(V@t.unsqueeze(-1)).squeeze(-1); T[:,3,3]=1
            T[~valid,:3,:3]=I[0]; T[~valid,:3,3]=t[~valid]
            return T.squeeze(0) if N==1 else T

        def se3_log_map(T, eps=1e-4, cos_bound=1e-4):
            if T.dim()==2: T=T.unsqueeze(0)
            N=T.shape[0]; R=T[:,:3,:3]; t=T[:,:3,3]
            rv=so3_log_map(R,eps,cos_bound)
            if rv.dim()==1: rv=rv.unsqueeze(0)
            angle=torch.norm(rv,dim=1); valid=angle>eps
            angle_s=torch.where(valid,angle,torch.ones_like(angle))
            rn=rv/angle_s.unsqueeze(-1)
            K=torch.zeros(N,3,3,device=T.device,dtype=T.dtype)
            K[:,0,1]=-rn[:,2];K[:,0,2]=rn[:,1];K[:,1,0]=rn[:,2]
            K[:,1,2]=-rn[:,0];K[:,2,0]=-rn[:,1];K[:,2,1]=rn[:,0]
            I=torch.eye(3,device=T.device).unsqueeze(0)
            s,c=torch.sin(angle),torch.cos(angle)
            A=s/angle_s; B=(1-c)/(angle_s**2)
            C=(1-A/(2*B+1e-8))/(angle_s**2+1e-8)
            Vinv=I-0.5*K+C.view(-1,1,1)*K@K
            trans=(Vinv@t.unsqueeze(-1)).squeeze(-1)
            trans[~valid]=t[~valid]
            lt=torch.cat([rv,trans],dim=1)
            return lt.squeeze(0) if N==1 else lt

        def matrix_to_axis_angle(M):
            return torch.cat([so3_log_map(M[:,:3,:3]), so3_log_map(M[:,:3,:3]).norm(dim=-1,keepdim=True)], dim=-1)

        def matrix_to_euler_angles(M, convention='XYZ'):
            R=M[:,:3,:3]; sy=torch.sqrt(R[:,0,0]**2+R[:,1,0]**2)
            x=torch.atan2(R[:,2,1],R[:,2,2])
            y=torch.atan2(-R[:,2,0],sy)
            z=torch.atan2(R[:,1,0],R[:,0,0])
            return torch.stack([x,y,z],dim=-1)

        def euler_angles_to_matrix(e, convention='XYZ'):
            if e.dim()==1: e=e.unsqueeze(0)
            N=e.shape[0]
            x,y,z=e[:,0],e[:,1],e[:,2]
            ctx,stx=torch.cos(x),torch.sin(x)
            cty,sty=torch.cos(y),torch.sin(y)
            ctz,stz=torch.cos(z),torch.sin(z)
            Rx=torch.zeros(N,3,3,device=e.device); Rx[:,0,0]=1; Rx[:,1,1]=ctx; Rx[:,1,2]=-stx; Rx[:,2,1]=stx; Rx[:,2,2]=ctx
            Ry=torch.zeros(N,3,3,device=e.device); Ry[:,0,0]=cty; Ry[:,0,2]=sty; Ry[:,1,1]=1; Ry[:,2,0]=-sty; Ry[:,2,2]=cty
            Rz=torch.zeros(N,3,3,device=e.device); Rz[:,0,0]=ctz; Rz[:,0,1]=-stz; Rz[:,1,0]=stz; Rz[:,1,1]=ctz; Rz[:,2,2]=1
            return Rz@Ry@Rx

        def rotation_6d_to_matrix(d6):
            if d6.dim()==1: d6=d6.unsqueeze(0)
            a1,a2=d6[:,:3],d6[:,3:6]
            b1=F.normalize(a1,dim=-1)
            b2=a2-(b1*a2).sum(-1,keepdim=True)*b1; b2=F.normalize(b2,dim=-1)
            b3=torch.cross(b1,b2,dim=-1)
            return torch.stack([b1,b2,b3],dim=-1)
    """)

    # ── pytorch3d/structures.py ──
    write_file(f"{PKG}/structures.py", """\
        import torch
        class Meshes:
            def __init__(self, verts=None, faces=None, textures=None, **kwargs):
                self.verts_list = verts if isinstance(verts, list) else [verts]
                self.faces_list = faces if isinstance(faces, list) else [faces]
                self.textures = textures
                self.device = self.verts_list[0].device if self.verts_list else 'cpu'
            def to(self, device):
                self.verts_list = [v.to(device) for v in self.verts_list]
                self.faces_list = [f.to(device) for f in self.faces_list]
                return self
            def __len__(self):
                return len(self.verts_list)
            def num_verts_per_mesh(self):
                return [v.shape[0] for v in self.verts_list]
    """)

    # ── pytorch3d/renderer/__init__.py ──
    write_file(f"{PKG}/renderer/__init__.py", """\
        import torch

        class _Fake:
            pass

        FoVPerspectiveCameras = _Fake
        PerspectiveCameras = _Fake
        PointLights = _Fake
        MeshRenderer = _Fake
        MeshRasterizer = _Fake
        RasterizationSettings = _Fake
        BlendParams = _Fake
        SoftSilhouetteShader = _Fake
        HardPhongShader = _Fake
        TexturesVertex = _Fake
        look_at_view_transform = lambda *a,**kw: (torch.eye(4), torch.eye(4))
        look_at_rotation = lambda *a,**kw: torch.eye(4)
    """)

    # ── pytorch3d/renderer/mesh/__init__.py ──
    write_file(f"{PKG}/renderer/mesh/__init__.py", "pass\n")

    # ── pytorch3d/renderer/mesh/rasterize_meshes.py ──
    write_file(f"{PKG}/renderer/mesh/rasterize_meshes.py", """\
        def barycentric_coordinates(*a, **kw):
            raise NotImplementedError("barycentric_coordinates not available without real pytorch3d")
    """)

    # ── pytorch3d/renderer/mesh/shader.py ──
    write_file(f"{PKG}/renderer/mesh/shader.py", """\
        class SoftDepthShader:
            pass
        class HardFlatShader:
            pass
    """)

    # ── pytorch3d/renderer/mesh/textures.py ──
    write_file(f"{PKG}/renderer/mesh/textures.py", """\
        class Textures:
            pass
    """)

    # ── pytorch3d/renderer/utils.py ──
    write_file(f"{PKG}/renderer/utils.py", "pass\n")

    print(f"\n✅ 完成！pytorch3d 兼容包已创建在: {PKG}")


if __name__ == "__main__":
    main()
