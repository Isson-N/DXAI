#!/usr/bin/env python3
"""Generate synthetic rotated hip DRR projections."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np

def _deps():
    import nibabel as nib
    from scipy.ndimage import affine_transform, zoom
    return nib, affine_transform, zoom

def _load(path):
    nib, _, _ = _deps(); im = nib.as_closest_canonical(nib.load(str(path)))
    return im.get_fdata(dtype=np.float32), np.asarray(im.header.get_zooms()[:3], dtype=float), im.affine

def _rotated(fem, bone, axis, center, theta, spacing, affine_transform):
    # arrays are z,y,x; transform in voxel coordinates (x,y,z)
    a = np.deg2rad(theta); u = axis / (np.linalg.norm(axis) + 1e-12)
    K = np.array([[0,-u[2],u[1]],[u[2],0,-u[0]],[-u[1],u[0],0]])
    R = np.eye(3)*np.cos(a) + (1-np.cos(a))*np.outer(u,u) + np.sin(a)*K
    # affine_transform operates array coords z,y,x; construct xyz then reverse
    Mxyz = R.T; cvox = center / spacing; offxyz = cvox - Mxyz @ cvox
    M = Mxyz[::-1, ::-1]; off = offxyz[::-1]
    return affine_transform(bone, M, offset=off, output_shape=bone.shape, order=1, mode="constant", cval=0), affine_transform(fem.astype(np.float32), M, offset=off, output_shape=fem.shape, order=1, mode="constant", cval=0)

def _one(ct, fem, hip, spacing, side, angles, flip_to):
    _, affine_transform, zoom = _deps()
    union = (fem > 0) | (hip > 0)
    if not union.any(): return []
    coords = np.argwhere(fem > 0)[:, [2,1,0]] * spacing
    s = coords[:,2]; lo, hi = np.percentile(s, [0,100]); cutoff = lo + .4*(hi-lo)
    low = coords[s <= cutoff];
    if (hi-lo) < 90: return []
    axis = np.linalg.eigh(np.cov(low, rowvar=False))[1][:, -1]; axis *= np.sign(axis[2] or 1)
    center = low.mean(0)
    bone = np.where(union, np.clip(ct, 0, None), 0).astype(np.float32)
    top = coords[s >= np.percentile(s,80)]; head = top.mean(0)
    # desired horizontal direction in RAS x
    want_right = flip_to == "medial_right"
    if ((head[0]-center[0]) > 0) != want_right:
        doflip = True
    else: doflip = False
    out=[]
    for theta in angles:
        rb, rm = _rotated(fem>0, bone, axis, center, theta, spacing, affine_transform)
        vol = np.where(hip>0, np.clip(ct,0,None), 0) + rb
        proj = vol.sum(axis=1) * spacing[1]  # z,x
        mproj = rm.max(axis=1) > .05
        sy,sx = np.where(mproj)
        if not len(sy): continue
        # resample physical pixel dimensions to 0.6 mm
        proj = zoom(proj, (spacing[2]/.6, spacing[0]/.6), order=1)
        mproj = zoom(mproj.astype(float), (spacing[2]/.6, spacing[0]/.6), order=0) > .1
        sy,sx=np.where(mproj); topy=int(sy.min()); top25=np.where(mproj & (np.indices(mproj.shape)[0] <= np.percentile(sy,25)))
        cx=float(np.mean(top25[1])) if len(top25[1]) else float(np.mean(sx)); cy=topy - 25/.6 + 132.5
        xcenter=cx
        if doflip: proj=proj[:,::-1]; xcenter=proj.shape[1]-1-xcenter
        x0=int(round(xcenter-150)); y0=int(round(cy-132.5)); outim=np.zeros((265,300),np.float32)
        yy0=max(0,y0); yy1=min(proj.shape[0],y0+265); xx0=max(0,x0); xx1=min(proj.shape[1],x0+300)
        outim[yy0-y0:yy1-y0,xx0-x0:xx1-x0]=proj[yy0:yy1,xx0:xx1]
        from dxaqc.foreign_patch import normalize
        out.append((normalize(outim), float(theta)))
    return out

def main():
    p=argparse.ArgumentParser(); p.add_argument("--dir",type=Path,required=True); p.add_argument("--out",type=Path,required=True); p.add_argument("--angles",type=float,nargs="+",default=[-30,-20,-10,0,10,20,30]); p.add_argument("--max-cases",type=int); p.add_argument("--flip-to",choices=["medial_right","medial_left"],default="medial_right"); p.add_argument("--preview",type=Path)
    a=p.parse_args(); images=[]; thetas=[]; sides=[]; ids=[]
    for i,d in enumerate(sorted(a.dir.glob("s*"))):
        if a.max_cases is not None and i>=a.max_cases: break
        try: ct,sp,_=_load(d/"ct.nii.gz")
        except Exception: continue
        for side in ("left","right"):
            try: fem,_,_=_load(d/f"segmentations/femur_{side}.nii.gz"); hip,_,_=_load(d/f"segmentations/hip_{side}.nii.gz")
            except Exception: continue
            for im,t in _one(ct,fem,hip,sp,side,a.angles,a.flip_to): images.append(im); thetas.append(t); sides.append("L" if side=="left" else "R"); ids.append(d.name)
    ims = np.asarray(images, np.float32).reshape((-1, 265, 300))
    a.out.parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(a.out,images=ims,theta=np.asarray(thetas,np.float32),side=np.asarray(sides),ct_id=np.asarray(ids))
    if a.preview and images:
        from PIL import Image,ImageDraw
        arr=np.asarray(images[:min(28,len(images))]); rows=[]
        for i in range(0,len(arr),7):
            row=arr[i:i+7]
            if len(row)<7: row=np.pad(row,((0,7-len(row)),(0,0),(0,0)))
            rows.append(np.concatenate(list(row),axis=1))
        sheet=np.concatenate(rows,axis=0); a.preview.parent.mkdir(parents=True,exist_ok=True); Image.fromarray((sheet*255).astype(np.uint8)).save(a.preview)
if __name__=="__main__": main()
