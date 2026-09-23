#!/usr/bin/env python3
"""Generate rotated proximal-femur DRR projections with geometric QC."""
from __future__ import annotations
import argparse
from collections import Counter
from pathlib import Path
import warnings
import numpy as np

OUT_SHAPE = (265, 300)
PIXEL_MM = .6

def _deps():
    import nibabel as nib
    from scipy.ndimage import affine_transform, zoom
    return nib, affine_transform, zoom

def _load(path):
    nib, _, _ = _deps(); im = nib.as_closest_canonical(nib.load(str(path)))
    return im.get_fdata(dtype=np.float32), np.asarray(im.header.get_zooms()[:3], float), im.affine

def _rotation(axis, theta):
    u=np.asarray(axis,float); u/=np.linalg.norm(u); a=np.deg2rad(theta)
    K=np.array([[0,-u[2],u[1]],[u[2],0,-u[0]],[-u[1],u[0],0]])
    return np.eye(3)*np.cos(a)+(1-np.cos(a))*np.outer(u,u)+np.sin(a)*K

def _rotated(fem, bone, axis, center, theta, spacing, affine_transform):
    """Rotate in RAS mm; scipy requires the inverse output-to-input map."""
    S=np.diag(spacing); M=np.linalg.inv(S)@_rotation(axis,theta).T@S
    c=np.asarray(center)/spacing; offset=c-M@c
    kw=dict(matrix=M,offset=offset,output_shape=bone.shape,mode="constant",cval=0)
    return (affine_transform(bone,order=1,**kw),
            affine_transform(fem.astype(np.float32),order=1,**kw))

def _crop(a,x0,y0):
    out=np.zeros(OUT_SHAPE,a.dtype); y1=max(0,y0); y2=min(a.shape[0],y0+OUT_SHAPE[0]); x1=max(0,x0); x2=min(a.shape[1],x0+OUT_SHAPE[1])
    if y2>y1 and x2>x1: out[y1-y0:y2-y0,x1-x0:x2-x0]=a[y1:y2,x1:x2]
    return out

def _shaft_stats(mask):
    yy,xx=np.where(mask & (np.indices(mask.shape)[0]>=mask.shape[0]//2))
    if len(yy)<20: return None
    pts=np.column_stack((xx,yy)).astype(float); _,v=np.linalg.eigh(np.cov(pts,rowvar=False)); axis=v[:,-1]
    return float(np.degrees(np.arctan2(abs(axis[0]),abs(axis[1])))), np.array([xx.mean(),yy.mean()])*PIXEL_MM

def _one(ct,fem,hip,spacing,side,angles,flip_to,return_reason=False,min_len=100.0,long_len=130.0):
    """Rotate the femur about a line through the head centre parallel to the shaft.

    Anatomically the leg rotates in the hip joint, so the head stays in the acetabulum;
    the shaft axis alone cannot be estimated for femurs cut just below the trochanters.
    """
    _,affine_transform,zoom=_deps(); fem=fem>0; hip=hip>0
    def done(value,reason=None): return (value,reason) if return_reason else value
    if not fem.any(): return done([],"empty femur mask")
    coords=np.argwhere(fem)*spacing; z=coords[:,2]; lo,hi=z.min(),z.max()
    if hi-lo<min_len: return done([],f"femur shorter than {min_len:g} mm")
    shaft=coords[z<=lo+.3*(hi-lo)]; axis=np.linalg.eigh(np.cov(shaft,rowvar=False))[1][:,-1]
    if axis[2]<0: axis=-axis
    if hi-lo<long_len: axis=np.array([0.,0.,1.])  # short femur: shaft PCA unreliable, use S-I
    tilt=np.degrees(np.arccos(abs(axis[2])))
    if tilt>30: return done([],f"shaft tilt {tilt:.0f} deg from S-I")
    head=coords[z>=hi-25].mean(0)  # top 25 mm of the femur = femoral head
    shaft_c=shaft.mean(0)
    doflip=((head[0]-shaft_c[0])>0)!=(flip_to=="medial_right")
    bone=np.where(fem|hip,np.clip(ct,0,None),0).astype(np.float32); pelvis=np.where(hip&~fem,np.clip(ct,0,None),0)
    frames=[]; factors=(spacing[2]/PIXEL_MM,spacing[0]/PIXEL_MM)
    # frame anchored to the (fixed) head centre: head at 25% height, medial third
    hx=head[0]/PIXEL_MM; hy=(fem.shape[2]*spacing[2]-head[2])/PIXEL_MM
    for theta in angles:
        femb=np.where(fem,bone,0)
        rb,rm=_rotated(fem,femb,axis,head,theta,spacing,affine_transform)
        vol=pelvis+rb
        proj=(vol.sum(axis=1)*spacing[1]).T[::-1,:]; mask=(rm.max(axis=1)>.05).T[::-1,:]
        proj=zoom(proj,factors,order=1); mask=zoom(mask.astype(np.uint8),factors,order=0)>0
        if doflip: proj=proj[:,::-1]; mask=mask[:,::-1]; cx=proj.shape[1]-1-hx
        else: cx=hx
        # after flip the head is medial = image right for medial_right
        x0=int(round(cx-OUT_SHAPE[1]*(.70 if flip_to=="medial_right" else .30))); y0=int(round(hy-OUT_SHAPE[0]*.25))
        cm=_crop(mask,x0,y0)
        if cm.sum()<.5*mask.sum(): return done([],f"femur mostly outside frame at {theta:g} deg")
        frames.append((_crop(proj,x0,y0),float(theta)))
    from dxaqc.foreign_patch import normalize
    # mirroring swaps internal/external rotation: keep theta anatomically consistent
    sign=-1.0 if doflip else 1.0
    return done([(normalize(im),sign*t) for im,t in frames])

def main():
    p=argparse.ArgumentParser(); p.add_argument("--dir",type=Path,required=True); p.add_argument("--out",type=Path,required=True); p.add_argument("--angles",type=float,nargs="+",default=[-30,-20,-10,0,10,20,30]); p.add_argument("--max-cases",type=int); p.add_argument("--flip-to",choices=["medial_right","medial_left"],default="medial_right"); p.add_argument("--preview",type=Path); p.add_argument("--shard",default="0/1",help="i/n: process every n-th case starting at i")
    a=p.parse_args(); images=[]; thetas=[]; sides=[]; ids=[]; passed=0; skipped=Counter()
    k,n=map(int,a.shard.split("/"))
    for i,d in enumerate(sorted(a.dir.glob("s*"))):
        if a.max_cases is not None and i>=a.max_cases: break
        if i%n!=k: continue
        ct=None
        for side in ("left","right"):
            try:
                fem,sp,_=_load(d/f"segmentations/femur_{side}.nii.gz")
                zz=np.flatnonzero((fem>0).any(axis=(0,1)))
                if len(zz)==0 or (zz[-1]-zz[0]+1)*sp[2]<100: skipped["femur shorter than 100 mm (pre-check)"]+=1; continue
                hip,_,_=_load(d/f"segmentations/hip_{side}.nii.gz")
                if ct is None: ct,sp,_=_load(d/"ct.nii.gz")
            except Exception as exc: warnings.warn(f"{d.name} {side}: load failed: {exc}"); continue
            made,reason=_one(ct,fem,hip,sp,side,a.angles,a.flip_to,True)
            if reason: skipped[reason]+=1; warnings.warn(f"{d.name} {side}: skipped: {reason}"); continue
            passed+=1
            for im,t in made: images.append(im); thetas.append(t); sides.append("L" if side=="left" else "R"); ids.append(d.name)
    ims=np.asarray(images,np.float32).reshape((-1,*OUT_SHAPE)); a.out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.out,images=ims,theta=np.asarray(thetas,np.float32),side=np.asarray(sides),ct_id=np.asarray(ids))
    print(f"QC summary: passed={passed}, skipped={sum(skipped.values())}")
    for reason,count in sorted(skipped.items()): print(f"  skipped {count}: {reason}")
    if a.preview and images:
        from PIL import Image
        arr=np.asarray(images[:min(28,len(images))]); rows=[]
        for i in range(0,len(arr),7):
            row=arr[i:i+7]
            if len(row)<7: row=np.pad(row,((0,7-len(row)),(0,0),(0,0)))
            rows.append(np.concatenate(list(row),axis=1))
        a.preview.parent.mkdir(parents=True,exist_ok=True); Image.fromarray((np.concatenate(rows)*255).astype(np.uint8)).save(a.preview)

if __name__=="__main__": main()
