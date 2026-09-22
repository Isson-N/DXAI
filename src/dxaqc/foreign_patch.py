"""Общие функции и модель патч-классификатора посторонних предметов."""
from __future__ import annotations
from pathlib import Path
import numpy as np

def read_image(path: Path) -> np.ndarray:
    import pydicom
    ds = pydicom.dcmread(path)
    arr = np.asarray(ds.pixel_array).astype(np.float32)
    if arr.ndim != 2: raise ValueError(f"Ожидался двумерный DICOM, получена форма {arr.shape}: {path}")
    if str(getattr(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        arr = (2 ** int(ds.BitsStored) - 1) - arr
    return normalize(arr)


def normalize(arr: np.ndarray) -> np.ndarray:
    """Нормировка по перцентилям 0,5/99,5 всего снимка. Одна функция на обучение и сервис:
    сервис передаёт уже прочитанные (инвертированные) пиксели, и без этой нормировки
    сеть получала бы значения в сотни раз больше тех, на которых училась."""
    arr = np.asarray(arr, dtype=np.float32)
    low, high = np.percentile(arr, (0.5, 99.5))
    arr = np.clip((arr-low)/(high-low), 0, 1) if high > low else np.zeros_like(arr)
    return arr.astype(np.float32, copy=False)

def grid_starts(length: int, size: int, stride: int) -> list[int]:
    if length <= size: return [0]
    vals = list(range(0, length-size+1, stride))
    if vals[-1] != length-size: vals.append(length-size)
    return vals

def extract_patch(image: np.ndarray, cx: float, cy: float, size: int) -> np.ndarray:
    x0, y0 = int(np.floor(cx-size/2)), int(np.floor(cy-size/2)); x1,y1=x0+size,y0+size
    top,left=max(0,-y0),max(0,-x0); bottom,right=max(0,y1-image.shape[0]),max(0,x1-image.shape[1])
    if top or bottom or left or right:
        padded=image; pads=[top,bottom,left,right]
        while any(pads):
            if padded.shape[0] < 2 or padded.shape[1] < 2: raise ValueError("reflect padding невозможен")
            t=min(pads[0],padded.shape[0]-1); b=min(pads[1],padded.shape[0]-1); l=min(pads[2],padded.shape[1]-1); r=min(pads[3],padded.shape[1]-1)
            padded=np.pad(padded,((t,b),(l,r)),mode="reflect"); pads=[pads[0]-t,pads[1]-b,pads[2]-l,pads[3]-r]
        return padded[y0+top:y1+top,x0+left:x1+left]
    return image[y0:y1,x0:x1]

def _torch():
    import torch
    return torch

class PatchNet:
    def __new__(cls,*a,**kw):
        torch=_torch(); nn=torch.nn
        class _Net(nn.Module):
            def __init__(self,width=32):
                super().__init__(); self.body=nn.Sequential(nn.Conv2d(1,width,3,padding=1),nn.BatchNorm2d(width),nn.ReLU(),nn.Conv2d(width,width,3,padding=1),nn.BatchNorm2d(width),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(width,width*2,3,padding=1),nn.BatchNorm2d(width*2),nn.ReLU(),nn.Conv2d(width*2,width*2,3,padding=1),nn.BatchNorm2d(width*2),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(width*2,width*4,3,padding=1),nn.BatchNorm2d(width*4),nn.ReLU(),nn.AdaptiveAvgPool2d(1)); self.head=nn.Linear(width*4,1)
            def forward(self,x): return self.head(self.body(x).flatten(1)).squeeze(-1)
        return _Net(*a,**kw)

class ForeignPatchModel:
    def __init__(self, models, threshold, size, stride, version, device): self.models,self.threshold,self.size,self.stride,self.version,self.device=models,threshold,size,stride,version,device
    @classmethod
    def load(cls,path,device="cpu"):
        torch=_torch(); d=torch.load(path,map_location=device); models=[]
        for state in d["models"]:
            m=PatchNet(); m.load_state_dict(state); m.to(device).eval(); models.append(m)
        return cls(models,float(d["threshold"]),int(d["size"]),int(d["stride"]),str(d.get("version","1")),device)
    def probability(self,pixels):
        torch=_torch(); arr=normalize(pixels); h,w=arr.shape; xs=grid_starts(w,self.size,self.stride); ys=grid_starts(h,self.size,self.stride); patches=np.stack([extract_patch(arr,x+self.size/2,y+self.size/2,self.size) for y in ys for x in xs]); vals=[]
        with torch.no_grad():
            for i in range(0,len(patches),256):
                x=torch.from_numpy(patches[i:i+256]).unsqueeze(1).to(self.device); vals.append(np.mean([torch.sigmoid(m(x)).cpu().numpy() for m in self.models],axis=0))
        return float(np.max(np.concatenate(vals)))
