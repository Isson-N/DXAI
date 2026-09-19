"""Архитектуры моделей проекта. Одно определение на обучение и на сервис.

U-Net с энкодером resnet18: 8 тепловых карт (центры Th12–L5 и две вершины гребней)
плюс три классификационные головы (состояние каждого гребня, видимость половины Th12).
"""
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class KeypointNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        try:
            self.encoder = timm.create_model(
                "resnet18", features_only=True, pretrained=pretrained, in_chans=3
            )
        except Exception as e:
            warnings.warn(f"Не удалось загрузить веса timm: {e}")
            self.encoder = timm.create_model(
                "resnet18", features_only=True, pretrained=False, in_chans=3
            )
        ch = self.encoder.feature_info.channels()
        self.d4 = ConvBlock(ch[4] + ch[3], 256)
        self.d3 = ConvBlock(256 + ch[2], 128)
        self.d2 = ConvBlock(128 + ch[1], 64)
        self.d1 = ConvBlock(64 + ch[0], 32)
        self.out = nn.Conv2d(32, 8, 1)
        self.cls_left = nn.Linear(ch[4], 3)
        self.cls_right = nn.Linear(ch[4], 3)
        self.cls_th12 = nn.Linear(ch[4], 3)

    def forward(self, x):
        fs = self.encoder(x)
        z = fs[-1]
        z = F.interpolate(z, size=fs[-2].shape[-2:], mode="bilinear", align_corners=False)
        z = self.d4(torch.cat([z, fs[-2]], 1))
        z = F.interpolate(z, size=fs[-3].shape[-2:], mode="bilinear", align_corners=False)
        z = self.d3(torch.cat([z, fs[-3]], 1))
        z = F.interpolate(z, size=fs[-4].shape[-2:], mode="bilinear", align_corners=False)
        z = self.d2(torch.cat([z, fs[-4]], 1))
        z = F.interpolate(z, size=fs[-5].shape[-2:], mode="bilinear", align_corners=False)
        z = self.d1(torch.cat([z, fs[-5]], 1))
        hm = self.out(z)
        pooled = F.adaptive_avg_pool2d(fs[-1], 1).flatten(1)
        return hm, torch.stack([self.cls_left(pooled), self.cls_right(pooled), self.cls_th12(pooled)], 1)
