"""RPAttn head C (Le, SOICT 2025, doi 10.1007/978-981-92-2590-3_26) on a timm backbone: reverse-pyramid fusion to the deepest stride, conv and pooled-key/value attention paths, spatial gate, GeM tail."""
import timm
import torch
from torch import nn


class SE(nn.Module):
    def __init__(self, d, r=16):
        super().__init__()
        self.f = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(d, d // r, 1), nn.SiLU(), nn.Conv2d(d // r, d, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.f(x)


def block(a, b, k=3, s=1, d=1):
    return nn.Sequential(nn.Conv2d(a, b, k, s, d * (k // 2), dilation=d, bias=False), nn.BatchNorm2d(b), nn.SiLU())


class RPAttnHead(nn.Module):
    def __init__(self, dims, num_classes, D=512, heads=8, p=3.0):
        super().__init__()
        d3, d4, d5 = dims
        self.p3 = nn.Sequential(block(d3, d3, s=2), block(d3, d3, s=2), block(d3, d3, d=2), SE(d3))
        self.p4 = nn.Sequential(block(d4, d4, s=2), block(d4, d4, d=2), SE(d4))
        self.p5 = nn.Sequential(block(d5, d5, d=3), SE(d5))
        self.fuse = nn.Sequential(block(d3 + d4 + d5, D, k=1), block(D, D))
        self.conv = nn.Sequential(block(D, D, k=1), block(D, D))
        self.pool = nn.AvgPool2d(2)
        self.attn = nn.MultiheadAttention(D, heads, batch_first=True)
        self.norm = nn.LayerNorm(D)
        self.mlp = nn.Sequential(nn.Linear(D, 4 * D), nn.SiLU(), nn.Linear(4 * D, D))
        self.gate = nn.Sequential(nn.Conv2d(2 * D, D, 1), nn.BatchNorm2d(D), nn.Sigmoid())
        self.gem = nn.Parameter(torch.tensor(p))
        self.tail = nn.Sequential(nn.BatchNorm1d(D), nn.Dropout(.3), nn.Linear(D, num_classes))

    def forward(self, c3, c4, c5):
        z = self.fuse(torch.cat([self.p3(c3), self.p4(c4), self.p5(c5)], 1))
        b, d, h, w = z.shape
        q, kv = z.flatten(2).transpose(1, 2), self.pool(z).flatten(2).transpose(1, 2)
        a = q + self.attn(q, kv, kv, need_weights=False)[0]
        a = self.norm(a + self.mlp(a)).transpose(1, 2).reshape(b, d, h, w)
        c = self.conv(z)
        g = self.gate(torch.cat([c, a], 1))
        f = g * a + (1 - g) * c
        p = self.gem.clamp_min(1.0)
        return self.tail(f.clamp_min(1e-6).pow(p).mean((2, 3)).pow(1 / p))


class RPAttnNet(nn.Module):
    def __init__(self, timm_name, num_classes, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(timm_name, pretrained=pretrained, features_only=True, out_indices=(1, 2, 3))
        self.head = RPAttnHead(self.backbone.feature_info.channels(), num_classes)

    def get_classifier(self):
        return self.head

    def forward(self, x):
        return self.head(*self.backbone(x))
