"""ACM — Asymmetric Spatial Context-aware ResUNet.

Ported from BasicIRSTD/model/ACM/.
Input:  [B, 1, H, W]  grayscale image
Output: [B, 1, H, W]  logits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm2d
from torchvision.models.resnet import BasicBlock


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class AsymBiChaFuse(nn.Module):
    """Asymmetric Bidirectional Channel Fusion."""

    def __init__(self, channels=64, r=4):
        super().__init__()
        self.channels = channels
        mid = channels // r

        self.topdown = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid, 1),
            nn.BatchNorm2d(mid, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1),
            nn.BatchNorm2d(channels, momentum=0.9),
            nn.Sigmoid(),
        )

        self.bottomup = nn.Sequential(
            nn.Conv2d(channels, mid, 1),
            nn.BatchNorm2d(mid, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1),
            nn.BatchNorm2d(channels, momentum=0.9),
            nn.Sigmoid(),
        )

        self.post = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels, momentum=0.9),
            nn.ReLU(inplace=True),
        )

    def forward(self, xh, xl):
        td = self.topdown(xh)
        bu = self.bottomup(xl)
        xs = 2 * torch.mul(xl, td) + 2 * torch.mul(xh, bu)
        return self.post(xs)


class _FCNHead(nn.Module):
    def __init__(self, in_channels, channels, momentum=0.9):
        super().__init__()
        inter = in_channels // 4
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, inter, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter, momentum=momentum),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(inter, channels, 1),
        )

    def forward(self, x):
        return self.block(x)


class ACM(nn.Module):
    """ASKCResUNet for infrared small target detection."""

    def __init__(
        self,
        in_channels=1,
        layers=None,
        channels=None,
        fuse_mode="AsymBi",
        tiny=False,
        classes=1,
    ):
        super().__init__()
        if layers is None:
            layers = [3, 3, 3]
        if channels is None:
            channels = [8, 16, 32, 64]

        self.tiny = tiny
        self.momentum = 0.9
        stem_width = channels[0]

        if tiny:
            self.stem = nn.Sequential(
                nn.BatchNorm2d(in_channels, momentum=self.momentum),
                nn.Conv2d(in_channels, stem_width * 2, 3, 1, 1, bias=False),
                nn.BatchNorm2d(stem_width * 2, momentum=self.momentum),
                nn.ReLU(inplace=True),
            )
        else:
            self.stem = nn.Sequential(
                nn.BatchNorm2d(in_channels, momentum=self.momentum),
                nn.Conv2d(in_channels, stem_width, 3, 2, 1, bias=False),
                nn.BatchNorm2d(stem_width, momentum=self.momentum),
                nn.ReLU(inplace=True),
                nn.Conv2d(stem_width, stem_width, 3, 1, 1, bias=False),
                nn.BatchNorm2d(stem_width, momentum=self.momentum),
                nn.ReLU(inplace=True),
                nn.Conv2d(stem_width, stem_width * 2, 3, 1, 1, bias=False),
                nn.BatchNorm2d(stem_width * 2, momentum=self.momentum),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(3, 2, 1),
            )

        self.layer1 = self._make_layer(BasicBlock, layers[0], channels[1], channels[1], 1)
        self.layer2 = self._make_layer(BasicBlock, layers[1], channels[2], channels[1], 2)
        self.layer3 = self._make_layer(BasicBlock, layers[2], channels[3], channels[2], 2)

        self.deconv2 = nn.ConvTranspose2d(channels[3], channels[2], 4, 2, 1)
        self.uplayer2 = self._make_layer(BasicBlock, layers[1], channels[2], channels[2], 1)
        self.fuse2 = AsymBiChaFuse(channels[2])

        self.deconv1 = nn.ConvTranspose2d(channels[2], channels[1], 4, 2, 1)
        self.uplayer1 = self._make_layer(BasicBlock, layers[0], channels[1], channels[1], 1)
        self.fuse1 = AsymBiChaFuse(channels[1])

        self.head = _FCNHead(channels[1], classes, self.momentum)
        self.inplanes = channels[1]

    def _make_layer(self, block, blocks, out_channels, in_channels, stride):
        downsample = None
        if stride != 1 or out_channels != in_channels:
            downsample = nn.Sequential(
                conv1x1(in_channels, out_channels, stride),
                nn.BatchNorm2d(out_channels * block.expansion, momentum=self.momentum),
            )
        layers = [block(in_channels, out_channels, stride, downsample, 1, norm_layer=nn.BatchNorm2d)]
        self.inplanes = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, out_channels, 1, norm_layer=nn.BatchNorm2d))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)

        deconvc2 = self.deconv2(c3)
        fusec2 = self.fuse2(deconvc2, c2)
        upc2 = self.uplayer2(fusec2)

        deconvc1 = self.deconv1(upc2)
        fusec1 = self.fuse1(deconvc1, c1)
        upc1 = self.uplayer1(fusec1)

        pred = self.head(upc1)

        if self.tiny:
            out = pred
        else:
            out = F.interpolate(pred, scale_factor=4, mode="bilinear")

        return out


if __name__ == "__main__":
    model = ACM(in_channels=1)
    inp = torch.randn(2, 1, 256, 256)
    out = model(inp)
    print(f"Input:  {inp.shape}")
    print(f"Output: {out.shape}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
