"""LightUNet — lightweight UNet for infrared small target segmentation.

Encoder channels: [8, 16, 32], bottleneck: 64.
Input:  [B, 1, H, W]  grayscale image
Output: [B, 1, H, W]  logits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBNReLU(in_ch, out_ch),
            ConvBNReLU(out_ch, out_ch),
        )
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        skip = self.conv(x)
        out = self.pool(skip)
        return out, skip


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            ConvBNReLU(out_ch + skip_ch, out_ch),
            ConvBNReLU(out_ch, out_ch),
        )

    def forward(self, x, skip):
        x = self.up(x)
        # handle size mismatch from pooling
        dh = skip.size(2) - x.size(2)
        dw = skip.size(3) - x.size(3)
        x = F.pad(x, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class LightUNet(nn.Module):
    def __init__(self, in_channels=1, encoder_channels=(8, 16, 32), bottleneck=64):
        super().__init__()
        enc = list(encoder_channels)

        # encoder
        self.enc1 = EncoderBlock(in_channels, enc[0])
        self.enc2 = EncoderBlock(enc[0], enc[1])
        self.enc3 = EncoderBlock(enc[1], enc[2])

        # bottleneck
        self.bottleneck = nn.Sequential(
            ConvBNReLU(enc[2], bottleneck),
            ConvBNReLU(bottleneck, bottleneck),
        )

        # decoder
        self.dec3 = DecoderBlock(bottleneck, enc[2], enc[2])
        self.dec2 = DecoderBlock(enc[2], enc[1], enc[1])
        self.dec1 = DecoderBlock(enc[1], enc[0], enc[0])

        # head
        self.head = nn.Conv2d(enc[0], 1, kernel_size=1)

    def forward(self, x):
        x, s1 = self.enc1(x)
        x, s2 = self.enc2(x)
        x, s3 = self.enc3(x)

        x = self.bottleneck(x)

        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)

        return self.head(x)


if __name__ == "__main__":
    model = LightUNet()
    inp = torch.randn(2, 1, 256, 256)
    out = model(inp)
    print(f"Input:  {inp.shape}")
    print(f"Output: {out.shape}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
