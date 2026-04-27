"""Generic lightweight UNet and named width variants used in the benchmark."""

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
        dh = skip.size(2) - x.size(2)
        dw = skip.size(3) - x.size(3)
        x = F.pad(x, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        encoder_channels=(8, 16, 32),
        bottleneck_channels=None,
        head_channels=None,
    ):
        super().__init__()

        if len(encoder_channels) != 3:
            raise ValueError(
                f"UNet expects exactly 3 encoder stages, got {len(encoder_channels)}"
            )

        enc = list(encoder_channels)
        bottleneck = bottleneck_channels or (enc[-1] * 2)
        head_in_channels = head_channels or enc[0]

        self.enc1 = EncoderBlock(in_channels, enc[0])
        self.enc2 = EncoderBlock(enc[0], enc[1])
        self.enc3 = EncoderBlock(enc[1], enc[2])

        self.bottleneck = nn.Sequential(
            ConvBNReLU(enc[2], bottleneck),
            ConvBNReLU(bottleneck, bottleneck),
        )

        self.dec3 = DecoderBlock(bottleneck, enc[2], enc[2])
        self.dec2 = DecoderBlock(enc[2], enc[1], enc[1])
        self.dec1 = DecoderBlock(enc[1], enc[0], enc[0])

        if head_in_channels == enc[0]:
            self.head_proj = nn.Identity()
        else:
            self.head_proj = nn.Conv2d(enc[0], head_in_channels, kernel_size=1)

        self.head = nn.Conv2d(head_in_channels, out_channels, kernel_size=1)

    def forward_features(self, x):
        x, s1 = self.enc1(x)
        x, s2 = self.enc2(x)
        x, s3 = self.enc3(x)

        bottleneck = self.bottleneck(x)

        dec3 = self.dec3(bottleneck, s3)
        dec2 = self.dec2(dec3, s2)
        dec1 = self.dec1(dec2, s1)
        head_input = self.head_proj(dec1)
        logits = self.head(head_input)

        return {
            "skip1": s1,
            "skip2": s2,
            "skip3": s3,
            "bottleneck": bottleneck,
            "dec3": dec3,
            "dec2": dec2,
            "dec1": dec1,
            "head_input": head_input,
            "logits": logits,
        }

    def forward(self, x):
        return self.forward_features(x)["logits"]


class LightUNet(UNet):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            encoder_channels=(8, 16, 32),
        )


class MiniUNet(UNet):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            encoder_channels=(4, 8, 16),
        )

class MiniUNetHead2(UNet):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            encoder_channels=(4, 8, 16),
            head_channels=2,
        )

class MicroUNet(UNet):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            encoder_channels=(2, 4, 8),
        )

if __name__ == "__main__":
    variants = {
        "LightUNet": LightUNet(),
        "MiniUNet": MiniUNet(),
        "MiniUNetHead2": MiniUNetHead2(),
        "MicroUNet": MicroUNet(),
    }
    inp = torch.randn(2, 1, 256, 256)
    for name, model in variants.items():
        out = model(inp)
        print(f"{name}: output={tuple(out.shape)} params={sum(p.numel() for p in model.parameters()):,}")
