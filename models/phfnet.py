"""PHFNet - "A Parallel Hybrid Fusion Network with Linear Attention for
Infrared Small Target Detection".

Ported from the official code (`repo/PHFNet/model/{HCFnet_xishu,main_blocks1}.py`,
class ``UCTNet``).  Self-contained (only torch).

Modifications at port time:
    - ``forward`` returns raw logits [B, 1, H, W] with NO internal sigmoid for
      the default ``gt_ds=False`` path (matches the official code, which only
      sigmoids inside the ``gt_ds=True`` branch).
    - Dropped the unused ``Down_wt`` class and its ``pytorch_wavelets`` import
      (defined but never instantiated in the official model).  ``timm``'s
      ``DropPath`` / ``trunc_normal_`` are inlined so no timm dependency remains.
    - ``in_features`` defaults to 1 (grayscale) instead of 3; ``shift_size`` and
      ``drop`` are constructor args (official IRSTD-1K uses shift=4/drop=0.1;
      SIRST uses shift=6/drop=0.5).
    - ``gt_ds`` branch kept: returns the 6 scoped outputs (sigmoeded) as in the
      official train path.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    return nn.init.trunc_normal_(tensor, mean=mean, std=std, a=a, b=b)


class DropPath(nn.Module):
    """Stochastic depth per sample (identity when drop_prob=0)."""

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x.div(keep) * mask


class conv_block(nn.Module):
    def __init__(self, in_features, out_features, kernel_size=(3, 3), stride=(1, 1),
                 padding=(1, 1), dilation=(1, 1), norm_type="bn", activation=True,
                 use_bias=True, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_features, out_features, kernel_size, stride,
                              padding, dilation, bias=use_bias, groups=groups)
        self.norm_type = norm_type
        self.act = activation
        if norm_type == "gn":
            self.norm = nn.GroupNorm(32 if out_features >= 32 else out_features, out_features)
        if norm_type == "bn":
            self.norm = nn.BatchNorm2d(out_features)
        if self.act:
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        if self.norm_type is not None:
            x = self.norm(x)
        if self.act:
            x = self.relu(x)
        return x


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = nn.Conv2d(2, 1, 7, 1, 3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = self.sigmoid(self.conv2d(torch.cat([avgout, maxout], dim=1)))
        return out * x


class DWCov(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Sequential(nn.Conv2d(dim, dim, 3, padding=1, groups=dim))

    def forward(self, x):
        return self.dwconv(x)


class shiftmlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.0, shift_size=5):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dim = in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwcov = DWCov(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.shift_size = shift_size
        self.pad = shift_size // 2
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels // m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        B, C, H, W = x.shape
        xn = F.pad(x, (self.pad,) * 4, "constant", 0)
        xs = torch.chunk(xn, self.shift_size, 1)
        x_shift = [torch.roll(x_c, shift, 2) for x_c, shift in zip(xs, range(-self.pad, self.pad + 1))]
        x_cat = torch.cat(x_shift, 1)
        x_cat = torch.narrow(x_cat, 2, self.pad, H)
        x_s = torch.narrow(x_cat, 3, self.pad, W)
        x = self.fc1(x_s)
        # Order matches the official implementation (dwcov -> act -> drop);
        # numerically identical in eval, divergent only during training.
        x = self.drop(self.act(self.dwcov(x)))

        xn = F.pad(x, (self.pad,) * 4, "constant", 0)
        xs = torch.chunk(xn, self.shift_size, 1)
        x_shift = [torch.roll(x_c, shift, 3) for x_c, shift in zip(xs, range(-self.pad, self.pad + 1))]
        x_cat = torch.cat(x_shift, 1)
        x_cat = torch.narrow(x_cat, 2, self.pad, H)
        x_s = torch.narrow(x_cat, 3, self.pad, W)
        return self.drop(self.fc2(x_s))


class shiftedBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, drop=0.0, shift_size=4,
                 drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = shiftmlp(in_features=dim, hidden_features=mlp_hidden_dim,
                            act_layer=act_layer, drop=drop, shift_size=shift_size)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels // m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        return x + self.drop_path(self.mlp(self.norm2(x)))


class BMSA(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.pool_h1 = nn.AdaptiveMaxPool2d((None, 1))
        self.pool_w1 = nn.AdaptiveMaxPool2d((1, None))
        self.conv1x1 = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(4, channels), nn.Sigmoid())
        self.conv5x5 = nn.Sequential(
            nn.Conv1d(channels, channels, 5, padding=2, bias=False),
            nn.GroupNorm(4, channels), nn.Sigmoid())

    def forward(self, x):
        b, c, h, w = x.size()
        x_h = self.conv1x1(self.pool_h(x).reshape((b, c, h))).reshape((b, c, h, 1))
        x_w = self.conv1x1(self.pool_w(x).reshape((b, c, w))).reshape((b, c, 1, w))
        temp1 = x * x_h * x_w
        x_h1 = self.conv5x5(self.pool_h1(x).reshape((b, c, h))).reshape((b, c, h, 1))
        x_w1 = self.conv5x5(self.pool_w1(x).reshape((b, c, w))).reshape((b, c, 1, w))
        temp2 = x * x_h1 * x_w1
        return temp1 + temp2


class TLA(nn.Module):
    """Local-Compensated Linear Attention."""

    def __init__(self, in_ch=256):
        super().__init__()
        self.query = nn.Sequential(nn.Conv2d(in_ch, in_ch, 1), nn.GELU(),
                                   nn.Conv2d(in_ch, in_ch, 1), nn.Softplus())
        self.key = nn.Sequential(nn.Conv2d(in_ch, in_ch, 1), nn.GELU(),
                                 nn.Conv2d(in_ch, in_ch, 1), nn.Softplus())
        self.value = nn.Sequential(nn.Conv2d(in_ch, in_ch, 1), nn.GELU())
        self.gate = nn.Sequential(nn.Conv2d(in_ch, in_ch, 1), nn.GELU())
        self.output_linear = nn.Conv2d(in_ch, in_ch, 1)
        self.norm = nn.InstanceNorm2d(num_features=in_ch)
        self.dwc = nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch)
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        residual = x
        x = self.norm(x)
        B, C, H, W = x.size()
        q = self.query(x); k = self.key(x); v = self.value(x); g = self.gate(x)
        v_residual = v
        q = q.view(B, C, H * W).contiguous().permute(0, 2, 1).contiguous()
        k = k.view(B, C, H * W).contiguous()
        v = v.view(B, C, H * W).contiguous().permute(0, 2, 1).contiguous()
        kv = torch.einsum("bcn, bnd -> bcd", k, v)
        z = torch.einsum("bnc,bc -> bn", q, k.sum(dim=-1)) / math.sqrt(C)
        z = 1.0 / (z + H * W)
        out = torch.einsum("bnc, bcd-> bnd", q, kv)
        out = out / math.sqrt(C)
        out = out + v
        out = torch.einsum("bnc, bn -> bnc", out, z)
        out = out.permute(0, 2, 1).contiguous().view(B, C, H, W)
        v_dwc = self.dwc(v_residual)
        out = out + self.alpha * v_dwc
        out = out * g
        out = self.output_linear(out)
        return out + residual


class PPA_xishu(nn.Module):
    """Parallel Perception Aggregation block (CNN + linear attention + shifted MLP)."""

    def __init__(self, in_features, filters, shift_size=4, drop=0.1):
        super().__init__()
        self.skip = conv_block(in_features, filters, (1, 1), (1, 1), (0, 0),
                               norm_type="bn", activation=False)
        self.c1 = conv_block(in_features, filters, (3, 3), (1, 1), (1, 1),
                             norm_type="bn", activation=True)
        self.c2 = conv_block(filters, filters, (3, 3), (1, 1), (1, 1),
                             norm_type="bn", activation=True)
        self.c3 = conv_block(filters, filters, (3, 3), (1, 1), (1, 1),
                             norm_type="bn", activation=True)
        self.sa = SpatialAttentionModule()
        self.cn = BMSA(filters)
        self.SMLP = shiftedBlock(dim=filters, mlp_ratio=1, drop=0.5, drop_path=0,
                                 norm_layer=nn.BatchNorm2d, shift_size=shift_size)
        self.TLA = TLA(in_ch=filters)
        self.bn1 = nn.BatchNorm2d(filters)
        self.drop = nn.Dropout2d(drop)
        self.relu = nn.ReLU()

    def forward(self, x):
        x_skip = self.skip(x)
        x_lga2 = self.TLA(x_skip)
        x_lga2 = self.SMLP(x_lga2)
        x1 = self.c1(x)
        x2 = self.c2(x1)
        x3 = self.c3(x2)
        x = x1 + x2 + x3 + x_skip + x_lga2
        x = self.cn(x)
        x = self.sa(x)
        x = self.drop(x)
        x = self.bn1(x)
        return self.relu(x)


class UCTNet(nn.Module):
    """PHFNet.  forward() returns raw logits [B, 1, H, W] when gt_ds=False."""

    def __init__(self, in_features=1, out_features=1, gt_ds=False, mode="train",
                 shift_size=4, drop=0.1):
        super().__init__()
        self.gt_ds = gt_ds
        self.mode = mode
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.p1 = PPA_xishu(in_features=in_features, filters=32, shift_size=shift_size, drop=drop)
        self.p2 = PPA_xishu(in_features=32, filters=64, shift_size=shift_size, drop=drop)
        self.p3 = PPA_xishu(in_features=64, filters=128, shift_size=shift_size, drop=drop)
        self.p4 = PPA_xishu(in_features=128, filters=256, shift_size=shift_size, drop=drop)
        self.p5 = PPA_xishu(in_features=256, filters=512, shift_size=shift_size, drop=drop)
        self.p6 = PPA_xishu(in_features=512, filters=256, shift_size=shift_size, drop=drop)
        self.p7 = PPA_xishu(in_features=256, filters=128, shift_size=shift_size, drop=drop)
        self.p8 = PPA_xishu(in_features=128, filters=64, shift_size=shift_size, drop=drop)
        self.p9 = PPA_xishu(in_features=64, filters=32, shift_size=shift_size, drop=drop)

        self.up1 = nn.Sequential(nn.ConvTranspose2d(512, 256, 2, 2), nn.BatchNorm2d(256), nn.ReLU())
        self.up2 = nn.Sequential(nn.ConvTranspose2d(256, 128, 2, 2), nn.BatchNorm2d(128), nn.ReLU())
        self.up3 = nn.Sequential(nn.ConvTranspose2d(128, 64, 2, 2), nn.BatchNorm2d(64), nn.ReLU())
        self.up4 = nn.Sequential(nn.ConvTranspose2d(64, 32, 2, 2), nn.BatchNorm2d(32), nn.ReLU())

        self.out = nn.Conv2d(32, out_features, 1)
        self.out1 = nn.Conv2d(512, out_features, 1)
        self.out2 = nn.Conv2d(256, out_features, 1)
        self.out3 = nn.Conv2d(128, out_features, 1)
        self.out4 = nn.Conv2d(64, out_features, 1)
        self.outconv = nn.Conv2d(5, 1, 1)

    def forward(self, x):
        x1 = self.p1(x)
        x2 = self.p2(self.maxpool(x1))
        x3 = self.p3(self.maxpool(x2))
        x4 = self.p4(self.maxpool(x3))
        x = self.p5(self.maxpool(x4))
        x1_res, x2_res, x3_res, x4_res = x1, x2, x3, x4

        out4 = F.interpolate(self.out1(x), scale_factor=16, mode="bilinear", align_corners=True)
        x = self.up1(x)
        x = self.p6(torch.cat((x, x4_res), dim=1))

        out3 = F.interpolate(self.out2(x), scale_factor=8, mode="bilinear", align_corners=True)
        x = self.up2(x)
        x = self.p7(torch.cat((x, x3_res), dim=1))

        out2 = F.interpolate(self.out3(x), scale_factor=4, mode="bilinear", align_corners=True)
        x = self.up3(x)
        x = self.p8(torch.cat((x, x2_res), dim=1))

        out1 = F.interpolate(self.out4(x), scale_factor=2, mode="bilinear", align_corners=True)
        x = self.up4(x)
        x = self.p9(torch.cat((x, x1_res), dim=1))

        out = self.out(x)  # raw logits

        if self.gt_ds:
            d0 = self.outconv(torch.cat((out1, out2, out3, out4, out), 1))
            if self.mode == "train":
                return tuple(torch.sigmoid(t) for t in (out4, out3, out2, out1, d0, out))
            return out
        return out
