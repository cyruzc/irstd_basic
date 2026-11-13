"""DNANet (TGRS 2023) - Dense Nested Attention Network.

Cross-validated across PAL / LESPS / BasicIRSTD codebases; self-contained
(no imports outside torch).  Repository contract: forward(x) -> raw logits.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm2d

class ChannelAttention(nn.Module):
    """CBAM channel attention (max+avg pooled MLP)."""

    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """CBAM spatial attention (7x7 by default, as used in Res_CBAM_block)."""

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class Res_CBAM_block(nn.Module):
    """Residual block with CBAM (channel + spatial attention)."""

    def __init__(self, in_channels, out_channels, stride=1):
        super(Res_CBAM_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None

        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.ca(out) * out
        out = self.sa(out) * out
        out += residual
        out = self.relu(out)
        return out


class DNANet(nn.Module):
    """DNA-Net (dense nested attention network, DNANet-L config).
    forward() returns ONLY the final raw logits [B, 1, H, W]."""

    def __init__(self, in_channels=1, num_classes=1, block=Res_CBAM_block,
                 num_blocks=(2, 2, 2, 2), nb_filter=(16, 32, 64, 128, 256),
                 deep_supervision=True, mode="test"):
        super(DNANet, self).__init__()
        self.mode = mode
        self.relu = nn.ReLU(inplace=True)
        self.deep_supervision = deep_supervision
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.down = nn.Upsample(scale_factor=0.5, mode="bilinear", align_corners=True)

        self.up_4 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)
        self.up_8 = nn.Upsample(scale_factor=8, mode="bilinear", align_corners=True)
        self.up_16 = nn.Upsample(scale_factor=16, mode="bilinear", align_corners=True)

        self.conv0_0 = self._make_layer(block, in_channels, nb_filter[0])
        self.conv1_0 = self._make_layer(block, nb_filter[0], nb_filter[1], num_blocks[0])
        self.conv2_0 = self._make_layer(block, nb_filter[1], nb_filter[2], num_blocks[1])
        self.conv3_0 = self._make_layer(block, nb_filter[2], nb_filter[3], num_blocks[2])
        self.conv4_0 = self._make_layer(block, nb_filter[3], nb_filter[4], num_blocks[3])

        self.conv0_1 = self._make_layer(block, nb_filter[0] + nb_filter[1], nb_filter[0])
        self.conv1_1 = self._make_layer(block, nb_filter[1] + nb_filter[2] + nb_filter[0],
                                        nb_filter[1], num_blocks[0])
        self.conv2_1 = self._make_layer(block, nb_filter[2] + nb_filter[3] + nb_filter[1],
                                        nb_filter[2], num_blocks[1])
        self.conv3_1 = self._make_layer(block, nb_filter[3] + nb_filter[4] + nb_filter[2],
                                        nb_filter[3], num_blocks[2])

        self.conv0_2 = self._make_layer(block, nb_filter[0] * 2 + nb_filter[1], nb_filter[0])
        self.conv1_2 = self._make_layer(block, nb_filter[1] * 2 + nb_filter[2] + nb_filter[0],
                                        nb_filter[1], num_blocks[0])
        self.conv2_2 = self._make_layer(block, nb_filter[2] * 2 + nb_filter[3] + nb_filter[1],
                                        nb_filter[2], num_blocks[1])

        self.conv0_3 = self._make_layer(block, nb_filter[0] * 3 + nb_filter[1], nb_filter[0])
        self.conv1_3 = self._make_layer(block, nb_filter[1] * 3 + nb_filter[2] + nb_filter[0],
                                        nb_filter[1], num_blocks[0])

        self.conv0_4 = self._make_layer(block, nb_filter[0] * 4 + nb_filter[1], nb_filter[0])

        self.conv0_4_final = self._make_layer(block, nb_filter[0] * 5, nb_filter[0])

        self.conv0_4_1x1 = nn.Conv2d(nb_filter[4], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_3_1x1 = nn.Conv2d(nb_filter[3], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_2_1x1 = nn.Conv2d(nb_filter[2], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_1_1x1 = nn.Conv2d(nb_filter[1], nb_filter[0], kernel_size=1, stride=1)

        if self.deep_supervision:
            # Kept for parameter parity with the original implementation; only
            # final4 is used by forward() (single-output benchmark contract).
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def _make_layer(self, block, input_channels, output_channels, num_blocks=1):
        layers = []
        layers.append(block(input_channels, output_channels))
        for i in range(num_blocks - 1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0), self.down(x0_1)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0), self.down(x1_1)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1), self.down(x0_2)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0), self.down(x2_1)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1), self.down(x1_2)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2), self.down(x0_3)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        Final_x0_4 = self.conv0_4_final(
            torch.cat([self.up_16(self.conv0_4_1x1(x4_0)),
                       self.up_8(self.conv0_3_1x1(x3_1)),
                       self.up_4(self.conv0_2_1x1(x2_2)),
                       self.up(self.conv0_1_1x1(x1_3)), x0_4], 1))

        if self.deep_supervision:
            # Original (LESPS/BasicIRSTD): in train mode returns
            #   [final1(x0_1).sigmoid(), final2(x0_2).sigmoid(),
            #    final3(x0_3).sigmoid(), final4(Final_x0_4).sigmoid()]
            # and in test mode final4(Final_x0_4).sigmoid().
            # Here: single RAW-LOGIT final output only (benchmark contract).
            # TODO: deep supervision loss over [x0_1, x0_2, x0_3, Final_x0_4].
            output4 = self.final4(Final_x0_4)
            return output4
        else:
            output = self.final(Final_x0_4)
            return output


# ===========================================================================
# UIUNet
# ===========================================================================
# Paper: Xin Wu, Danfeng Hong, Jocelyn Chanussot, "UIU-Net: U-Net in U-Net for
#        Infrared Small Object Detection", IEEE TIP 2023, 32:364-376.
#        Official code: https://github.com/danfenghong/IEEE_TIP_UIU-Net
#
# Sources cross-validated:
#   * PAL:        model/UIU/UIU_no_sigmoid.py + model/UIU/fusion.py (UIU_No_Sigmoid)
#   * BasicIRSTD: model/UIUNet/model_UIUNet.py + model/UIUNet/fusion.py (UIUNET)
#
# Differences found across codebases:
#   1. in_ch default: PAL=3 (RGB), BasicIRSTD=1 (IR). Official repo default is
#      3 but the IRSTD benchmark feeds single-channel images. -> we use 1.
#   2. Sigmoid: BasicIRSTD returns F.sigmoid(d0), F.sigmoid(d1)... (7 outputs
#      in train mode, d0 in test mode); PAL removed the sigmoids.
#      -> RAW LOGITS, and only the fused output d0 is returned.
#   3. AsymBiChaFuseReduce norm: PAL = GroupNorm(1, C) (BN commented out);
#      BasicIRSTD = BatchNorm2d, byte-identical to the official UIU-Net repo
#      (verified against danfenghong/IEEE_TIP_UIU-Net model/fusion.py).
#      -> BatchNorm2d (canonical).
#   4. Architecture identical otherwise: outer U2-like U-Net (RSU7/6/5/4/4F
#      stages, 64/128/256/512/512 channels) whose decoder fuses skip + upsample
#      features with AsymBiChaFuseReduce (two output branches concatenated
#      before each decoder RSU stage), plus 6 side outputs fused by a 1x1
#      outconv into d0. ~50.5M params.
#   5. _upsample_like: F.upsample (deprecated) replaced by the equivalent
#      F.interpolate(..., mode='bilinear', align_corners=False).
# ===========================================================================

