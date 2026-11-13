"""SDS-Net (TGRS 2025) - "SDS-Net: Shallow-Deep Synergism-detection Network
for infrared dim and small target detection" (arXiv:2506.06042, official
code: https://github.com/PhysiLearn/SDS-Net).

Ported from the PAL codebase (model/SDSNet/SDSNet_no_sigmoid.py,
class SDSNet_No_Sigmoid).  Self-contained (imports: torch / stdlib only).

Dependencies inlined at port time:
    - ``ml_collections.ConfigDict`` (used only for a handful of scalar
      hyper-parameters) -> a minimal local config namespace.
    - ``einops.rearrange`` -> equivalent reshape/transpose one-liners
      (``_rearrange_bchw_to_b1cn``, ``to_3d``/``to_4d`` helpers), keeping
      the exact same tensor layouts.
    - ``thop`` was only used in the original ``__main__`` profiling block
      and is dropped.

Other modifications at port time:
    - Constructor signature changed to ``SDSNet(in_channels=1,
      n_classes=1)``; the original ``mode`` / ``deepsuper`` flags are
      removed because forward() always takes the single-output
      (``mode='test'``) path: it returns the raw logit map ``out`` of
      shape [B, 1, H, W] with no internal sigmoid and no deep-supervision
      auxiliary outputs.
    - The deep-supervision heads (``gt_conv1/2/3``, ``outconv``) are kept
      as submodules so the parameter count and state_dict layout match the
      PAL model, but they are never evaluated by forward().
    - Dead helper classes never instantiated by the model
      (``UpFusionBlock``, ``MDFA`` and its channel/spatial/position
      attention submodules) were not carried over.
    - The original ``print('Deep-Supervision: ...')`` on construction was
      removed.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair


# ---------------------------------------------------------------------------
# Inlined replacements for the PAL file's external dependencies
# ---------------------------------------------------------------------------
def to_3d(x):
    # einops: rearrange(x, 'b c h w -> b (h w) c')
    return x.flatten(2).transpose(1, 2)


def to_4d(x, h, w):
    # einops: rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
    b, _, c = x.shape
    return x.transpose(1, 2).reshape(b, c, h, w)


def _head1_chw(x):
    # einops: rearrange(x, 'b (head c) h w -> b head c (h w)', head=1)
    b, c, h, w = x.shape
    return x.reshape(b, 1, c, h * w)


class _SDSConfig(object):
    """Minimal stand-in for the original ml_collections.ConfigDict tree."""

    def __init__(self):
        transformer = {
            "num_heads": 4,
            "num_layers": 4,
            "embeddings_dropout_rate": 0.1,
            "attention_dropout_rate": 0.1,
            "dropout_rate": 0,
        }
        self.transformer = transformer
        self.KV_size = 224
        self.patch_sizes = [16, 8, 4]
        self.base_channel = 32
        self.n_classes = 1


# ---------------------------------------------------------------------------
# Model components (architecture preserved from the PAL source)
# ---------------------------------------------------------------------------
class Backbone(nn.Module):
    def __init__(self, n_channels, in_channels, num_blocks=3):
        """Backbone with downsampling blocks and a fixed final block for d4."""
        super(Backbone, self).__init__()
        self.num_blocks = num_blocks - 1
        self.down_blocks = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)

        for i in range(num_blocks):
            input_channels = n_channels if i == 0 else in_channels * (2 ** (i - 1))
            output_channels = in_channels * (2 ** i)
            self.down_blocks.append(Res_block(input_channels, output_channels))

        self.final_block = Res_block(in_channels * (2 ** (num_blocks - 1)), in_channels * (2 ** (num_blocks - 1)))

    def forward(self, x):
        features = []
        for block in self.down_blocks:
            x = block(x)
            features.append(x)
            x = self.pool(x)

        d3 = self.final_block(x)
        features.append(d3)

        return features


class Embeddings(nn.Module):
    def __init__(self, patchsize, in_channels):
        super().__init__()

        patch_size = _pair(patchsize)

        self.patch_embeddings = nn.Conv2d(in_channels=in_channels,
                                          out_channels=in_channels,
                                          kernel_size=patch_size,
                                          stride=patch_size)

    def forward(self, x):
        x = self.patch_embeddings(x)
        return x


class FeatrueMapping(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super(FeatrueMapping, self).__init__()
        if kernel_size == 3:
            padding = 1
        else:
            padding = 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor

    def forward(self, x):
        x = nn.Upsample(scale_factor=self.scale_factor, mode='bilinear')(x)

        out = self.conv(x)
        out = self.norm(out)
        out = self.activation(out)
        return out


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm3d(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm3d, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class Shallow_branch(nn.Module):
    def __init__(self, channel_num):
        super(Shallow_branch, self).__init__()

        channel_all = sum(channel_num)
        self.norms = nn.ModuleList([LayerNorm3d(c, LayerNorm_type='WithBias') for c in channel_num])
        self.attn_norm = LayerNorm3d(channel_all, LayerNorm_type='WithBias')
        self.msca = MSCA(channel_num)
        self.mdfa = nn.ModuleList([CBAMLayer(c) for c in channel_num])

    def process_residual(self, cx, norm, ffn, org):
        cx = cx + org
        cx_org = cx
        x = norm(cx)
        x = ffn(x)
        x = x + cx_org
        return x

    def forward(self, E1, E2, E3):
        E4 = [E for E in [E1, E2, E3] if E is not None]
        E4 = torch.cat(E4, dim=1)
        cx1, cx2, cx3 = self.msca(E1, E2, E3, E4)
        cxs = [cx1, cx2, cx3]
        norms = self.norms
        mdfas = self.mdfa
        orgs = [E1, E2, E3]
        p_embs = [self.process_residual(cx, norm, mdfa, org) for cx, norm, mdfa, org in zip(cxs, norms, mdfas, orgs)]

        return tuple(p_embs)


class Deep_branch(nn.Module):
    def __init__(self, channel_num):
        super(Deep_branch, self).__init__()

        self.norms = LayerNorm3d(channel_num, LayerNorm_type='WithBias')
        self.mssa = MSSA(channel_num)
        self.mdfa = CBAMLayer(channel_num)

    def forward(self, E):
        """Forward pass, including residual connections and feature enhancement"""
        cx = self.mssa(E)
        cx = cx + E  # Residual connection
        x = self.norms(cx)
        x = self.mdfa(x)
        x = x + cx  # Second residual connection
        return x


class Deep_residual_Block(nn.Module):
    def __init__(self, config, channel_num):
        super(Deep_residual_Block, self).__init__()

        self.norms = LayerNorm3d(channel_num, LayerNorm_type='WithBias')
        self.layer = nn.ModuleList([Deep_branch(channel_num) for _ in range(config.transformer["num_layers"])])

    def forward(self, d3):
        for layer_block in self.layer:
            d3 = layer_block(d3)

        d3 = self.norms(d3)

        return d3


class Shallow_residual_Block(nn.Module):
    def __init__(self, config, channel_num):
        super(Shallow_residual_Block, self).__init__()

        self.norms = nn.ModuleList([LayerNorm3d(c, LayerNorm_type='WithBias') for c in channel_num])
        self.layer = nn.ModuleList([Shallow_branch(channel_num) for _ in range(config.transformer["num_layers"])])

    def forward(self, E1, E2, E3):
        for layer_block in self.layer:
            E1, E2, E3 = layer_block(E1, E2, E3)

        E1 = self.norms[0](E1)
        E2 = self.norms[1](E2)
        E3 = self.norms[2](E3)

        return E1, E2, E3


class Shallow_Module(nn.Module):
    def __init__(self, config, channel_num=[64, 128, 256], patchSize=[32, 16, 8]):
        super().__init__()

        self.embeddings = nn.ModuleList(
            [Embeddings(patchSize[i], in_channels=channel_num[i]) for i in range(len(channel_num))])
        self.srb = Shallow_residual_Block(config, channel_num)
        self.FM = nn.ModuleList([
            FeatrueMapping(channel_num[i], channel_num[i], kernel_size=1, scale_factor=(patchSize[i], patchSize[i])) for
            i in range(len(channel_num))
        ])

    def forward(self, x1, x2, x3):
        xs = [x1, x2, x3]
        embs = [emb(x) if x is not None else None for emb, x in zip(self.embeddings, [x1, x2, x3])]
        o1, o2, o3 = self.srb(*embs)
        outputs = []
        for i, (encoded, en, FM) in enumerate(zip([o1, o2, o3], xs, self.FM)):
            if en is not None:
                fm = FM(encoded)
                fm = fm + en
                outputs.append(fm)
            else:
                outputs.append(None)

        return tuple(outputs)


class Deep_Module(nn.Module):
    def __init__(self, config, channel_num=256, patchSize=4):
        super().__init__()

        self.embeddings = Embeddings(patchSize, in_channels=channel_num)
        self.drb = Deep_residual_Block(config, channel_num)
        self.FM = FeatrueMapping(channel_num, channel_num, kernel_size=1, scale_factor=(patchSize, patchSize))

    def forward(self, d3):
        emb = self.embeddings(d3)
        o3 = self.drb(emb)
        output = self.FM(o3)
        return output


class Res_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Res_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)
        return out


class CBAMLayer(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(CBAMLayer, self).__init__()

        # Channel attention compresses H,W to 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )

        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x

        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x


class ADSF(nn.Module):
    def __init__(self, channel, m=-0.80, b=1, gamma=2):
        super(ADSF, self).__init__()

        self.w = torch.nn.Parameter(torch.FloatTensor([m]), requires_grad=True)
        self.mix_block = nn.Sigmoid()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        t = int(abs((math.log(channel, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        self.conv1 = nn.Conv1d(1, 1, kernel_size=k, padding=int(k / 2), bias=False)
        self.fc = nn.Conv2d(channel, channel, 1, padding=0, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        ax1 = self.avg_pool(x1)
        ax2 = self.avg_pool(x2)
        ax1 = self.conv1(ax1.squeeze(-1).transpose(-1, -2)).transpose(-1, -2)  # (1, C, 1)
        ax2 = self.fc(ax2).squeeze(-1).transpose(-1, -2)  # (1, C, 1)
        out1 = torch.sum(torch.matmul(ax1, ax2), dim=1).unsqueeze(-1).unsqueeze(-1)  # (1, C, 1, 1)
        out1 = self.sigmoid(out1)
        out2 = torch.sum(torch.matmul(ax2.transpose(-1, -2), ax1.transpose(-1, -2)), dim=1).unsqueeze(-1).unsqueeze(-1)
        out2 = self.sigmoid(out2)
        mix_factor = self.mix_block(self.w)
        out = out1 * mix_factor + out2 * (1 - mix_factor)
        out = self.conv1(out.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        out = self.sigmoid(out)

        return torch.cat([(x2 * out), x1], dim=1)


class MSCA(nn.Module):
    def __init__(self, channel_num, LayerNorm_type='WithBias'):
        super(MSCA, self).__init__()

        self.KV_size = channel_num[0] + channel_num[1] + channel_num[2]

        self.msm1 = MSM(channel_num[0])
        self.msm2 = MSM(channel_num[1])
        self.msm3 = MSM(channel_num[2])
        self.msmk = MSM(self.KV_size)
        self.msmv = MSM(self.KV_size)

        self.project_out1 = nn.Conv2d(channel_num[0], channel_num[0], kernel_size=1)
        self.project_out2 = nn.Conv2d(channel_num[1], channel_num[1], kernel_size=1)
        self.project_out3 = nn.Conv2d(channel_num[2], channel_num[2], kernel_size=1)

        self.psi = nn.InstanceNorm2d(1)
        self.softmax = nn.Softmax(dim=3)

    def forward(self, E1, E2, E3, E4):
        B1, C1, h, w = E1.shape

        q1 = self.msm1(E1)
        q2 = self.msm2(E2)
        q3 = self.msm3(E3)
        k = self.msmk(E4)
        v = self.msmv(E4)

        q1, q2, q3 = _head1_chw(q1), _head1_chw(q2), _head1_chw(q3)
        k = _head1_chw(k)
        v = _head1_chw(v)

        q1, q2, q3 = F.normalize(q1, dim=-1), F.normalize(q2, dim=-1), F.normalize(q3, dim=-1)
        k = F.normalize(k, dim=-1)

        attn1, attn2, attn3 = (q1 @ k.transpose(-2, -1)) / math.sqrt(self.KV_size), (
                    q2 @ k.transpose(-2, -1)) / math.sqrt(self.KV_size), (q3 @ k.transpose(-2, -1)) / math.sqrt(
            self.KV_size)

        attention_probs1, attention_probs2, attention_probs3 = self.softmax(self.psi(attn1)), self.softmax(
            self.psi(attn2)), self.softmax(self.psi(attn3))
        out1, out2, out3 = attention_probs1 @ v, attention_probs2 @ v, attention_probs3 @ v

        out_1, out_2, out_3 = out1.mean(dim=1), out2.mean(dim=1), out3.mean(dim=1)

        b = out_1.shape[0]
        out_1 = out_1.reshape(b, C1, h, w)
        out_2 = out_2.reshape(b, out_2.shape[1], h, w)
        out_3 = out_3.reshape(b, out_3.shape[1], h, w)

        O1, O2, O3 = self.project_out1(out_1), self.project_out2(out_2), self.project_out3(out_3)

        return O1, O2, O3


class MSSA(nn.Module):
    def __init__(self, channel_num, LayerNorm_type='WithBias'):
        super(MSSA, self).__init__()

        self.channel_num = channel_num
        self.msmq = MSM(channel_num)
        self.msmk = MSM(channel_num)
        self.msmv = MSM(channel_num)

        self.project_out1 = nn.Conv2d(channel_num, channel_num, kernel_size=1)

        self.psi = nn.InstanceNorm2d(1)
        self.softmax = nn.Softmax(dim=3)

    def forward(self, E4):
        B1, C1, h, w = E4.shape

        q = self.msmq(E4)
        k = self.msmk(E4)
        v = self.msmv(E4)

        q = _head1_chw(q)
        k = _head1_chw(k)
        v = _head1_chw(v)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn1 = (q @ k.transpose(-2, -1)) / math.sqrt(self.channel_num)

        attention_probs1 = self.softmax(self.psi(attn1))
        out1 = attention_probs1 @ v

        out_1 = out1.mean(dim=1)

        out_1 = out_1.reshape(out_1.shape[0], C1, h, w)

        O1 = self.project_out1(out_1)

        return O1


class MSM(nn.Module):
    def __init__(self, channel, LayerNorm_type='WithBias'):
        super(MSM, self).__init__()

        self.norm = LayerNorm3d(channel, LayerNorm_type)  # Normalization layer

        self.conv_layers = self._create_conv_layers(channel)

    def _create_conv_layers(self, channel):
        conv_layers = nn.ModuleDict({
            f'conv1_1_{i}': nn.Conv2d(channel, channel, (1, size), padding=(0, size // 2), groups=channel)
            for i, size in enumerate([7, 11, 21])
        })
        conv_layers.update({
            f'conv2_1_{i}': nn.Conv2d(channel, channel, (size, 1), padding=(size // 2, 0), groups=channel)
            for i, size in enumerate([7, 11, 21])
        })
        return conv_layers

    def forward(self, x):
        x = self.norm(x)
        out = sum(self.conv_layers[f'conv1_1_{i}'](x) for i in range(3)) + sum(
            self.conv_layers[f'conv2_1_{i}'](x) for i in range(3))
        return out


class DCBL(nn.Module):
    def __init__(self, in_channels, out_channels, activation=None):
        super(DCBL, self).__init__()
        # NOTE: do NOT use a module object as a default value (e.g.
        # ``activation=nn.ReLU()``).  A default module is created once at
        # class-definition time and shared by every DCBL instance in the
        # process, leaking state across models (a forward hook registered on one
        # model would keep firing on the next one).  Build a fresh module.
        if activation is None:
            activation = nn.ReLU()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            activation,
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            activation
        )

    def forward(self, x):
        return self.conv_layers(x)


class SDSNet(nn.Module):
    """SDS-Net.  forward() returns ONLY the final single-path output
    ``out`` as raw logits [B, 1, H, W] (original ``mode='test'``
    behaviour); no internal sigmoid.

    Input spatial size must be divisible by 16 (patch-embedding based
    shallow/deep synergy modules); e.g. 256 or 512."""

    def __init__(self, in_channels=1, n_classes=1):
        super().__init__()

        config = _SDSConfig()

        self.in_channels = in_channels
        self.n_classes = n_classes
        base = config.base_channel
        # Backbone
        self.backbone = Backbone(in_channels, base, num_blocks=3)

        self.shallow = Shallow_Module(config, channel_num=[base, base * 2, base * 4],
                                      patchSize=config.patch_sizes)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.deep = Deep_Module(config, channel_num=base * 4, patchSize=4)
        self.adsf1 = ADSF(base * 4)
        self.adsf2 = ADSF(base * 2)
        self.adsf3 = ADSF(base)

        self.dcbl1 = DCBL(base * 8, base * 2)
        self.dcbl2 = DCBL(base * 4, base)
        self.dcbl3 = DCBL(base * 2, base)

        self.out = nn.Conv2d(base, n_classes, kernel_size=(1, 1), stride=(1, 1))

        # Deep-supervision heads: retained for parameter-count / state_dict
        # fidelity with the PAL model, unused by forward().
        self.gt_conv3 = nn.Sequential(nn.Conv2d(base * 4, 1, 1))
        self.gt_conv2 = nn.Sequential(nn.Conv2d(base * 2, 1, 1))
        self.gt_conv1 = nn.Sequential(nn.Conv2d(base * 1, 1, 1))
        self.outconv = nn.Conv2d(4 * 1, 1, 1)

    def forward(self, x):
        features = self.backbone(x)

        x1, x2, x3, d3 = features[0], features[1], features[2], features[3]

        org1, org2, org3, org4 = x1, x2, x3, d3

        x1, x2, x3 = self.shallow(x1, x2, x3)
        d3 = self.deep(d3)

        x1 = x1 + org1
        x2 = x2 + org2
        x3 = x3 + org3
        d3 = d3 + org4

        d2 = self.dcbl1(self.adsf1(self.up(d3), x3))
        d1 = self.dcbl2(self.adsf2(self.up(d2), x2))
        out = self.out(self.dcbl3(self.adsf3(self.up(d1), x1)))

        # Original mode='test' path: raw logits, no sigmoid, no aux outputs.
        return out
