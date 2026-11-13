"""Model registry — unified wrapper for all benchmark models."""

import torch.nn as nn

from loss import BCELogits, CombinedLoss, FocalLogits, SoftIoU
from models import (
    ACM,
    DNANet,
    GSFANet,
    LightUNet,
    MiniUNet,
    MiniUNetHead2,
    MPCNet,
    MicroUNet,
    MSHNet,
    SCTransNet,
    SDSNet,
    UIUNet,
    UCTNet,
)

LOSS_REGISTRY = {
    "BCE+SoftIoU": lambda: CombinedLoss([BCELogits(), SoftIoU()], weights=[1.0, 1.0]),
    "Focal": lambda: FocalLogits(alpha=0.25, gamma=2.0),
}

MODEL_REGISTRY = {
    "LightUNet": lambda: LightUNet(in_channels=1),
    "MiniUNet": lambda: MiniUNet(in_channels=1),
    "MiniUNetHead2": lambda: MiniUNetHead2(in_channels=1),
    "MicroUNet": lambda: MicroUNet(in_channels=1),
    "ACM": lambda: ACM(in_channels=1),
    # --- ported baselines (default factory args chosen so forward returns logits) ---
    "DNANet": lambda: DNANet(in_channels=1),                 # mode='test' -> single tensor
    "MSHNet": lambda: MSHNet(in_channels=1),
    "GSFANet": lambda: GSFANet(in_channels=1),               # size=256
    "MPCNet": lambda: MPCNet(Train=False),                   # forward() -> [B,1,H,W]
    "PHFNet": lambda: UCTNet(in_features=1, out_features=1,
                             gt_ds=False, mode="test"),      # gt_ds=False -> raw logits
    "SCTransNet": lambda: SCTransNet(in_channels=1),         # img_size=256
    "SDSNet": lambda: SDSNet(in_channels=1),
    "UIUNet": lambda: UIUNet(in_channels=1),                 # mode='test' -> d0
}

LOSS_NAMES = tuple(LOSS_REGISTRY.keys())
MODEL_NAMES = tuple(MODEL_REGISTRY.keys())


class Net(nn.Module):
    def __init__(self, model_name, loss_name="BCE+SoftIoU"):
        super().__init__()
        self.model_name = model_name
        self.loss_name = loss_name

        if loss_name not in LOSS_REGISTRY:
            raise ValueError(f"Unknown loss: {loss_name}. Choose from {list(LOSS_REGISTRY)}")
        if model_name not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model: {model_name}. Choose from {list(MODEL_REGISTRY)}")
        self.cal_loss = LOSS_REGISTRY[loss_name]()
        self.model = MODEL_REGISTRY[model_name]()

    def forward(self, img):
        return self.model(img)

    def forward_features(self, img):
        if not hasattr(self.model, "forward_features"):
            raise AttributeError(f"Model {self.model_name} does not expose forward_features()")
        return self.model.forward_features(img)

    def loss(self, logits, gt_mask):
        return self.cal_loss(logits, gt_mask)
