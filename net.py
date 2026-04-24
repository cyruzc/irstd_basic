"""Model registry — unified wrapper for all benchmark models."""

import torch.nn as nn
from loss import BCELogits, SoftIoU, FocalLogits, CombinedLoss
from models import LightUNet, ACM

LOSS_REGISTRY = {
    "BCE+SoftIoU": lambda: CombinedLoss([BCELogits(), SoftIoU()], weights=[1.0, 1.0]),
    "Focal": lambda: FocalLogits(alpha=0.25, gamma=2.0),
}


class Net(nn.Module):
    def __init__(self, model_name, loss_name="BCE+SoftIoU"):
        super().__init__()
        self.model_name = model_name
        self.loss_name = loss_name

        if loss_name not in LOSS_REGISTRY:
            raise ValueError(f"Unknown loss: {loss_name}. Choose from {list(LOSS_REGISTRY)}")
        self.cal_loss = LOSS_REGISTRY[loss_name]()

        if model_name == "LightUNet":
            self.model = LightUNet(in_channels=1, encoder_channels=(8, 16, 32))
        elif model_name == "ACM":
            self.model = ACM(in_channels=1)
        else:
            raise ValueError(f"Unknown model: {model_name}")

    def forward(self, img):
        return self.model(img)

    def loss(self, logits, gt_mask):
        return self.cal_loss(logits, gt_mask)
