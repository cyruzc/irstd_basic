"""Calculate model parameters and FLOPs."""

import argparse
import torch

from net import MODEL_NAMES, Net

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", default="UNet", type=str, choices=MODEL_NAMES)
parser.add_argument("--input_size", default=256, type=int)
opt = parser.parse_args()


def main():
    model = Net(model_name=opt.model_name).cuda()
    params = sum(p.numel() for p in model.parameters())
    print(f"Model: {opt.model_name}")
    print(f"Parameters: {params:,}")

    try:
        from thop import profile
        inp = torch.randn(1, 1, opt.input_size, opt.input_size).cuda()
        flops, _ = profile(model, inputs=(inp,))
        print(f"FLOPs: {flops / 1e6:.2f}M")
    except ImportError:
        print("Install thop for FLOPs: pip install thop")


if __name__ == "__main__":
    main()
