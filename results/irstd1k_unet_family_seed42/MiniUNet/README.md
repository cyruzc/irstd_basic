# MiniUNet Variants

This directory groups the two `MiniUNet` head variants under the current naming convention:

- `head4` = standard `MiniUNet [4,8,16]` with 4-channel final head input
- `head2` = `MiniUNet [4,8,16]` with head bottleneck compressed to 2 channels

Both variant folders also include copied `best_iou` and `best_niou` `.pth.tar` checkpoints for direct reuse.

## Source Mapping

| Variant | Source run directory | Source model name in artifact |
|---|---|---|
| `head4` | `log/IRSTD-1K/MiniUNet/seed42` | `MiniUNet` |
| `head2` | `log/IRSTD-1K/MicroUNetHead2/seed42` | `MicroUNetHead2` |

## Quick Comparison

| Variant | Params | FLOPs (256x256) | best_iou: IoU / nIoU | best_niou: IoU / nIoU |
|---|---:|---:|---|---|
| `head4` | 30,469 | 153.35M | `0.6107 / 0.6116` | `0.6069 / 0.6167` |
| `head2` | 30,477 | 153.75M | `0.6080 / 0.6098` | `0.5953 / 0.6172` |
