# IRSTD-1K UNet Family Results

This directory reorganizes the canonical `seed42` UNet-family results under the current naming convention:

- `LightUNet` = encoder channels `[8, 16, 32]`
- `MiniUNet` = encoder channels `[4, 8, 16]`
- `MicroUNet` = encoder channels `[2, 4, 8]`

The copied result artifacts are stored as `source_*` files to preserve provenance from the original run directories.
Each curated result directory now also includes the copied `best_iou` and `best_niou` checkpoints as `.pth.tar` artifacts.
For `MiniUNet`, both head variants are included:

- `MiniUNet/head4` = standard linear head with 4-channel input
- `MiniUNet/head2` = compressed head bottleneck with 2-channel input

## Source Mapping

The curated results use the canonical names `LightUNet`, `MiniUNet`, and `MicroUNet`.
Raw legacy source paths are preserved in `manifest.json` when they differ from the canonical naming.

## Key Results

| Model | Variant | Params | FLOPs (256x256) | best_iou: IoU / nIoU / PD / FA | best_niou: IoU / nIoU / PD / FA |
|---|---|---:|---:|---|---|
| `LightUNet` | `head8` | 121,033 | 592.97M | `0.6117 / 0.6278 / 0.9158 / 3.78e-05` | `0.6111 / 0.6279 / 0.8956 / 2.77e-05` |
| `MiniUNet` | `head4` | 30,469 | 153.35M | `0.6107 / 0.6116 / 0.9259 / 3.10e-05` | `0.6069 / 0.6167 / 0.9327 / 4.10e-05` |
| `MiniUNet` | `head2` | 30,477 | 153.75M | `0.6080 / 0.6098 / 0.8990 / 3.31e-05` | `0.5953 / 0.6172 / 0.9360 / 4.83e-05` |
| `MicroUNet` | `head2` | 7,723 | 40.89M | `0.0034 / 0.0121 / 0.3939 / 2.73e-03` | `0.0034 / 0.0121 / 0.3939 / 2.73e-03` |

## Notes

- This folder is a curated comparison view; it does not modify the original run directories.
- `MiniUNet/head2` here is the normalized name for the run previously stored under `MicroUNetHead2`.
- See `summary.csv` and `manifest.json` for machine-readable summaries.
