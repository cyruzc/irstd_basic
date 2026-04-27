# IRSTD Benchmark

统一的红外小目标分割 benchmark 框架。所有模型在相同的训练参数下公平对比。

## 支持模型

命名约定：

- `LightUNet` = `[8,16,32]`
- `MiniUNet` = `[4,8,16]`，包含 `head4` 和 `head2` 两个变体
- `MicroUNet` = `[2,4,8]`

| 模型 | 参数量 | FLOPs (256×256) |
|------|--------|-----------------|
| LightUNet [8,16,32] | 121K | 593M |
| MiniUNet head4 [4,8,16] | 30K | 153M |
| MiniUNet head2 [4,8,16] | 30K | 154M |
| MicroUNet [2,4,8] | 7.7K | 40.9M |
| ACM | 398K | 352M |

所有模型输出 **logits**（未经 sigmoid），sigmoid 仅在 loss 内部和 eval 时应用。

## 统一训练参数

| 参数 | 值 |
|------|------|
| Loss | BCEWithLogits + SoftIoU (可选: FocalLogits) |
| Optimizer | AdamW (lr=3e-4, weight_decay=1e-4) |
| Scheduler | 5-epoch warmup + CosineAnnealing (eta_min=1e-6) |
| Grad Clip | 1.0 |
| Epochs | 300 |
| Batch Size | 16 |
| Patch Size | 256 |
| Threshold | 0.5 |

## 评测指标

| 指标 | 计算方式 | 时机 |
|------|---------|------|
| IoU | 全局累计交集/并集 | 每 epoch (GPU) |
| nIoU | 逐样本 IoU 取均值 | 每 epoch (GPU) |
| PD / FA | 检测概率 / 虚警率 | 训练结束后 (CPU, skimage) |

## 环境依赖

```bash
pip install torch torchvision pillow numpy scikit-image thop
```

## 数据集

数据集目录结构：

```
datasets/<dataset_name>/
├── images/          # 灰度图 (PNG/BMP)
├── masks/           # 二值掩码 (PNG/BMP)
└── img_idx/
    ├── train_<name>.txt
    └── test_<name>.txt
```

内置预设归一化参数的数据集：NUAA-SIRST、NUDT-SIRST、IRSTD-1K、NUDT-SIRST-Sea、SIRST3、IRDST-real。其他数据集会自动从训练集计算 mean/std。

## 快速开始

```bash
# 训练两个模型，单 seed
python train.py --model_names LightUNet ACM --dataset_names IRSTD-1K

# 训练三种 UNet 宽度变体
python train.py --model_names LightUNet MiniUNet MicroUNet --dataset_names IRSTD-1K

# 训练 MiniUNet 的 head2 变体
python train.py --model_names MiniUNetHead2 --dataset_names IRSTD-1K

# 给同一组实验自定义目录名
python train.py --model_names MiniUNetHead2 --dataset_names IRSTD-1K --experiment_name head2_probe

# 多 seed 训练（自动计算 mean±std）
python train.py --model_names LightUNet ACM --seeds 42 123 456

# 使用 Focal loss
python train.py --model_names LightUNet --loss Focal

# 断点续训
python train.py --model_names LightUNet --resume log/IRSTD-1K/LightUNet/seed42

# 测试指定 checkpoint
python test.py --checkpoint log/IRSTD-1K/LightUNet/seed42/best_niou.pth.tar

# 测试所有 seed，输出 mean±std
python test.py --model_name LightUNet --dataset_name IRSTD-1K

# 查看模型参数量和 FLOPs
python cal_params.py --model_name LightUNet
python cal_params.py --model_name ACM
```

## 命令行参数

```
python train.py --help

  --model_names       模型名称 (LightUNet, MiniUNet, MiniUNetHead2, MicroUNet, ACM)
                      其中 MiniUNet=MiniUNet head4, MiniUNetHead2=MiniUNet head2
  --dataset_names     数据集名称
  --dataset_dir       数据集根目录 (default: ./datasets)
  --batch_size        batch size (default: 16)
  --patch_size        训练裁剪尺寸 (default: 256)
  --n_epochs          训练轮数 (default: 300)
  --lr                初始学习率 (default: 3e-4)
  --weight_decay      权重衰减 (default: 1e-4)
  --min_lr            最小学习率 (default: 1e-6)
  --warmup            warmup 轮数 (default: 5)
  --loss              损失函数 BCE+SoftIoU | Focal (default: BCE+SoftIoU)
  --grad_clip         梯度裁剪 (default: 1.0, 0=关闭)
  --seeds             随机种子列表 (default: 42)
  --threshold         二值化阈值 (default: 0.5)
  --save              日志保存目录 (default: ./log)
  --experiment_name   可选实验子目录名，插入到 log/<dataset>/<model>/ 下
  --threads           DataLoader workers (default: 8)
  --save_interval     周期性保存间隔，0=不保存 (default: 0)
  --resume            断点续训目录路径
```

## 输出结构

```
log/<dataset>/<model>/<experiment_name>/seed<seed>/
├── config.json              # 超参数快照
├── metrics.csv              # 每 epoch: loss, IoU, nIoU
├── best_iou.json            # 最佳 IoU epoch + 指标
├── best_iou.pth.tar         # 最佳 IoU 模型权重
├── best_niou.json           # 最佳 nIoU epoch + 指标
├── best_niou.pth.tar        # 最佳 nIoU 模型权重
├── final_eval.json          # 两个 best 模型的全指标评测结果
└── checkpoints/             # 周期性 checkpoint (--save_interval > 0)
    ├── checkpoint_ep50.pth.tar
    └── ...

log/<dataset>/<model>/<experiment_name>/summary.txt    # 多 seed 聚合 (mean±std)
```

如果未设置 `--experiment_name`，重复运行同一 seed 会自动附加时间戳（如 `seed42_1745487123`），不覆盖旧数据。

## 评测策略

**训练中**：每 epoch 在测试集上计算 IoU + nIoU（纯 GPU，无开销），追踪 `best_iou` 和 `best_niou` 两个 checkpoint。

**训练结束**：分别加载 best_iou 和 best_niou checkpoint，跑一次全指标评测（IoU, nIoU, PD, FA），结果写入 `final_eval.json`。

**多 seed 聚合**：所有 seed 训练完成后自动生成 `summary.txt`，分别按 best_iou 和 best_niou 展示每个 seed 的结果和 mean±std。

## 添加新模型

1. 在 `models/` 下创建模型文件，确保输入 `[B,1,H,W]`，输出 `[B,1,H,W]` **logits**（不含 sigmoid）
2. 在 `models/__init__.py` 中导出
3. 在 `net.py` 中注册模型名

当前的 `UNet` 系列已经统一为一个通用骨干 `models/unet.py`，不同宽度通过注册表暴露为不同模型名。
