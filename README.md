# IRSTD Benchmark

统一的红外小目标分割 benchmark 框架。所有模型在相同的训练参数下公平对比。

## 仓库结构

```
irstd_basic/
├── train.py                  # 统一训练入口（多模型 / 多数据集 / 多 seed）
├── test.py                   # 评测入口（经典协议 + OPDC/hIoU 协议）
├── net.py                    # 模型注册表 + loss 注册表 + Net 包装
├── dataset.py                # 数据加载（img_idx 划分文件驱动）
├── loss.py                   # BCEWithLogits / SoftIoU / Focal
├── metrics.py                # 全部评测指标：经典协议 (IoU/nIoU/PD/FA) + OPDC 匹配/hIoU/误差分解
├── utils.py                  # 归一化参数、seed、优化器
├── cal_params.py             # 参数量 / FLOPs 统计
├── models/                   # 模型实现（unet / acm / …）
├── probe/                    # 难例机制分析脚本
├── results/                  # 归档的实验产物（各 run 的 json / csv / ckpt）
├── tools/audit/              # 评测协议审计：划分审计、论文 Tab.1 数据、OPDC/hIoU 验证
├── CHANGELOG.md
└── README.md
```

快速自检：

```bash
python metrics.py           # 指标自测（经典 + OPDC/hIoU，用例 A–F）
python test.py --help
```

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

### 移植基线（`models/` 内，均遵循 logits 输出约定）

| 模型 | 注册名 | 参数量 | 工厂参数说明 |
|---|---|---:|---|
| DNANet | `DNANet` | 4.70M | `mode='test'` → 单输出 [B,1,H,W] |
| MSHNet | `MSHNet` | 4.07M | — |
| GSFANet | `GSFANet` | 2.97M | `size=256` |
| MPCNet | `MPCNet` | 2.28M | `Train=False`，`forward()` 单输出 |
| PHFNet | `PHFNet` | 19.52M | 类名 `UCTNet`，`gt_ds=False` → 原始 logits |
| SCTransNet | `SCTransNet` | 11.19M | `img_size=256`，`forward()` 取最后一路输出 |
| SDSNet | `SDSNet` | 2.49M | — |
| UIUNet | `UIUNet` | 50.54M | `mode='test'` → `d0` |

参数量与论文 Tab.1 对齐（DNANet 4.7M / MSHNet 4.1M / UIUNet 50.5M / SCTransNet 11.2M）。
FLOPs 请用 `cal_params.py` 在本机重新生成（不同 thop/torch 版本会有偏差，不要跨环境比较）。

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
| PD / FA | 检测概率 / 虚警率（经典协议：质心距 < 3，贪心匹配） | 训练结束后 (CPU, skimage) |
| PD / FA (OPDC) | 同上，但匹配改用 OPDC（IoU≥0.5 优先 + 距离补偿，匈牙利分配） | 训练结束后 (CPU) |
| **hIoU** | 分级 IoU（目标级 × 像素级），详见下节 | 训练结束后 (CPU) |

### 两套评测协议（同一个 `metrics.py`）

`metrics.py` 保留**经典协议**（与既有论文/BasicIRSTD 的数字可比）；
`metrics.py` 中的 OPDC 部分实现 **NeurIPS'25 D&B《Rethinking Evaluation of IRSTD》** 的两处修正
（arXiv:2509.16888，官方实现 [PyIRSTDMetrics](https://github.com/lartpang/PyIRSTDMetrics)）。

**1) OPDC 匹配**（修的是"算不算命中"这个判定原子）

经典协议用 `质心距离 < 3` 贪心匹配，有两个结构性问题：

- 距离是**绝对量**，不随目标尺寸缩放 → 大目标整体偏 3px，像素 IoU 还有 0.74，却被判成"完全漏检"；
- 匹配是**贪心**的（先到先得），结果依赖连通域的枚举顺序 → 同一份 mask 可能得到不同 PD。

OPDC 改成两阶段 + 全局最优分配：

| 阶段 | 规则 | 治什么 |
|---|---|---|
| 1 | 匈牙利分配，保留 IoU ≥ 0.5 的对 | 大目标偏移、粘连、过预测 |
| 2 | 仅对剩余未匹配对，保留质心距 < 3 的对 | 极小目标（IoU 天然偏低，如 2×2 目标偏 1px 时 IoU 只有 0.33） |

两个准则互为盲区补丁，所以是两阶段而不是一刀切。

**2) hIoU**（修的是"命中之后画得好不好"没被任何指标看见）

```
hIoU = IoU_loc × IoU_seg
     = [TP / (TP+FP+FN)] × [Σ(匹配对像素IoU) / TP]
     = Σ(匹配对像素IoU) / (TP+FP+FN)          ← TP 约掉了
```

物理含义：**目标级 IoU 里，"一个匹配对记 1 分"换成"记它的像素 IoU 分"**。一个数同时惩罚漏检（FN 进分母）、虚警（FP 进分母）、画不准（分子变小）。

它补的盲区是：经典 `Fa` 只累计**未匹配**预测组件的面积（`metrics.py` 的 `PD_FA` 同理），所以匹配上的目标内部多出来的像素**一个都不算虚警**。于是"匹配上但 mask 糊成一团"这种错误在 PD/FA 体系里数学上不可见 —— 例如 4×4 的 GT、6×6 同心预测：`PD=1, FA=0`，但 `hIoU=0.444`。

为什么用乘法不用加权和：加法允许"定位满分"线性补偿"分割烂"（`(1.0, 0.3)` 加权和仍有 0.65），乘法不允许（`hIoU > t` ⟺ 两个分量都 `> t`）。

### 用法

```bash
# 默认 full：同时打印经典协议与新协议，方便和 Tab.1 对照
python test.py --model_name LightUNet --dataset_name IRSTD-1K

# 只看新协议 / 只看经典协议
python test.py --model_name LightUNet --protocol opdc
python test.py --model_name LightUNet --protocol basic

# 7 类误差分解（论文第三部分，慢）
python test.py --checkpoint log/IRSTD-1K/LightUNet/seed42/best_niou.pth.tar --error_analysis

# 导出逐图计数，便于后续做阈值策略分析
python test.py --model_name LightUNet --dump out/lightunet_irstd1k.json
```

输出字段：`IoU nIoU PD FA`（经典） / `PD_opdc FA_opdc hIoU segIoU locIoU`（新协议）/
`err_seg_* err_loc_*`（误差分解，加 `--error_analysis` 时）。

指标模块可独立自检（不依赖 torch）：

```bash
python metrics.py          # 跑内置自测用例 A-F（经典 + OPDC/hIoU）
```

自测覆盖：大目标偏移（旧法 TP=0 / OPDC TP=1）、极小目标（纯 IoU 失败 / OPDC 救回）、
贪心顺序依赖（TP=1/2 vs 2/2）、PD/FA 盲区（hIoU=0.444）、hIoU 闭式一致性、误差分解互补性。

### 与官方 toolkit 的一致性

`metrics.py` 的 OPDC 部分在 500 个随机用例与官方 `PyIRSTDMetrics` 逐位对齐
（TP/FP/FN、Σ 匹配对 IoU、fp_area、Pd、Fa、hIoU 以及 7 类误差分解全部一致），
因此本仓库出的 hIoU 可以直接和论文 Tab.1 对比。

> `--penalize_invalid` 提供了一版更严格的分配（把非法对的代价设为极大值再单次匈牙利）。
> 官方实现是"先分配后过滤"，可能把一个合法对消耗在非法配对上再丢弃，随机用例上约 0.2% 会少算 TP；
> 该选项只会增加 TP、不会虚高，默认关闭以保持与官方数值一致。

## 环境依赖

```bash
pip install torch torchvision pillow numpy scikit-image scipy thop
```

> 新增 `scipy`：OPDC 的匈牙利分配用到 `scipy.optimize.linear_sum_assignment`
> （scikit-image 本身也依赖 scipy，通常已经装上）。

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
