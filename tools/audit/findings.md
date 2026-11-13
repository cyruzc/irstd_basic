# IRSTD 评测协议审计：已核实的事实

> 复现命令：`python split_audit.py`
> 所有数字均来自公开仓库的 split 文件，可逐条复核。

## 0. 为什么值得做

`BasicIRSTD` 是这个领域事实上的评测基座（DNANet / ISNet 作者维护，多数论文用它出表）。
它的划分文件只有 `train_*.txt` 和 `test_*.txt` —— **没有验证集**。
没有 val，阈值选择就没有合法落脚点，只能在 test 上扫。

## 1. 事实一：NUAA-SIRST 有两套互不兼容的"官方"划分

同一批 427 张图：

| 来源 | train | val | test |
|---|---|---|---|
| Dai 官方（`open-sirst-v2/splits/*_v1`） | 256 | 85 | 86 |
| BasicIRSTD（`img_idx`） | 213 | — | 214 |

交叉核对：

- 两套**测试集只重合 42 张**。
- **122 张** Dai 划分里的训练图，被 BasicIRSTD 当作测试图。
- **79 张** BasicIRSTD 的训练图，落在 Dai 划分的 val/test 里。

⇒ 论文里写的 "NUAA-SIRST SOTA"，很可能**不在同一个测试集上**。
   这是可量化的、写在代码里的不可比性，不是猜疑。

## 2. 事实二：SIRST-V2 有正式的 val（少数派）

`open-sirst-v2` 提供：

```
splits/train_full.txt  512
splits/val_full.txt    256
splits/test_full.txt   256     # 70/15/15
```

以及 `train_v1 / val_v1 / test_v1`（= 427 张原始 SIRST，即 NUAA-SIRST）等变体。
**这是目前少数带官方 val 的单帧数据集**，适合做"干净协议"的基准。

## 3. 事实三：SIRST-V2 的划分不是按序列分组的

同一序列前缀同时出现在三个 split：

| 前缀 | train | val | test |
|---|---|---|---|
| Misc | 219 | 101 | 107 |
| iraytek | 113 | 59 | 62 |
| S7 / S2 / S4 / TH / 20210725-S1 … | 均有 | 均有 | 均有 |

图像来自真实视频序列抽帧，随机分图 ⇒ **存在近邻帧泄漏风险**。
待补：图像级近邻分析（对齐后 NCC / SSIM），统计"测试图在训练集里有高度相似帧"的比例。

## 4. 事实四：SIRST-V2 不是统一分辨率

抽查 5 张：`263×168`、`299×235`、`301×213`、`347×261`、`1280×1024`。
分辨率不统一 ⇒ resize 到 256×256 会改变目标像素占比，
和 `Fa` 的 per-pixel 归一化耦合，跨设置的数字不可比。

## 5. 补充：领域里唯一"定义了工作点"的赛道

ICPR 2024 Resource-Limited IRSTD Challenge（数据集 WideIRSTD）：

- Track 1: 6000 训练+验证发布，**500 测试完全不发布**
- Track 2: 9000 训练+验证发布，**2000 测试完全不发布**
- 指标约束：**Fa < 1e-4 才算有效分数**（这就是合法工作点）
- 禁止使用预训练模型
- 公开了排行榜与 toolbox

⇒ 想在"未泄漏的 held-out"上验证结论，这是最干净的出口。
注意 Track 1 是点监督、Track 2 是轻量赛道，不能和全监督大模型直接对比。

## 5b. 数据集地图（据 2026 综述 arXiv:2610.18276 与各仓库核实）

| 数据集 | 规模 | 分辨率 | 官方 val | 备注 |
|---|---|---|---|---|
| SIRST-V2 | 515 目标图 + 510 背景 | 尺寸不一 | ✅ 512/256/256 | 序列抽帧，划分非序列分组 |
| **MIRSTD**（CVPR25 SAIST） | 1427 真实 + 1327 合成 | 统一 1024×1024 | ✅ **5:3:2** | 多模态（GPT-4V 文本 + 人工校验） |
| WideIRSTD / LimitIRSTD | 6000 / 9000 | 256~3200 | ✅✅ + 500/2000 未发布 test | 唯一真 held-out |
| IRDST | **142,727**（40,650 真实/85 场景 + 102,077 仿真/317 场景） | 多 | ❌ train/test | 三种标注粒度（mask/bbox/中心点） |
| IRSTD-1K | 1001 | 512×512 | ❌ 800/201 | ISNet 提出 |
| NUDT-SIRST | 1327 | 256×256 | ❌ 663/664 | 合成数据 |
| NUST-SIRST（=MDvsFA 数据） | ~10,000 训练 / 100 测试 | — | ❌ | 2019，综述批评"仿真痕迹、标注不合逻辑" |
| HIT-UAV | 2029 / 579 / 290（train/test/val） | 高空无人机热红外 | ✅ val.json | **目标检测**（人/车，5 类，mAP），不是 PD/FA 分割 |
| IRSTDID-800 | 800 | — | — | 条带噪声去噪基准 |
| IRSatVideo-LEO | 200 序列 / 91,021 帧 / 218,038 目标 | — | — | 天基视频 |
| NUDT-SIRST-Sea | 天基船只 | — | ❌ | 80% 简单背景 |
| Maritime-SIRST | 1131 图 / 2647 目标 | 256×256 | — | Landsat-8 近红外 |
| ISDD | 1284 图 / 3061 船 | 500×500 | — | Landsat-8 |
| IRSDSS | 1491 | — | — | 海浪杂波少（~10%） |
| BSIRST-v1/v2 | v2: 1000 图 | — | — | 空气环境（高原湖泊鸟/湿地） |
| MIST（2026） | 合成引擎 | — | — | 运动目标，低 SCR |
| IR-SatDense（2026） | 2154 背景 512×512 | 512×512 | — | 按背景复杂度 4 级 / 密度分级 |

> 注：**不存在名为 "HIT-IR" 的数据集**。HIT 相关的公开数据集是 HIT-UAV（目标检测任务）。

## 5c. 一句可以写进引言的话

SIRST 官方 README 原文：

> "The name of this dataset series is 'SIRST' and 'SIRST v2'. **Please don't call it as 'NUAA-SIRST'.**"

连数据集的官方名称都被全行业误用 —— 这是"协议与命名都缺乏维护"的直接证据，
比任何间接推断都有力，且只需一行引用。

## 6. 工具与索引

- 方法复现：<https://github.com/XinyiYing/BasicIRSTD>
- 指标实现（统一、多指标）：<https://github.com/IRSTD/STD-EvalKit>
- 方法/数据集索引（只收有代码的）：<https://github.com/Tianfang-Zhang/awesome-infrared-small-targets>
- 评测反思：<https://arxiv.org/abs/2509.16888>

## 6b. 综述清单（按用途排，不按引用排）

| 综述 | venue | 年份 | 读哪几节 |
|---|---|---|---|
| Zhao, Li, Li, Hu, Ma, Tao. *Single-Frame Infrared Small-Target Detection: A Survey* | **IEEE GRSM**（遥感顶刊杂志） | 2022 | taxonomy 图 + 数据集表（~2h） |
| Kou, Wang, Peng, ... Fu. *Infrared Small Target Segmentation Networks: A Survey* | **Pattern Recognition** | 2023 | 分类术语（领域通用语） |
| *Deep Learning Based Infrared Small Object Segmentation: Challenges and Future Directions* | **Information Fusion** | 2025 | 挑战导向部分；明确说用统一框架而非照抄结构 |
| Bouwmans et al. *Traditional and Recent Approaches… Part 1 / Part 2* | arXiv 2610.18276 / 2610.18291 | 2026 | **Table 1（综述的综述）** + 评测章节 |
| *Small and dim target detection in infrared imagery: A review* | Neurocomputing 630 | 2025 | 反例：自认"统一复现超出范围"，只汇编自报数字 |

## 6c. ⚠️ 定位修正：评测批评已经被 NeurIPS 2025 做过了

`Rethinking Evaluation of Infrared Small Target Detection` 实际发表在中 **NeurIPS 2025 (Datasets & Benchmarks)**，
官方代码 <https://github.com/lartpang/PyIRSTDMetrics>（364★，持续维护）。

他们做的是：混合指标 hIoU + 系统化错误分析 + 跨数据集评估。
他们**没有**做的是（也就是你的缝隙）：

1. 量化"旧协议到底偏了多少"——阈值选在 test 上 vs val 上的差值
2. 数据集划分分叉的审计（NUAA 两套划分测试集只重合 42 张）
3. 近邻帧泄漏的正名/证伪
4. seed 方差（5 seed + bootstrap CI）

⇒ 定位从"批评评测"改为"**测量旧协议的系统性偏差**"，并复用 PyIRSTDMetrics 作为指标实现。
这样是"承接"而不是"重复"，审稿人不会说 you are duplicating NeurIPS'25。

## 7. 下一步（每条都能独立写成一个结果）

1. 把「两套划分测试集只重合 42 张」写成 motivation 段。
2. 对 SIRST-V2 做近邻泄漏检测，给出比例。
3. 用 `val_full` 选阈值、`test_full` 只跑一次，报 A/B/C/D 四种报告方式的差 + 5 seed 方差。
4. 在 WideIRSTD held-out 上验证跨域结论。
