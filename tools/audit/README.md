# tools/audit —— 评测协议审计工具箱

本目录是「这个 benchmark 的评测协议本身是否可信」的研究材料，和训练/评测代码解耦，
不参与 `train.py` / `test.py` 的运行。

主线依据：Pang et al. *Rethinking Evaluation of Infrared Small Target Detection*,
**NeurIPS 2025 (Datasets & Benchmarks)**, arXiv:2509.16888，官方实现
[PyIRSTDMetrics](https://github.com/lartpang/PyIRSTDMetrics)。
仓库内的实现见根目录 [`metrics_opdc.py`](../../metrics_opdc.py)。

## 文件

| 文件 | 用途 |
|---|---|
| [`split_audit.py`](split_audit.py) | 数据集划分审计：NUAA-SIRST 两套官方划分的重合度、SIRST-V2 是否按序列分组 |
| [`findings.md`](findings.md) | 已核实事实清单（含下一步） |
| [`tab1_data.py`](tab1_data.py) | 论文 Tab.1 的 14 方法 × 9 设置数据源，`import` 即用 |
| [`tab1_rethinking.csv`](tab1_rethinking.csv) | 长表：model × setting × metric（1008 行） |
| [`tab1_hIoU_wide.csv`](tab1_hIoU_wide.csv) | 宽表：14 × 9 的 hIoU 矩阵，方便画热力图 |
| [`tab1_tables.md`](tab1_tables.md) | 上述数据的 markdown 表格 |
| [`demos/`](demos/) | OPDC / hIoU 的机制验证实验（可直接跑） |

## 已核实的关键数字

- **同一批 427 张 NUAA-SIRST 有两套互不兼容的官方划分**：
  Dai 官方 `256/85/86`（带 val） vs BasicIRSTD `213/—/214`。
  两套**测试集只重合 42 张**；122 张 Dai 划分的训练图被 BasicIRSTD 当作测试图。
- **SIRST-V2** 是少数带官方 val 的单帧数据集：`train/val/test = 512/256/256`（70/15/15）。
  但它的划分**不按序列分组**（`Misc` 在三个 split 里分别是 219/101/107），存在近邻帧泄漏风险。
- **Pd 对跨域退化几乎无感**（Tab.1 统计）：同域均值 0.957 → 跨域 0.882（−0.075），
  而 hIoU 从 0.640 → 0.435（−0.206）。
- **NUDT 同域 0.817 是幻觉**：同一批模型换到 NUDT→IRSTD1k 只剩 0.19–0.41。

## 跑 demos

```bash
cd tools/audit/demos
git clone --depth 1 https://github.com/lartpang/PyIRSTDMetrics.git
python3 -c "import numpy, scipy, skimage" || pip install numpy scipy scikit-image

python demo_opdc.py    # 案例 A/B/C
python demo_hiou.py    # 案例 D/E/F
python audit_opdc.py   # OPDC 分配次优性审计
python audit_opdc2.py  # 上面那条的频率统计
```

脚本里 `sys.path.insert(0, 'PyIRSTDMetrics')`，所以要么把仓库克隆成 `PyIRSTDMetrics/`，
要么改这一行。

> demos 依赖官方 toolkit 仅用于**对照**；仓库内的实现（`metrics_opdc.py`）不依赖它，
> 只依赖 numpy / scipy / scikit-image。
