# Changelog

按时间倒序。日期为本地时间。

## 2026-11-13

### 新增 OPDC 匹配 + hIoU 评测协议

依据 *Rethinking Evaluation of Infrared Small Target Detection*（NeurIPS 2025 D&B，
arXiv:2509.16888），把论文的两处修正接入本仓库。

**新增**

- `metrics_opdc.py` —— 与训练框架解耦的 CPU 评测模块（numpy / scipy / scikit-image）：
  - `OPDCMatching`：两阶段匹配（IoU≥0.5 重叠优先 + 匈牙利全局分配；剩余对再用质心距<3 补偿）。
    替代经典协议的贪心匹配，解决「大目标偏 3px 被判全漏」与「匹配结果依赖枚举顺序」两个问题。
  - `MatchingMetrics`：一次给出 `Pd / Fa / seg_iou / loc_iou / hIoU`，
    其中 `hIoU = Σ(匹配对像素 IoU) / (TP+FP+FN)`（`TP` 在闭式里约掉）。
  - `HierarchicalErrorAnalysis`：7 类误差分解，满足 `Σ loc_err = 1−loc_iou`、`Σ seg_err = 1−seg_iou`。
  - `DistanceOnlyMatching`：保留经典贪心匹配，供 A/B 对照。
  - `python metrics_opdc.py` 跑内置自测（用例 A–F）。
- `tools/audit/` —— 评测协议审计材料：数据集划分审计、论文 Tab.1 数据、OPDC/hIoU 机制验证脚本。

**修改**

- `test.py`
  - 新增 `--protocol {basic,opdc,full}`（默认 `full`，经典协议与新协议并排输出，便于和论文 Tab.1 对照）；
  - 新增 `--overlap_threshold` / `--distance_threshold` / `--penalize_invalid` / `--error_analysis`；
  - 新增 `--dump`，导出逐图计数（`tp/fp/fn/tp_iou/fp_area/area`）供后续阈值策略分析；
  - 多 seed 汇总自动包含新字段。
- `README.md` —— 补充两套协议的动机、`hIoU` 公式与用法。
- `requirements.txt` —— 显式声明 `scipy`（OPDC 的匈牙利分配）。

**兼容性**

- `metrics.py`、`train.py` 未改动：每 epoch 的 GPU 指标（IoU / nIoU）保持原样，
  新协议只在训练后的 `test.py` 中计算。
- `--protocol basic` 可完全复现改动前的输出。

**验证**

- 内置自测全部通过（6 组用例）。
- 与官方 `PyIRSTDMetrics` 在 500 个随机用例上逐位一致：
  TP/FP/FN、Σ 匹配对 IoU、`fp_area`、`Pd`、`Fa`、`hIoU`、7 类误差分解全部对齐；
  官方 `examples/test_data` 两张真实 mask 上也一致。
- 因此本仓库的 `hIoU` 可与论文 Tab.1 直接对比。

**已知问题**

- 官方 OPDC 实现是「先分配后过滤」，可能把合法对消耗在非法配对上再丢弃，
  随机用例约 0.2% 会少算 TP（方向恒为保守）。`--penalize_invalid` 提供更严格的单次约束分配，
  默认关闭以保持与官方数值一致。
- 阈值仍按经典做法固定（默认 0.5）；`--threshold` 可调，但「在 val 上选阈值」尚未接入。
