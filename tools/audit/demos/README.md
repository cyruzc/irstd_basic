# OPDC / hIoU 机制验证实验

所有数字均由官方实现 [PyIRSTDMetrics](https://github.com/lartpang/PyIRSTDMetrics) 跑出来的，不是从论文抄的。

## 一次性准备

```bash
cd demos
git clone --depth 1 https://github.com/lartpang/PyIRSTDMetrics.git
python3 -m venv _venv && ./_venv/bin/pip install numpy scipy scikit-image
```

## 运行

```bash
./_venv/bin/python demo_opdc.py     # 案例 A/B/C：OPDC 两阶段为什么互补、旧贪心为什么顺序相关
./_venv/bin/python demo_hiou.py     # 案例 D/E/F：hIoU 的盲区修补、闭式形式、乘vs加
./_venv/bin/python audit_opdc.py    # 审计：两阶段"先分配后过滤"的次优性
./_venv/bin/python audit_opdc2.py   # 上面那条的批量频率统计
```

脚本里 `sys.path.insert(0, 'PyIRSTDMetrics')` 假设仓库克隆在当前目录（或改路径）。

## 结论摘要

| 案例 | 结论 |
|---|---|
| A | 20×20 目标偏 3px：像素 IoU=0.739 但旧距离法 TP=0，OPDC TP=1 |
| B | 2×2 目标偏 1px：IoU=0.333 纯重叠准则失败，OPDC 距离补偿救回 |
| C | 旧贪心 TP=1/2，OPDC(匈牙利) TP=2/2 —— 旧实现顺序相关 |
| D | GT 4×4 vs 预测 6×6 同心：Pd=1.0、Fa=0，但 hIoU=0.444 |
| E | 验证 hIoU = Σ(匹配对 IoU)/(TP+FP+FN)，6 个场景全部吻合 |
| F | (loc,seg)=(1.0,0.3)：乘法 0.300，加法 0.650（加法允许线性补偿） |
| 审计 | OPDC 匹配少算 TP 的频率 0~0.2%，方向恒为保守；一行可修 |
