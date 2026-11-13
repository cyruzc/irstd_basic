#!/usr/bin/env python3
"""IRSTD split / leakage audit.

复现两件事实（只依赖标准库，无需 numpy）：

1. NUAA-SIRST 存在两套互不兼容的"官方"划分：
   - Dai 官方 (open-sirst-v2/splits/*_v1) : train 256 / val 85 / test 86
   - BasicIRSTD (img_idx)                 : train 213 / test 214 / 无 val
   两套测试集只重合 42 张。

2. SIRST-V2 的 train/val/test 不是按序列分组的：同一序列前缀
   （Misc / iraytek / S7 ...）在三个 split 中同时出现，存在近邻帧泄漏风险。

用法:
    python split_audit.py            # 全部检查
    python split_audit.py --cache d  # 指定缓存目录
"""
from __future__ import annotations

import argparse
import collections
import pathlib
import re
import urllib.request

SIRST_V2 = "https://raw.githubusercontent.com/YimianDai/open-sirst-v2/master/splits/{name}.txt"
BASIC = "https://raw.githubusercontent.com/XinyiYing/BasicIRSTD/main/datasets/{ds}/img_idx/{name}.txt"


def fetch(url: str, cache: pathlib.Path) -> str:
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / url.rsplit("/", 1)[-1]
    if not path.exists():
        with urllib.request.urlopen(url, timeout=60) as r:
            path.write_bytes(r.read())
    return path.read_text()


def load(url: str, cache: pathlib.Path) -> set[str]:
    names = set()
    for line in fetch(url, cache).splitlines():
        line = line.strip()
        if line:
            names.add(re.sub(r"\.(png|jpg|jpeg|bmp)$", "", line, flags=re.I))
    return names


def seq_prefix(name: str) -> str:
    """把 '20210725-S1-1041' -> '20210725-S1'，'Misc_249' -> 'Misc'。"""
    return re.sub(r"[-_]\d+$", "", name)


def audit_nuaa(cache: pathlib.Path) -> None:
    print("=" * 70)
    print("A. NUAA-SIRST（427 张）：两套官方划分的重合度")
    print("=" * 70)
    dai_tr = load(SIRST_V2.format(name="train_v1"), cache)
    dai_va = load(SIRST_V2.format(name="val_v1"), cache)
    dai_te = load(SIRST_V2.format(name="test_v1"), cache)
    b_tr = load(BASIC.format(ds="NUAA-SIRST", name="train_NUAA-SIRST"), cache)
    b_te = load(BASIC.format(ds="NUAA-SIRST", name="test_NUAA-SIRST"), cache)

    print(f"Dai 官方      train/val/test = {len(dai_tr)}/{len(dai_va)}/{len(dai_te)}  (合计 {len(dai_tr | dai_va | dai_te)})")
    print(f"BasicIRSTD    train/val/test = {len(b_tr)}/0/{len(b_te)}  (合计 {len(b_tr | b_te)})")
    print()
    print(f"两套 test 的交集                 : {len(b_te & dai_te)} 张")
    print(f"BasicIRSTD 的 train 落在 Dai val/test : {len(b_tr & (dai_va | dai_te))} 张  <- 一边训练一边测试")
    print(f"Dai 的 train 被 BasicIRSTD 当 test  : {len(dai_tr & b_te)} 张")
    print(f"Dai train 与 Basic train 交集    : {len(dai_tr & b_tr)} 张")
    print()
    print("结论: 论文里写的 'NUAA-SIRST SOTA' 未必在同一个测试集上。")
    print()


def audit_sirstv2(cache: pathlib.Path) -> None:
    print("=" * 70)
    print("B. SIRST-V2：划分是否按序列分组")
    print("=" * 70)
    splits = {
        s: load(SIRST_V2.format(name=s), cache)
        for s in ("train_full", "val_full", "test_full")
    }
    print({k: len(v) for k, v in splits.items()}, "  <- 70/15/15")
    print()
    pre = {k: collections.Counter(seq_prefix(x) for x in v) for k, v in splits.items()}
    shared = set(pre["train_full"]) & set(pre["val_full"]) & set(pre["test_full"])
    print(f"三个 split 共有的序列前缀: {len(shared)} 个 "
          f"(总共 {len(set().union(*map(set, pre.values())))} 个前缀)")
    print()
    print(f"{'序列前缀':<20}{'train':>7}{'val':>7}{'test':>7}")
    for k in sorted(shared, key=lambda p: -pre["train_full"][p])[:12]:
        print(f"{k:<20}{pre['train_full'][k]:>7}{pre['val_full'][k]:>7}{pre['test_full'][k]:>7}")
    print()
    print("结论: 该划分是随机分图，不是按序列分组 -> 存在近邻帧泄漏的可能。")
    print("      需要补做图像级近邻分析（对齐后 NCC/SSIM）才能定性。")


def main() -> None:
    ap = argparse.ArgumentParser()
    # cache 放在仓库的 tmp/ 下（.gitignore 已忽略 tmp/*），避免污染 tools/audit
    default_cache = pathlib.Path(__file__).resolve().parents[2] / "tmp" / "audit-cache"
    ap.add_argument("--cache", default=str(default_cache))
    args = ap.parse_args()
    cache = pathlib.Path(args.cache)
    audit_nuaa(cache)
    audit_sirstv2(cache)


if __name__ == "__main__":
    main()
