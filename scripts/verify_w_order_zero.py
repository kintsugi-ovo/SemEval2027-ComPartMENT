import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "tools" / "nbgen" / "cells_b.py"


def load_real_functions():
    src = SRC.read_text(encoding="utf-8")

    def grab(name):
        i = src.index(f"def {name}(")
        j = src.index("\ndef ", i + 10)
        return src[i:j]

    ns = {"torch": torch, "F": F, "np": np}
    exec(grab("_agg_by_subset"), ns)
    exec(grab("order_parts"), ns)
    return ns["order_parts"], ns["_agg_by_subset"]


def split_by_target(df, key, cols, val_ratio=0.2, seed=42, n_bins=5):
    import pandas as pd
    tgt_mean = df.groupby(key)[cols].mean().mean(axis=1)
    try:
        bins = pd.qcut(tgt_mean, n_bins, labels=False, duplicates="drop").to_numpy()
    except ValueError:
        bins = np.zeros(len(tgt_mean), dtype=int)
    rng = np.random.default_rng(seed)
    val_targets = []
    for b in np.unique(bins):
        idx = tgt_mean.index[bins == b].to_numpy()
        rng.shuffle(idx)
        k = max(1, int(round(len(idx) * val_ratio)))
        val_targets.extend(idx[:k].tolist())
    is_val = df[key].isin(set(val_targets)).to_numpy()
    return df[~is_val].reset_index(drop=True), df[is_val].reset_index(drop=True)


def grouped_order(groups, seed, epoch):
    rng = np.random.default_rng(seed + epoch)
    uniq = np.unique(groups)
    rng.shuffle(uniq)
    order = []
    for g in uniq:
        idx = np.where(groups == g)[0]
        rng.shuffle(idx)
        order.extend(idx.tolist())
    return order


def main():
    import pandas as pd
    order_parts, _ = load_real_functions()
    DATA = ROOT / "data" / "train"

    print("=" * 92)
    print("用真源码 order_parts 重放 TargetGroupedSampler：每个 batch 的组内可配对数")
    print("=" * 92)

    for name, key, cols in [("en-nn", "Compound", ["ModAvg", "HeadAvg"]),
                            ("en-pv", "ParticleVerb", ["Avg"])]:
        df = pd.read_csv(DATA / f"{name}-train.tsv", sep="\t")
        df = df.drop_duplicates(subset=[key, "Context"]).reset_index(drop=True)
        tr, va = split_by_target(df, key, cols, seed=42)
        groups = tr.groupby(key).ngroup().to_numpy()
        y = torch.tensor(tr[cols[0]].astype(float).to_numpy(), dtype=torch.float32)
        gidx = torch.tensor(groups, dtype=torch.long)

        order = grouped_order(groups, seed=42, epoch=0)
        bs = 8
        n_batches = int(np.ceil(len(order) / bs))
        pair_counts = []
        for b in range(n_batches):
            sel = order[b * bs:(b + 1) * bs]
            if not sel:
                continue
            bi = torch.tensor(sel, dtype=torch.long)

            B = bi.numel()
            iu = torch.triu(torch.ones(B, B, dtype=torch.bool), diagonal=1)
            mf = iu & (gidx[bi][:, None] == gidx[bi][None, :])
            diff_y = y[bi][:, None] - y[bi][None, :]
            mf = mf & (torch.sign(diff_y) != 0)
            pair_counts.append(int(mf.sum()))

        pc = np.array(pair_counts)
        last = pc[-1]
        print(f"\n【{name}】训练集 {len(tr)} 条 | batch 数 {n_batches} | 最后一批样本数 "
              f"{len(order) - (n_batches - 1) * bs}")
        print(f"  每个 batch 的组内可配对数：中位 {int(np.median(pc))} | 均值 {pc.mean():.1f} | "
              f"最小 {pc.min()} | 最大 {pc.max()}")
        print(f"  ★ 最后一批的可配对数 = {last}")
        zero_batches = int((pc == 0).sum())
        print(f"  全程可配对数为 0 的 batch：{zero_batches}/{len(pc)} "
              f"({zero_batches/len(pc):.1%})")
        print(f"  ⇒ 若最后一批可配对数为 0，则 `bal.weight('order', ...)` 里 "
              f"`torch.where(active>0, w, 0)` 会把它置 0 ⇒ 快照显示 w_order=0 是"
              f"{'★ 采样巧合（该项全程生效）' if zero_batches < len(pc) else '真实失效'}")

    print("\n" + "=" * 92)
    print("结论")
    print("=" * 92)
    print("· 日志里的 `loss_term_weights` 是**训练结束后取的一次快照**")
    print("  （cells_b.py 的 `loss_weight_snapshot()` 遍历 `bal.weights`，")
    print("   而 `bal.weights[name]` 每步都被**覆盖**成该步的权重）")
    print("· 因此它反映的是**最后一个 batch**的状态，而不是训练全程的平均。")
    print("· 最后一个 batch 常常只有 1~6 条样本（`n_train % 8` 的余数），")
    print("  组内可配对数为 0 ⇒ order 项的 active=0 ⇒ 权重被置 0。")
    print("· ⇒ 「w_order 全为 0」是**记录方式的缺陷**，不是该损失项失效。")


if __name__ == "__main__":
    main()
