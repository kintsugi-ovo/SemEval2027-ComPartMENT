import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data" / "train"

SUBSETS = {
    "en-nn": ("en-nn-train.tsv", "Compound", [("ModAvg", "ModStd"), ("HeadAvg", "HeadStd")]),
    "de-nn": ("de-nn-train.tsv", "Compound", [("ModAvg", "ModStd"), ("HeadAvg", "HeadStd")]),
    "en-pv": ("en-pv-train.tsv", "ParticleVerb", [("Avg", "Std")]),
    "de-pv": ("de-pv-train.tsv", "ParticleVerb", [("Avg", "Std")]),
}

F32 = np.float32
EPS = 1e-6


def var_for_order(vals, order):
    s1 = F32(0.0)
    s2 = F32(0.0)
    for i in order:
        x = F32(vals[i])
        s1 = F32(s1 + x)
        s2 = F32(s2 + F32(x * x))
    n = F32(len(vals))
    mean = F32(s1 / n)
    raw = F32(F32(s2 / n) - F32(mean * mean))
    var = F32(raw * F32(F32(n) / F32(max(n - 1, 1))))
    return float(var), float(s2), float(mean)


def main():
    print("=" * 100)
    print("组内 std 的 fp32 抵消误差体检（模拟 CUDA index_add_ 的原子累加顺序）")
    print("=" * 100)
    rows = []
    for name, (fn, key, cols) in SUBSETS.items():
        df = pd.read_csv(TRAIN / fn, sep="\t")
        df.columns = [c.strip() for c in df.columns]
        for col, _ in cols:
            groups = list(df.groupby(key)[col])
            n_neg = n_zero = n_tiny = n_ok = 0
            worst_var = 1e9
            worst_grad = 0.0
            n_group = 0
            for _, g in groups:
                vals = [float(v) for v in g.tolist()]
                n = len(vals)
                n_group += 1
                if n < 2:

                    var, _, _ = var_for_order(vals, range(n))
                    if var < 0:
                        n_neg += 1
                    elif var == 0:
                        n_zero += 1
                    continue

                orders = [list(range(n)), list(range(n))[::-1]]
                if n <= 5:
                    orders += [list(p) for p in itertools.permutations(range(n))]
                for od in orders:
                    var, _, _ = var_for_order(vals, od)
                    if var < 0:
                        n_neg += 1
                    elif var == 0:
                        n_zero += 1
                    elif var < EPS:
                        n_tiny += 1
                        worst_grad = max(worst_grad, 0.5 / np.sqrt(var))
                    else:
                        n_ok += 1
                    worst_var = min(worst_var, abs(var))
            tot = n_neg + n_zero + n_tiny + n_ok
            rows.append(dict(
                子集=name, 列=col, 目标组数=n_group,
                var为负_切断梯度=n_neg, var恰为0=n_zero,
                var极小正数_梯度爆炸=n_tiny, var正常=n_ok,
                最坏sqrt导数=round(worst_grad, 1),
                占比_非正常=f"{(n_neg + n_zero + n_tiny) / max(tot, 1):.1%}",
            ))
    rep = pd.DataFrame(rows)
    print(rep.to_string(index=False))
    print()
    print("=" * 100)
    print("读法")
    print("=" * 100)
    print("· var 为负 ⇒ clamp(min=0) 把梯度切断（安全，但该组在这一项上**没有梯度**）")
    print("· var 恰为 0 ⇒ sqrt 的导数是 0.5/0 = inf；clamp 在边界上的梯度行为")
    print("  （本机实测 PyTorch 2.14：clamp(min=0) 在 x=0 处梯度 = 0，安全）")
    print("· var 极小正数 ⇒ sqrt 导数 = 0.5/√var 巨大 ⇒ 梯度爆炸（不是 NaN，但会把参数推飞）")
    print("· 「var 恰为 0」在 cnt=1 的组上**必然**发生 ⇒ 这类组在 en-nn 里真实存在")
    return rep


if __name__ == "__main__":
    main()
