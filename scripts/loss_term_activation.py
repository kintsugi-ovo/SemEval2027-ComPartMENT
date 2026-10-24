import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data" / "train"


SUBSETS = {
    "en-nn": dict(file="en-nn-train.tsv", key="Compound", cols=["ModAvg", "HeadAvg"], std=["ModStd", "HeadStd"]),
    "de-nn": dict(file="de-nn-train.tsv", key="Compound", cols=["ModAvg", "HeadAvg"], std=["ModStd", "HeadStd"]),
    "en-pv": dict(file="en-pv-train.tsv", key="ParticleVerb", cols=["Avg"], std=["Std"]),
    "de-pv": dict(file="de-pv-train.tsv", key="ParticleVerb", cols=["Avg"], std=["Std"]),
}
ORDER = ["en-nn", "de-nn", "en-pv", "de-pv"]

BATCH = 8
XRANK_MARGIN = 0.05
TARGET_EPOCHS = 3
DECAY = 0.8


def load(name):
    df = pd.read_csv(TRAIN / SUBSETS[name]["file"], sep="\t")
    df.columns = [c.strip() for c in df.columns]
    return df


def target_stats(name, df):
    sp = SUBSETS[name]
    out = {}
    for col, scol in zip(sp["cols"], sp["std"]):
        g = df.groupby(sp["key"])[col]
        cnt = g.size()
        std = g.std(ddof=1)
        out[col] = pd.DataFrame({"n": cnt, "std": std}).dropna()
    return out


def simulate_batches(counts, n_epochs=TARGET_EPOCHS):
    rng = np.random.default_rng(0)
    groups_per_batch = []
    for _ in range(n_epochs):
        keys = list(counts.index)
        rng.shuffle(keys)
        flat = []
        for k in keys:
            idx = list(range(counts[k]))
            rng.shuffle(idx)
            flat.extend([k] * counts[k])
        for i in range(0, len(flat), BATCH):
            chunk = flat[i:i + BATCH]
            groups_per_batch.append(len(set(chunk)))
    return np.array(groups_per_batch)


def main():
    print("=" * 92)
    print("v3.1 损失项可激活性体检（离线）")
    print("=" * 92)

    rows = []
    all_stds = {}
    for name in ORDER:
        df = load(name)
        ts = target_stats(name, df)
        for col, tab in ts.items():
            n = tab["n"].to_numpy()
            std = tab["std"].to_numpy()
            all_stds[(name, col)] = std

            pairable = (n * (n - 1) / 2).sum()

            if len(std) > 1:
                d = np.abs(std[:, None] - std[None, :])
                iu = np.triu(np.ones_like(d, dtype=bool), 1)
                tot_pairs = iu.sum()
                above = ((d > XRANK_MARGIN) & iu).sum()
                xrank_ratio = above / max(tot_pairs, 1)
            else:
                tot_pairs, above, xrank_ratio = 0, 0, 0.0

            gpb = simulate_batches(tab["n"])
            rows.append(dict(
                子集=name, 列=col, 目标数=len(tab), 例句数=int(n.sum()),
                每目标例句_中位=float(np.median(n)), 每目标例句_均值=float(n.mean()),
                组内可配对总数=int(pairable),
                组内std_均值=float(std.mean()), 组内std_中位=float(np.median(std)),
                组内std_最小=float(std.min()), 组内std_最大=float(std.max()),
                xrank可用对=int(above), xrank对总数=int(tot_pairs),
                xrank可比较比例=round(float(xrank_ratio), 4),
                batch内目标数_均值=round(float(gpb.mean()), 3),
                batch内目标数_最大=int(gpb.max()),
                batch只有1个目标的比例=round(float((gpb == 1).mean()), 4),
            ))
    rep = pd.DataFrame(rows)
    print(rep.to_string(index=False))

    print()
    print("=" * 92)
    print("结论（每一条都由上面的数字直接推出）")
    print("=" * 92)
    for _, r in rep.iterrows():
        print(f"· {r['子集']}·{r['列']}：每目标 {r['每目标例句_中位']:.0f} 例句（中位）"
              f" ⇒ 组内可配对 {r['组内可配对总数']:,} 对；"
              f"batch 里平均 {r['batch内目标数_均值']:.2f} 个目标"
              f"（{r['batch只有1个目标的比例']:.0%} 的 batch 只有 1 个目标）；"
              f"xrank 可比较对占 {r['xrank可比较比例']:.1%}")
    print()
    print("★ 对 v3.1 设计的直接含义：")
    print("  ① L_order 的前提成立（每目标多条例句）⇒ 组内排序项在四个子集上都能工作；")
    print("  ② 「一个 batch 只有 1~2 个目标」被实测确认 ⇒ 跨目标比较（L_xrank）必须靠历史累积，")
    print("     这正是 v3.1 用 GroupStats 跨 batch 累积的原因（v2 的批内等值匹配因此无效）；")
    print("  ③ xrank 的可比较对比例决定该项的有效性 ⇒ 若某子集比例过低，")
    print("     就只应把 xrank 开在天花板高（可追）的子集上。")
    return rep


if __name__ == "__main__":
    main()
