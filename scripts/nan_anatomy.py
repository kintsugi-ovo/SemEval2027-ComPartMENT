from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]


def hr(t):
    print("\n" + "=" * 92)
    print(t)
    print("=" * 92)


def m2_with_degeneracy(pred, gold, groups, min_targets=2):
    f = pd.DataFrame({"g": groups, "p": pred, "y": gold})
    ps = f.groupby("g")["p"].std(ddof=1)
    gs = f.groupby("g")["y"].std(ddof=1)
    ok = ps.notna() & gs.notna()
    deg = int((ps[ok] == 0).sum())
    n_ok = int(ok.sum())
    if n_ok < min_targets or ps[ok].nunique() <= 1:
        return float("nan"), deg, n_ok
    return float(stats.spearmanr(ps[ok], gs[ok])[0]), deg, n_ok


def main():
    hr("1. 数学根源：Pearson / Spearman 的分母为什么会是 0")
    for x, y, lbl in [([1.0, 1.0, 1.0], [1.0, 2.0, 3.0], "预测侧全常数"),
                      ([1.0, 2.0, 3.0], [2.0, 2.0, 2.0], "真值侧全常数"),
                      ([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], "正常")]:
        X, Y = np.array(x), np.array(y)
        print(f"  {lbl:14s} σx={X.std():.4f}  σy={Y.std():.4f}  "
              f"r={np.corrcoef(X, Y)[0, 1]:.4f}  spearman={stats.spearmanr(X, Y)[0]:.4f}")
    print("\n  r = Σ(x−x̄)(y−ȳ) / √(Σ(x−x̄)² · Σ(y−ȳ)²)")
    print("  ⇒ 分母里只要有一个标准差为 0（输入是常数），整个式子就是 0/0 = 未定义 ⇒ nan")

    hr("2. 边界一：点数 n 到底要多小才会 nan（这里要更正一个常见误解）")
    for n in (1, 2, 3, 4):
        try:
            r = stats.spearmanr(list(range(n)), list(range(n)))[0]
            print(f"  n={n}: spearmanr = {r}")
        except Exception as e:
            print(f"  n={n}: 抛错 {type(e).__name__}")
    print("\n  ★ 结论：**n=2 时相关系数是可定义的**，但结果只能是 +1 或 −1")
    print("    （两种秩排列）⇒ 任何两个非恒定的输入都能拿到 ±1 ⇒ **零判别力**，但**不是 nan**。")
    print("    真正因点数不足而 nan 的只有 **n=1**。")
    print("    ⇒ 项目代码里的 `ok.sum() > 2` 是**实用门槛**（要求 ≥3 个目标，")
    print("      因为 2 个目标只能得到 ±1、无法区分系统好坏），**不是数学必需**。")
    print("    ⇒ 直接后果：Trial 数据每子集只有 2 个目标 ⇒ 指标 2 在 Trial 上**没有判别力**")

    hr("3. 边界二：单例句目标的 std 是 NaN（自由度为 0），不是 0")
    print("  np.std([3.0], ddof=1)                =", np.std([3.0], ddof=1))
    print("  pandas groupby.std(ddof=1) 单元素组   =",
          list(pd.DataFrame({"g": ["a"], "v": [3.0]}).groupby("g")["v"].std(ddof=1).values))
    print("  ⇒ std 只有 1 个样本时**无法定义**（ddof=1 需要 n≥2）")
    print("  ⇒ 代码里必须用 `ps.notna() & gs.notna()` 过滤掉这类组；")
    print("    过滤后若有效目标太少 ⇒ 整个设置 nan")

    hr("4. 三个指标的 nan 风险对比")
    print("  指标 1 ρ = Spearman(逐例句预测, 真值)  → 预测或真值全常数时 nan（**也会 nan**）")
    print("  指标 2 ρ = Spearman(组内std列)          → 预测 std 全为 0 时 nan")
    print("  指标 3 RMSE                             → **永远不会 nan**（除非输入本身含 NaN）")
    print("\n  ★ 对比要点：RMSE 对一个「全错但有方差」的系统照样给分；")
    print("    两个 ρ 对一个「完全没有信息」的系统直接 undefined。")
    print("    ⇒ 这就是官方把 RMSE 称为 'complementing'（补充）的原因。")

    hr("5. 最危险的不是 nan，是「部分塌缩」")
    rng = np.random.default_rng(7)
    G, N = 20, 6
    gold, grp = [], []
    for g in range(G):
        base, spread = rng.uniform(2.5, 4.5), rng.uniform(0.2, 1.2)
        gold += list(base + spread * rng.normal(0, 1, N))
        grp += [g] * N
    gold, grp = np.array(gold), np.array(grp)
    pred = gold + rng.normal(0, 0.5, len(gold))

    print(f"  模拟设置：{G} 个目标 × {N} 条例句")
    print(f"  {'场景':26s} {'指标2':>9s} {'退化目标数':>11s} {'有效目标数':>10s}")
    r, deg, n = m2_with_degeneracy(pred, gold, grp)
    print(f"  {'正常系统':26s} {r:9.4f} {deg:11d} {n:10d}")
    for k in (1, 3, 5, 10, 19, 20):
        p2 = pred.copy()
        for g in range(k):
            m = grp == g
            p2[m] = np.mean(gold[m])
        r, deg, n = m2_with_degeneracy(p2, gold, grp)
        print(f"  {f'前 {k} 个目标塌缩成常数':26s} {r:9.4f} {deg:11d} {n:10d}")
    print("\n  ★ 三个必须记住的点：")
    print("    ① 部分塌缩**不会**触发 nan ⇒ 指标 2 看起来还是一个数，容易被忽略")
    print("    ② 但它**不可靠且非单调**：塌缩 19 个目标给 0.2586，反而比塌缩 10 个的 0.0418 高")
    print("       （塌缩目标的 std=0 排在最低端，真值 std 有高有低 ⇒ 秩被打乱后可能「碰巧」正相关）")
    print("    ③ **有效目标数始终是 20** ⇒ 代码里 `ok.sum()` 的检查拦不住部分塌缩")
    print("    ⇒ 所以必须额外监控「组内 std 退化目标数」这个预警指标（官方没有，我们自己加的）")

    hr("6. nan 在「两层报告」里的传染")
    settings = [0.4659, 0.3799, 0.0383, 0.5272, 0.5861, float("nan")]
    import math
    finite = [v for v in settings if math.isfinite(v)]
    print(f"  假设 6 个设置里 1 个是 nan：{settings}")
    print(f"  非 nan-safe 平均（把 nan 当 0）：({sum(finite):.4f} + 0)/6 = {sum(finite)/6:.4f}")
    print(f"  nan-safe 平均（跳过 nan）      ：{sum(finite):.4f}/{len(finite)} = {sum(finite)/len(finite):.4f}")
    print(f"  若把 nan 直接传进 numpy.mean  ：{np.mean(settings):.4f}   ← 整个合计被毁")
    print("\n  ⇒ 本项目实现用的是 **nan-safe 平均**（`_nanmean`）")
    print("  ⇒ 若 6 个设置**全**是 nan，则合计也 nan ⇒ 等于该系统完全无信息")

    hr("7. 本项目现状（v1 / v2 实测）")
    print("  两个模型、v1 与 v2 各 5 次运行、6 个评测设置：")
    print("     · 指标 2 全部**非 nan** ✅")
    print("     · 「组内 std 退化目标数」全部为 **0** ✅（无任何目标塌缩）")
    print("  ⇒ 当前没有 nan 风险；但这个数字是**监控项**，任何一次改动后都要复看。")


if __name__ == "__main__":
    main()
