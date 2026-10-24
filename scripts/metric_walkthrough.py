from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data" / "train"

N_TARGETS = 4
N_PER_TARGET = 3


def hr(title):
    print("\n" + "=" * 92)
    print(title)
    print("=" * 92)


def spearman_steps(x, y, name=""):
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    rx = stats.rankdata(x)
    ry = stats.rankdata(y)
    d = rx - ry
    print(f"  -- {name} 的 Spearman（n={n}）")
    print("     原始 x (预测) :", list(np.round(x, 4)))
    print("     原始 y (真值) :", list(np.round(y, 4)))
    print("     x 的秩        :", list(rx))
    print("     y 的秩        :", list(ry))
    print("     d = 秩差      :", list(d))
    print("     d²            :", [round(v * v, 4) for v in d])
    print(f"     Σd² = {float((d ** 2).sum()):.4f}")
    simple = 1 - 6 * float((d ** 2).sum()) / (n * (n * n - 1))
    pear = float(np.corrcoef(rx, ry)[0, 1])
    final = float(stats.spearmanr(x, y)[0])
    print(f"     简化公式 1 - 6Σd²/(n(n²-1)) = 1 - 6×{(d ** 2).sum():.4f}/({n}×({n}²-1)) = {simple:.6f}")
    print(f"     秩的 Pearson（并列时用）      = {pear:.6f}")
    print(f"     scipy.stats.spearmanr        = {final:.6f}   ← 最终采用")
    return final


def group_std_steps(values, label, ddof=1):
    v = np.asarray(values, float)
    n = len(v)
    mean = v.mean()
    dev = v - mean
    sq = dev ** 2
    denom = n - 1 if ddof == 1 else n
    var = sq.sum() / denom
    sd = np.sqrt(var)
    print(f"     [{label}] 原始值 {list(np.round(v, 4))}")
    print(f"       均值 = Σx/n = {mean:.4f}")
    print(f"       偏差 = {list(np.round(dev, 4))}")
    print(f"       偏差² = {list(np.round(sq, 4))}")
    print(f"       Σ偏差² = {sq.sum():.4f}")
    print(f"       方差 = Σ偏差²/({n}-1) = {sq.sum():.4f}/{n-1} = {var:.6f}")
    print(f"       标准差 = sqrt({var:.6f}) = {sd:.6f}")
    return sd


def metric2_from_scratch(pred, gold, groups, label, ddof=1):
    print(f"\n  === 系统 {label} 的指标 2 ===")
    f = pd.DataFrame({"g": groups, "p": pred, "y": gold})
    ps, gs = [], []
    for g, sub in f.groupby("g", sort=True):
        print(f"   -- 目标 {g}（{len(sub)} 条例句）")
        sp = group_std_steps(sub["p"].values, f"预测 std @ {g}", ddof)
        sg = group_std_steps(sub["y"].values, f"人工 std @ {g}", ddof)
        ps.append(sp)
        gs.append(sg)
    ps, gs = np.array(ps), np.array(gs)
    print(f"   -- 汇总成两列（每个目标一个数）")
    print(f"      预测 std 列 ps = {list(np.round(ps, 4))}")
    print(f"      人工 std 列 gs = {list(np.round(gs, 4))}")
    if len(np.unique(ps)) <= 1:
        print(f"      ★ ps 全部相同（{ps[0]:.4f}）⇒ 相关系数未定义 ⇒ **指标 2 = nan**")
        return float("nan")
    return spearman_steps(ps, gs, f"跨目标：Spearman(预测std, 人工std)")


def rmse_steps(pred, gold, label):
    p = np.asarray(pred, float)
    g = np.asarray(gold, float)
    e = p - g
    print(f"  -- {label} 的 RMSE（n={len(p)}）")
    print("     逐例句误差 e = 预测 - 真值 :", list(np.round(e, 4)))
    print("     e²                          :", [round(v * v, 4) for v in e])
    print(f"     Σe² = {float((e ** 2).sum()):.4f}")
    print(f"     MSE = Σe²/n = {float((e ** 2).sum()):.4f}/{len(p)} = {float((e ** 2).mean()):.6f}")
    print(f"     RMSE = sqrt(MSE) = {float(np.sqrt((e ** 2).mean())):.6f}")
    return float(np.sqrt((e ** 2).mean()))


def main():
    rng = np.random.default_rng(42)


    df = pd.read_csv(TRAIN / "en-nn-train.tsv", sep="\t")
    targets = list(dict.fromkeys(df["Compound"]))[:N_TARGETS]
    parts = []
    for t in targets:
        parts.append(df[df["Compound"] == t].head(N_PER_TARGET))
    sl = pd.concat(parts).reset_index(drop=True)

    hr("0. 数据切片（真实 en-nn 训练集，前 4 个目标 × 每目标 3 条例句）")
    print(sl[["ContextID", "Compound", "Mod", "Head", "ModAvg", "ModStd", "HeadAvg", "HeadStd"]].to_string(index=False))
    print(f"\n  目标数 = {sl['Compound'].nunique()}，例句数 = {len(sl)}")
    print("  官方定义：指标 2 的分组单位 = 「一个目标（target）」，组内 = 该目标的所有例句")

    gold = sl["ModAvg"].values
    groups = sl["Compound"].values
    offset_map = {targets[0]: +1.5, targets[1]: -1.0, targets[2]: +0.5, targets[3]: 0.0}

    offsets = np.array([offset_map[g] for g in groups])
    pred_shift = gold + offsets
    pred_const = np.full(len(gold), 3.0)
    pred_real = gold + rng.normal(0, 0.6, len(gold))

    hr("1. 三个对照系统")
    print(f"  A 偏移模型   : 预测 = 真值 + 每目标固定偏移 {offset_map}")
    print("                  （同一个目标的所有例句加同一个常数 ⇒ 组内 std 不变，跨目标秩被破坏）")
    print(f"  B 常数模型   : 预测 = 3.0（对所有例句、所有目标都一样）")
    print(f"  C 真实模型   : 预测 = 真值 + N(0, 0.6)（种子 42）")

    hr("2. 指标 1：Compositionality correlation = Spearman(预测, 真值)")
    m1_shift = spearman_steps(pred_shift, gold, "系统 A")
    m1_const = spearman_steps(pred_const, gold, "系统 B")
    m1_real = spearman_steps(pred_real, gold, "系统 C")

    hr("3. 指标 2：Context variance correlation（官方定义的完整两步）")
    m2_shift = metric2_from_scratch(pred_shift, gold, groups, "A 偏移模型")
    m2_const = metric2_from_scratch(pred_const, gold, groups, "B 常数模型")
    m2_real = metric2_from_scratch(pred_real, gold, groups, "C 真实模型")

    hr("4. 指标 3：RMSE")
    r_shift = rmse_steps(pred_shift, gold, "系统 A")
    r_const = rmse_steps(pred_const, gold, "系统 B")
    r_real = rmse_steps(pred_real, gold, "系统 C")

    hr("5. 关键对照：A（偏移）vs B（常数）vs C（真实）")
    print(f"{'系统':12s} {'指标1 ρ':>10s} {'指标2 ρ':>12s} {'指标3 RMSE':>12s}")
    for lab, a, b, c in [("A 偏移模型", m1_shift, m2_shift, r_shift),
                         ("B 常数模型", m1_const, m2_const, r_const),
                         ("C 真实模型", m1_real, m2_real, r_real)]:
        print(f"{lab:12s} {a:10.4f} {str(round(b,4)) if np.isfinite(b) else 'nan':>12s} {c:12.4f}")
    print("\n  读法：")
    print("   · A 的指标 1 被整体偏移毁掉，但**指标 2 与 C 一样好** ⇒ 指标 2 只看秩，免疫整体偏移")
    print("   · B 的指标 1 也直接 nan（相关系数输入是常数）⇒ 三个指标里有两个会 nan")
    print("   · B 的指标 2 = nan 的**唯一**来源：每个目标的预测 std 全为 0 ⇒ 跨目标相关系数无定义")

    hr("6. 跨设置汇总：官方要求「各设置分别报」+「跨设置取 mean」")
    print("  本切片只演示 en-nn·ModAvg 一个设置。真实评测有 6 个设置：")
    for s in ["en-nn·ModAvg", "en-nn·HeadAvg", "de-nn·ModAvg", "de-nn·HeadAvg", "en-pv·Avg", "de-pv·Avg"]:
        print(f"     - {s}")

    demo = [0.459, 0.380, 0.038, 0.527, 0.586, 0.283]
    print(f"\n  用 v1 的 DeBERTa joint 实测值举例：{demo}")
    print(f"     合计(指标1) = mean = {sum(demo):.4f}/6 = {sum(demo)/6:.4f}"
          f"   （报告里的精确值是 0.3790，差异来自上面四舍五入到 3 位）")
    print("  ⚠️ 合计 = 各设置值的**平均**（nan 安全），**不是**把各设置的行拼起来算一个总 ρ")


if __name__ == "__main__":
    main()
