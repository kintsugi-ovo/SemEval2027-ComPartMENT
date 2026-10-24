import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data" / "train"
OUT = ROOT / "scripts" / "metric_noise.json"

SPEC = {
    "en-nn": ("en-nn-train.tsv", "Compound", ["ModAvg", "HeadAvg"]),
    "de-nn": ("de-nn-train.tsv", "Compound", ["ModAvg", "HeadAvg"]),
    "en-pv": ("en-pv-train.tsv", "ParticleVerb", ["Avg"]),
    "de-pv": ("de-pv-train.tsv", "ParticleVerb", ["Avg"]),
}
VAL_RATIO = 0.2
N_BOOT = 800
SEED = 42



def load_val():
    rng = np.random.default_rng(SEED)
    out = {}
    for name, (fname, key, cols) in SPEC.items():
        d = pd.read_csv(TRAIN / fname, sep="\t")
        targets = list(d[key].unique())
        rng.shuffle(targets)
        val = set(targets[: max(1, int(len(targets) * VAL_RATIO))])
        sub = d[d[key].isin(val)]
        for c in cols:
            out[(name, c)] = (sub[c].to_numpy(float), sub[key].to_numpy())
    return out


def m1(pred, gold):
    return spearmanr(pred, gold).statistic


def m2(pred, gold, grp):
    df = pd.DataFrame({"g": grp, "p": pred, "y": gold}).groupby("g")
    ps, ys = df["p"].std(ddof=1), df["y"].std(ddof=1)
    ok = ps.notna() & ys.notna()
    if ok.sum() < 3:
        return np.nan
    return spearmanr(ps[ok], ys[ok]).statistic


def m3(pred, gold):
    return float(np.sqrt(((pred - gold) ** 2).mean()))



def make_sys(V, target_rho, seed, kind="m1"):
    r = np.random.default_rng(seed)
    out = {}
    for k, (g, grp) in V.items():
        sd = g.std()
        best = None
        for a in np.linspace(0.05, 6.0, 90):
            x = np.clip(g + r.normal(0, a * sd, len(g)), 0, 5)
            if kind == "m2":
                rr = m2(x, g, grp)
            else:
                rr = m1(x, g)
            if np.isfinite(rr) and (best is None or abs(rr - target_rho) < abs(best[1] - target_rho)):
                best = (x, rr)
        out[k] = (best[0], grp)
    return out, float(best[1])


def boot_diff(V, sysA, sysB, kind, n_boot=N_BOOT):
    rng = np.random.default_rng(SEED + 1)
    keys = list(V.keys())
    groups = {k: np.unique(V[k][1]) for k in keys}
    la, lb, diff = [], [], []
    for _ in range(n_boot):
        SA, SB = {}, {}
        for k in keys:
            gs = groups[k]
            pick = rng.choice(gs, size=len(gs), replace=True)
            idx = np.concatenate([np.where(V[k][1] == t)[0] for t in pick])
            gold, grp = V[k][0][idx], V[k][1][idx]
            SA[k] = (sysA[k][0][idx], gold, grp)
            SB[k] = (sysB[k][0][idx], gold, grp)
        if kind == "m2":
            ca = np.nanmean([m2(*SA[k]) for k in keys])
            cb = np.nanmean([m2(*SB[k]) for k in keys])
        elif kind == "m3":
            ca = np.mean([m3(SA[k][0], SA[k][1]) for k in keys])
            cb = np.mean([m3(SB[k][0], SB[k][1]) for k in keys])
        else:
            ca = np.mean([m1(SA[k][0], SA[k][1]) for k in keys])
            cb = np.mean([m1(SB[k][0], SB[k][1]) for k in keys])
        la.append(ca); lb.append(cb); diff.append(cb - ca)
    return dict(sd_a=float(np.std(la)), sd_b=float(np.std(lb)), sd_diff=float(np.std(diff)))


def summarize(d):
    return dict(
        sd_diff=round(d["sd_diff"], 4),
        mde_80=round(2.8 * d["sd_diff"], 4),
        ci95_halfwidth=round(1.96 * d["sd_diff"], 4),
    )


def main():
    V = load_val()
    res = {"说明": "目标级 cluster bootstrap；两系统用同一批重采样下标（配对比较）",
           "验证集规模": {f"{k[0]}|{k[1]}": int(len(V[k][0])) for k in V},
           "n_boot": N_BOOT}

    A, ra = make_sys(V, 0.45, 1, "m1")
    B, rb = make_sys(V, 0.50, 2, "m1")
    res["指标1_合计ρ"] = summarize(boot_diff(V, A, B, "m1"))
    res["指标1_合计ρ"]["合成系统真实ρ"] = [round(ra, 3), round(rb, 3)]

    A, ra = make_sys(V, 0.40, 11, "m2")
    B, rb = make_sys(V, 0.42, 12, "m2")
    res["指标2_合计ρ"] = summarize(boot_diff(V, A, B, "m2"))
    res["指标2_合计ρ"]["合成系统真实ρ"] = [round(ra, 3), round(rb, 3)]

    A = {k: (np.clip(V[k][0] + np.random.default_rng(21).normal(0, 0.9 * V[k][0].std(), len(V[k][0])), 0, 5), V[k][1]) for k in V}
    B = {k: (np.clip(V[k][0] + np.random.default_rng(22).normal(0, 0.95 * V[k][0].std(), len(V[k][0])), 0, 5), V[k][1]) for k in V}
    res["指标3_合计RMSE"] = summarize(boot_diff(V, A, B, "m3"))


    def judge(metric, gap):
        thr = res[metric]["mde_80"]
        return dict(差值=round(gap, 4), 阈值=thr,
                    判定=("显著（超过噪声阈值）" if abs(gap) >= thr else "落在噪声内，不足以断言"))

    res["已发表结论的判读"] = {
        "Qwen joint 指标1 高于 DeBERTa joint": judge("指标1_合计ρ", 0.4950 - 0.3790),
        "DeBERTa joint epoch3 高于 epoch2（仍在上升）": judge("指标1_合计ρ", 0.3790 - 0.2227),
        "Qwen joint 指标2 高于 DeBERTa joint": judge("指标2_合计ρ", 0.3864 - 0.3402),
        "Qwen joint 指标2 在 epoch3 回落（过训练）": judge("指标2_合计ρ", 0.4189 - 0.3864),
        "Qwen joint RMSE 低于 DeBERTa joint": judge("指标3_合计RMSE", 0.8429 - 0.9162),
        "DeBERTa de-nn·ModAvg 指标1 高于随机初始化": judge("指标1_合计ρ", 0.0383 - (-0.0031)),
        "Qwen de-nn·ModAvg 指标1 高于随机初始化": judge("指标1_合计ρ", 0.3259 - (-0.0031)),
        "训练后指标2 高于随机初始化（DeBERTa joint）": judge("指标2_合计ρ", 0.3402 - 0.0636),
        "Qwen 指标1 距 oracle 的差距": judge("指标1_合计ρ", 0.7837 - 0.4950),
        "Qwen en-pv 指标2 高于 en-nn·HeadAvg 指标2": judge("指标2_合计ρ", 0.6734 - 0.1050),
    }

    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    for k, v in res.items():
        if isinstance(v, dict):
            print(f"\n== {k}")
            for kk, vv in v.items():
                print(f"   {kk}: {vv}")
    print(f"\n已保存: {OUT}")


if __name__ == "__main__":
    main()
