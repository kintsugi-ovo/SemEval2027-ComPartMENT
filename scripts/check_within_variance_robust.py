import pandas as pd, numpy as np, json
from pathlib import Path
from scipy import stats as st

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "train"
SPEC = {
    "en-nn": dict(key="Compound", labs=["ModAvg", "HeadAvg"], stds=["ModStd", "HeadStd"]),
    "de-nn": dict(key="Compound", labs=["ModAvg", "HeadAvg"], stds=["ModStd", "HeadStd"]),
    "en-pv": dict(key="ParticleVerb", labs=["Avg"], stds=["Std"]),
    "de-pv": dict(key="ParticleVerb", labs=["Avg"], stds=["Std"]),
}
out = {}
for name, sp in SPEC.items():
    df = pd.read_csv(DATA / f"{name}-train.tsv", sep="\t")
    r = {}
    for lab, std in zip(sp["labs"], sp["stds"]):
        d = df[[sp["key"], lab, std]].copy()
        g = d.groupby(sp["key"])
        agg = g.agg(n=(lab, "size"), mean=(lab, "mean"),
                    var=(lab, "var"),
                    noise=(std, lambda s: (s ** 2).mean() / 10.0))
        agg = agg[agg["n"] >= 2]

        slope, icpt, r2, p, se = st.linregress(agg["noise"], agg["var"])

        rel = []
        rs = np.random.RandomState(0)
        idx = {k: v for k, v in d.groupby(sp["key"]).groups.items()}
        for _ in range(200):
            a, b = [], []
            for k, rows in idx.items():
                if len(rows) < 2:
                    continue
                p1 = rs.permutation(rows)
                h = len(p1) // 2
                a.append(d.loc[p1[:h], lab].mean())
                b.append(d.loc[p1[h:], lab].mean())
            if len(a) >= 5:
                rel.append(st.pearsonr(a, b).statistic)

        d2 = d.copy()
        d2[std] = d2[std].sample(frac=1, random_state=1).to_numpy()
        g2 = d2.groupby(sp["key"])
        agg2 = g2.agg(n=(lab, "size"), var=(lab, "var"),
                      noise=(std, lambda s: (s ** 2).mean() / 10.0))
        agg2 = agg2[agg2["n"] >= 2]
        slope2, icpt2 = st.linregress(agg2["noise"], agg2["var"])[:2]
        r[lab] = dict(
            口径1_组内观测方差=round(float(agg["var"].mean()), 4),
            口径1_噪声期望=round(float(agg["noise"].mean()), 4),
            口径1_信号估计=round(float(max(0.0, agg["var"].mean() - agg["noise"].mean())), 4),
            口径2_斜率=round(float(slope), 3), 口径2_截距=round(float(icpt), 4),
            口径2_R2=round(float(r2), 3),
            口径3_split_half_r=round(float(np.mean(rel)), 4),
            口径3_split_half_sd=round(float(np.std(rel)), 4),
            口径4_打乱后斜率=round(float(slope2), 3),
            口径4_打乱后截距=round(float(icpt2), 4),
        )
        print(f"{name}/{lab}: " + json.dumps(r[lab], ensure_ascii=False))
    out[name] = r
Path(ROOT / "scripts" / "within_variance_robust.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print("\nwrote scripts/within_variance_robust.json")
