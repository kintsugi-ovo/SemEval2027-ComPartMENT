import pandas as pd, numpy as np, json
from pathlib import Path
from scipy import stats as st

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "train"
SPEC = {
    "en-nn": dict(key="Compound", labs=["ModAvg", "HeadAvg"], stds=["ModStd", "HeadStd"], task="nn"),
    "de-nn": dict(key="Compound", labs=["ModAvg", "HeadAvg"], stds=["ModStd", "HeadStd"], task="nn"),
    "en-pv": dict(key="ParticleVerb", labs=["Avg"], stds=["Std"], task="pv"),
    "de-pv": dict(key="ParticleVerb", labs=["Avg"], stds=["Std"], task="pv"),
}
res = {}
for name, sp in SPEC.items():
    df = pd.read_csv(DATA / f"{name}-train.tsv", sep="\t")
    g = df.groupby(sp["key"])
    r = {}
    for lab, std in zip(sp["labs"], sp["stds"]):
        y = df[lab].to_numpy(float)

        gm = g[lab].mean()
        within = y - df[sp["key"]].map(gm).to_numpy(float)
        between = df[sp["key"]].map(gm).to_numpy(float) - y.mean()

        noise_row = (df[std].to_numpy(float) ** 2) / 10.0

        per_t = g.agg(obs_var=(lab, "var"), n=(lab, "size"), noise=(std, lambda s: (s ** 2).mean() / 10.0))
        per_t = per_t[per_t["n"] >= 2]
        obs_mean = per_t["obs_var"].mean()
        noise_mean = per_t["noise"].mean()
        signal_var = max(0.0, obs_mean - noise_mean)
        r[lab] = dict(
            total_var=round(float(y.var()), 4),
            between_var=round(float(between.var()), 4),
            within_var=round(float(within.var()), 4),
            within_obs_var=round(float(obs_mean), 4),
            within_noise_var=round(float(noise_mean), 4),
            within_signal_var=round(float(signal_var), 4),
            signal_share=round(float(signal_var / obs_mean), 4) if obs_mean > 0 else None,

            type_share=round(float(between.var() / y.var()), 4),
        )

    if sp["task"] == "nn":
        tm = g["ModAvg"].mean(); th = g["HeadAvg"].mean()
        r["mod_head_target_level"] = dict(
            pearson=round(float(np.corrcoef(tm, th)[0, 1]), 4),
            spearman=round(float(st.spearmanr(tm, th).statistic), 4),
            mod_gt_head_frac=round(float((tm > th).mean()), 4),
            head_gt_mod_frac=round(float((th > tm).mean()), 4),
            same_sign_frac=round(float(np.mean(np.sign(tm - tm.mean()) == np.sign(th - th.mean()))), 4),
        )

        r["mod_head_rowwise_corr"] = round(float(
            df[["ModAvg", "HeadAvg"]].corr().iloc[0, 1]), 4)
    res[name] = r
    print(f"=== {name} ===")
    print(json.dumps(r, ensure_ascii=False, indent=1))

Path(ROOT / "scripts" / "decomp_value.json").write_text(
    json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
print("\nwrote scripts/decomp_value.json")
