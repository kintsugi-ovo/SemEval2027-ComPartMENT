import pandas as pd, numpy as np, json, re, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from scipy import stats as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]; DATA = ROOT / "data" / "train"
SPEC = {"en-nn": ("Compound", ["ModAvg", "HeadAvg"], ["ModStd", "HeadStd"]),
        "de-nn": ("Compound", ["ModAvg", "HeadAvg"], ["ModStd", "HeadStd"]),
        "en-pv": ("ParticleVerb", ["Avg"], ["Std"]),
        "de-pv": ("ParticleVerb", ["Avg"], ["Std"])}
SEED = 42
res = {}
for name, (key, labs, stds) in SPEC.items():
    df = pd.read_csv(DATA / f"{name}-train.tsv", sep="\t")
    df = df.drop_duplicates(subset=[key, "Context"]).reset_index(drop=True)
    keys = df[key].astype(str).to_numpy()
    uniq = sorted(set(keys)); rng = np.random.RandomState(SEED); rng.shuffle(uniq)
    te_set = set(uniq[:max(1, int(round(len(uniq) * 0.2)))])
    te = np.array([k in te_set for k in keys]); tr = ~te
    for lab, std in zip(labs, stds):
        ctx = df["Context"].astype(str).to_numpy()
        y = df[lab].to_numpy(float)
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, sublinear_tf=True)
        Xtr = vec.fit_transform(ctx[tr]); Xte = vec.transform(ctx[te])
        m = Ridge(alpha=1.0).fit(Xtr, y[tr]); p = m.predict(Xte)
        d = pd.DataFrame({"k": keys[te], "p": p, "g": y[te], "sd": df[std].to_numpy()[te]})
        g = d.groupby("k")
        agg = g.agg(ps=("p", "std"), gs=("g", "std"), ms=("sd", "mean"), n=("g", "size"))
        agg = agg[agg["n"] >= 2]
        agg = agg[(agg["ps"] > 0) & (agg["gs"] > 0)]
        r2 = float(st.spearmanr(agg["ps"], agg["gs"]).statistic)
        r_ambig = float(st.spearmanr(agg["ps"], agg["ms"]).statistic)
        r_base = float(st.spearmanr(agg["gs"], agg["ms"]).statistic)

        agg["gnoise"] = np.sqrt(agg["ms"] ** 2 / 10.0)
        r_noiseonly = float(st.spearmanr(agg["ps"], agg["gnoise"]).statistic)
        res[f"{name}|{lab}"] = dict(
            m2_probe=r2, rho_predstd_vs_ambiguity=round(r_ambig, 4),
            rho_goldstd_vs_ambiguity=round(r_base, 4),
            rho_predstd_vs_noise_expect=round(r_noiseonly, 4))
        print(f"{name}|{lab}: 指标2(探针)={r2:+.3f} | ρ(预测std,歧义度)={r_ambig:+.3f} "
              f"| ρ(gold std,歧义度)={r_base:+.3f} | ρ(预测std,噪声期望)={r_noiseonly:+.3f}")
Path(ROOT / "scripts" / "m2_mechanism.json").write_text(
    json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
print("\nwrote scripts/m2_mechanism.json")
