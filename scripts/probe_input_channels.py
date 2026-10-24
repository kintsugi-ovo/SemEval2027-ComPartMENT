import pandas as pd, numpy as np, json, re, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from scipy import stats as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "train"
SEED = 42
SPEC = {
    "en-nn": dict(key="Compound", labs=["ModAvg", "HeadAvg"], task="nn"),
    "de-nn": dict(key="Compound", labs=["ModAvg", "HeadAvg"], task="nn"),
    "en-pv": dict(key="ParticleVerb", labs=["Avg"], task="pv"),
    "de-pv": dict(key="ParticleVerb", labs=["Avg"], task="pv"),
}

def target_split(keys, frac=0.2, seed=SEED):
    uniq = sorted(set(keys)); rng = np.random.RandomState(seed); rng.shuffle(uniq)
    n_te = max(1, int(round(len(uniq) * frac))); te = set(uniq[:n_te])
    return np.array([k in te for k in keys])

def ctx_of(ctx, mwe, w=8):
    toks = re.findall(r"[A-Za-zÄÖÜäöüß]+", str(ctx).lower())
    mw = re.findall(r"[A-Za-zÄÖÜäöüß]+", str(mwe).lower())
    for i in range(max(1, len(toks) - len(mw) + 1)):
        if toks[i:i+len(mw)] == mw:
            return " ".join(toks[max(0, i-w): i+len(mw)+w])
    return " ".join(toks[:2*w])

def three_metrics(pred, gold, keys):
    m1 = st.spearmanr(pred, gold).statistic
    d = pd.DataFrame({"k": keys, "p": pred, "g": gold})
    g = d.groupby("k")
    ps = g["p"].std(ddof=1); gs = g["g"].std(ddof=1)
    ok = ps.notna() & gs.notna() & (ps > 0) & (gs > 0)
    m2 = float(st.spearmanr(ps[ok], gs[ok]).statistic) if ok.sum() >= 3 else float("nan")
    rmse = float(np.sqrt(((pred - gold) ** 2).mean()))
    return m1, m2, rmse

def run(df, sp, mode, lab, tr, te, keys):
    k = sp["key"]
    whole = df[k].astype(str).to_numpy()
    ctx = [ctx_of(c, m) for c, m in zip(df["Context"], df[k])]
    if sp["task"] == "nn":
        mod = df["Mod"].astype(str).to_numpy(); head = df["Head"].astype(str).to_numpy()
    else:
        mod = df["Base"].astype(str).to_numpy(); head = df["Particle"].astype(str).to_numpy()
    if mode == "A":
        txt = np.array([f"{w} || {c}" for w, c in zip(whole, ctx)])
    elif mode == "B":
        txt = np.array([f"{w} || MOD={m} || HEAD={h} || {c}"
                        for w, m, h, c in zip(whole, mod, head, ctx)])
    elif mode == "C":
        txt = np.array(ctx)
    elif mode == "D":
        const = mod if lab == "ModAvg" else (head if lab == "HeadAvg" else mod)
        txt = np.array([f"{x} || {c}" for x, c in zip(const, ctx)])
    elif mode == "E":
        txt = whole
    y = df[lab].to_numpy(float)
    pipe = Ridge(alpha=1.0)
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, sublinear_tf=True)
    Xtr = vec.fit_transform(txt[tr]); Xte = vec.transform(txt[te])
    pipe.fit(Xtr, y[tr])
    p = pipe.predict(Xte)
    return three_metrics(p, y[te], keys[te])

res = {}
for name, sp in SPEC.items():
    df = pd.read_csv(DATA / f"{name}-train.tsv", sep="\t")
    df = df.drop_duplicates(subset=[sp["key"], "Context"]).reset_index(drop=True)
    keys = df[sp["key"]].astype(str).to_numpy()
    te = target_split(keys); tr = ~te
    res[name] = {}
    for lab in sp["labs"]:
        r = {}
        for mode in ["A", "B", "C", "D", "E"]:
            m1, m2, rmse = run(df, sp, mode, lab, tr, te, keys)
            r[mode] = dict(m1=round(float(m1), 4), m2=round(float(m2), 4), rmse=round(rmse, 4))
        res[name][lab] = r
        print(f"=== {name} / {lab} ===")
        for mode, label in [("A", "MWE+语境"), ("B", "MWE+切分+语境"), ("C", "仅语境"),
                            ("D", "成分+语境"), ("E", "MWE整串(无语境)")]:
            v = r[mode]
            print(f"  {mode} {label:20s} m1={v['m1']:+.4f}  m2={v['m2']:+.4f}  rmse={v['rmse']:.4f}")
Path(ROOT / "scripts" / "probe_input_channels.json").write_text(
    json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
print("\nwrote scripts/probe_input_channels.json")
