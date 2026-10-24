import os, re, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import seaborn as sns
from scipy import stats

sns.set_style("whitegrid")
plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 160, "font.size": 10.5,
                     "axes.grid": True, "grid.alpha": .25, "grid.linewidth": .6,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "axes.titlesize": 12,
                     "legend.frameon": False, "savefig.bbox": "tight"})
_AVAIL = {f.name for f in font_manager.fontManager.ttflist}
for cand in ["Arial Unicode MS", "Hiragino Sans GB", "Songti SC", "STHeiti"]:
    if cand in _AVAIL:
        plt.rcParams["font.sans-serif"] = [cand, "DejaVu Sans"]
        break
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "reports", "figures")
os.makedirs(FIG, exist_ok=True)

SPECS = [("en-nn", "Compound", ["ModAvg", "HeadAvg"], "英语·名词复合词", "#3C6E9E"),
         ("de-nn", "Compound", ["ModAvg", "HeadAvg"], "德语·名词复合词", "#7FA8C4"),
         ("en-pv", "ParticleVerb", ["Avg"], "英语·小品词动词", "#C2703D"),
         ("de-pv", "ParticleVerb", ["Avg"], "德语·小品词动词", "#E0A87E")]

MODERN_EN = ["internet", "computer", "phone", "email", "online", "website",
             "software", "digital", "web ", "app ", "network", "database"]
ARCHAIC_EN = [r"\bcolour\b", r"\bhonour\b", r"\bfavour\b", r"\bbehaviour\b",
              r"\bwhilst\b", r"\btowards\b", r"\bthou\b", r"\bthee\b", r"\bhath\b",
              r"\bought\b", r"\bnaught\b", r"\b-moment\b"]


def era_score(text, lang):
    t = str(text).lower()
    s, fired = 0.0, False
    ys = [int(y) for y in re.findall(r"\b(1[6-9]\d\d|20[0-2]\d)\b", t)]
    if ys:
        s += (1950 - float(np.mean(ys))) / 100.0
        fired = True
    if lang == "de":
        if "ß" in t:
            s += 0.5
            fired = True
        if re.search(r"\b\w*(ae|oe|ue|ss)\w*\b", t) and not re.search(r"[äöüß]", t):
            s += 0.4
            fired = True
    else:
        nmod = sum(1 for w in MODERN_EN if w in t)
        nar = sum(1 for p in ARCHAIC_EN if re.search(p, t))
        if nmod or nar:
            s += 0.5 * nmod - 0.4 * nar
            fired = True
        if re.search(r"\b(can't|don't|didn't|won't|it's)\b", t):
            s -= 0.2
            fired = True
    return s, fired


def load(name, key, lang):
    df = pd.read_csv(os.path.join(ROOT, "data", "train", f"{name}-train.tsv"), sep="\t")
    es = [era_score(x, lang)[0] for x in df.Context]
    fd = [era_score(x, lang)[1] for x in df.Context]
    df["era_score"] = es
    df["era_fired"] = fd
    return df


def safe_spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 5 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(stats.spearmanr(a, b)[0])


def main():
    out = {}
    data = {}
    for name, key, cols, lb, c in SPECS:
        lang = "de" if name.startswith("de") else "en"
        df = load(name, key, lang)
        data[name] = (df, lang, cols, lb, c)
        rec = {}
        for col in cols:

            fired = df.era_fired.mean()

            sub = df[df.era_fired]
            r_all = safe_spearman(sub.era_score, sub[col])

            rw, npos = [], 0
            for _, g in df.groupby(key):
                if len(g) >= 5:
                    r = safe_spearman(g.era_score, g[col])
                    if not np.isnan(r):
                        rw.append(r)
            rw = np.array(rw)
            npos = int((rw > 0).sum()) if rw.size else 0

            sd_h, sd_e = [], []
            for _, g in df.groupby(key):
                if len(g) >= 3 and g.era_fired.any():
                    sd_h.append(g[col].std(ddof=1))
                    sd_e.append(g.era_score.std(ddof=1))
            r_var = safe_spearman(sd_e, sd_h)
            rec[col] = dict(
                n_sent=len(df),
                coverage=round(float(fired), 3),
                n_with_signal=len(sub),
                rho_global_on_signalled=safe_round(r_all),
                n_targets_within=int(rw.size),
                rho_within=safe_round(float(rw.mean())) if rw.size else None,
                frac_within_pos=round(npos / rw.size, 3) if rw.size else None,
                rho_target_std=safe_round(r_var),
            )
            print(f"{name:<6} {col:<8} 覆盖率 {fired:5.1%} ({len(sub)}/{len(df)}) | "
                  f"信号子集全局 ρ {fmt(r_all)} | 目标内 ρ 均值 {fmt(rw.mean()) if rw.size else 'n/a'}"
                  f"(为正 {npos}/{rw.size}) | 目标级 ρ(era.std,人工std) {fmt(r_var)}")
        out[name] = rec
        print()


    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.4))


    ax = axes[0]
    xs = np.arange(4)
    covs, means = [], []
    for name, key, cols, lb, c in SPECS:
        df, lang, cols_, lb_, c_ = data[name]
        covs.append(df.era_fired.mean() * 100)
        means.append(df[cols[0]].mean())
    b1 = ax.bar(xs - .2, covs, .38, color="#4A7A9B", edgecolor="white", label="年代信号覆盖率 (%)")
    ax.set_ylabel("例句中表层年代信号触发的比例 (%)", color="#4A7A9B")
    ax.tick_params(axis="y", labelcolor="#4A7A9B")
    ax.set_ylim(0, max(covs) * 1.35 + 5)
    for x, v in zip(xs - .2, covs):
        ax.text(x, v + .8, f"{v:.1f}%", ha="center", fontsize=8.6, color="#4A7A9B")
    ax2 = ax.twinx()
    ax2.bar(xs + .2, means, .38, color="#C2703D", edgecolor="white", label="平均评分")
    ax2.set_ylabel("平均组合性评分", color="#C2703D")
    ax2.tick_params(axis="y", labelcolor="#C2703D")
    ax2.set_ylim(0, 5)
    ax2.grid(False)
    for x, v in zip(xs + .2, means):
        ax2.text(x, v + .08, f"{v:.2f}", ha="center", fontsize=8.6, color="#C2703D")
    ax.set_xticks(xs)
    ax.set_xticklabels([lb.replace("·", "\n") for _, _, _, lb, _ in SPECS], fontsize=8.2)
    ax.set_title("(a) 先看前提：表层年代信号有多稀疏？", fontsize=10.8)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7.8, loc="upper right")


    ax = axes[1]
    for name, key, cols, lb, c in SPECS:
        df, lang, cols_, lb_, c_ = data[name]
        sub = df[df.era_fired]
        ax.scatter(sub.era_score, sub[col := cols_[0]], s=9, alpha=.30, color=c,
                   edgecolor="none", label=f"{lb}  ρ={fmt(safe_spearman(sub.era_score, sub[cols_[0]]))}")
    ax.axhline(0, color="none")
    ax.set_xlabel("历史感打分 era_score（越大越早）")
    ax.set_ylabel("组合性评分")
    ax.legend(fontsize=7.2, markerscale=1.8, loc="best")
    ax.set_title("(b) 仅在「有年代信号」的例句上：年代感 ↑ 是否伴随评分变化？", fontsize=10.6)


    ax = axes[2]
    for i, (name, key, cols, lb, c) in enumerate(SPECS):
        df, lang, cols_, lb_, c_ = data[name]
        sd_h, sd_e = [], []
        for _, g in df.groupby(key):
            if len(g) >= 3 and g.era_fired.any():
                sd_h.append(g[cols_[0]].std(ddof=1))
                sd_e.append(g.era_score.std(ddof=1))
        r = safe_spearman(sd_e, sd_h)
        if np.isnan(r):
            continue
        ax.bar(i, r, .55, color=c, edgecolor="white")
        ax.text(i, r + (.012 if r >= 0 else -.032), f"{r:+.2f}", ha="center", fontsize=9)
    ax.axhline(0, color="#B03A2E", ls="--", lw=1.2)
    ax.set_xticks(xs); ax.set_xticklabels([lb.replace("·", "\n") for _, _, _, lb, _ in SPECS], fontsize=8.2)
    ax.set_ylabel("Spearman ρ")
    ax.set_ylim(-.22, .32)
    ax.set_title("(c) 指标二能否只靠「年代离散度」拿到？\n目标级 ρ：系统 std ≈ era.std vs 人工 std", fontsize=10.6)

    fig.suptitle("图13  年代效应诊断：表层年代信号在训练语料中过于稀疏，"
                 "不足以解释目标内评分方差，指标二无法靠「识别年代」刷分", y=1.04,
                 fontsize=12.6, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig13_era_effect.png")
    plt.close(fig)

    json.dump(out, open(os.path.join(ROOT, "scripts", "era_effect.json"), "w"),
              ensure_ascii=False, indent=1)
    print("\n写入 scripts/era_effect.json 与 figures/fig13_era_effect.png")


def safe_round(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return round(float(v), 3)


def fmt(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "  n/a "
    return f"{v:+.3f}"


if __name__ == "__main__":
    main()
