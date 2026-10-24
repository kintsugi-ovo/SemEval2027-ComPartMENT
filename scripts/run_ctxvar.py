import os, json
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
        plt.rcParams["font.sans-serif"] = [cand, "DejaVu Sans"]; break
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "reports", "figures")
os.makedirs(FIG, exist_ok=True)

SPECS = [("en-nn", "Compound", ["ModAvg", "HeadAvg"], "英语·名词复合词", "#3C6E9E"),
         ("de-nn", "Compound", ["ModAvg", "HeadAvg"], "德语·名词复合词", "#7FA8C4"),
         ("en-pv", "ParticleVerb", ["Avg"], "英语·小品词动词", "#C2703D"),
         ("de-pv", "ParticleVerb", ["Avg"], "德语·小品词动词", "#E0A87E")]


def main():
    out = {}
    rows = []
    series = {}
    for name, key, cols, lb, c in SPECS:
        df = pd.read_csv(os.path.join(ROOT, "data", "train", f"{name}-train.tsv"), sep="\t")
        rec = {}
        for col in cols:
            g = df.groupby(key)[col]
            n, sd = g.size(), g.std(ddof=1)

            pred = df.groupby(key)[col].transform("mean")
            gold = df[col]
            m1 = stats.spearmanr(pred, gold)[0]
            s_sd = df.groupby(key).apply(lambda _, _c=col: 0.0).astype(float) \
                if False else df.groupby(key)[col].transform("mean")
            sys_sd = pd.Series(dtype=float)
            tmp = pd.DataFrame({"k": df[key], "p": pred})
            sys_sd = tmp.groupby("k")["p"].std(ddof=1)
            ok = sd.notna() & (sd > 0)
            m2 = stats.spearmanr(sys_sd[ok], sd[ok])[0] if sys_sd[ok].nunique() > 1 else np.nan
            rmse = float(np.sqrt(((pred - gold) ** 2).mean()))
            rows.append(dict(subset=name, label=lb, col=col, m1=m1, m2=m2, rmse=rmse))
            rec[col] = dict(targets=int(g.ngroups),
                            targets_lt2=int((n < 2).sum()),
                            targets_eq2=int((n == 2).sum()),
                            targets_ge5=int((n >= 5).sum()),
                            sd_mean=round(float(sd.mean()), 3),
                            sd_median=round(float(sd.median()), 3),
                            sd_max=round(float(sd.max()), 3),
                            sd_cv=round(float(sd.mean() / gold.mean()), 3),
                            type_level_oracle_m1=round(float(m1), 3),
                            type_level_oracle_m2=None if np.isnan(m2) else round(float(m2), 3),
                            type_level_oracle_rmse=round(rmse, 3))
            series[(name, col)] = sd.dropna()
        out[name] = rec

    R = pd.DataFrame(rows)
    json.dump(out, open(os.path.join(ROOT, "scripts", "context_variance.json"), "w"),
              ensure_ascii=False, indent=1)


    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.2))


    ax = axes[0]
    pos, data_, colors, ticks = 0, [], [], []
    for name, key, cols, lb, c in SPECS:
        col = cols[0]
        v = series[(name, col)].values
        data_.append(v); colors.append(c); ticks.append(lb.replace("·", "\n"))
        pos += 1
    bp = ax.boxplot(data_, positions=range(len(data_)), widths=.55, patch_artist=True,
                    showfliers=False, medianprops=dict(color="#2C3E50", lw=1.6))
    for p_, c in zip(bp["boxes"], colors):
        p_.set_facecolor(c); p_.set_alpha(.85); p_.set_edgecolor("white")
    for i, v in enumerate(data_):
        ax.plot(np.random.default_rng(i).normal(i, .05, len(v)), v, "o", ms=1.8,
                color="white", alpha=.3, mec="none")
    ax.set_xticks(range(len(ticks))); ax.set_xticklabels(ticks, fontsize=8.2)
    ax.set_ylabel("该目标跨例句的人工评分标准差")
    ax.set_title("(a) 指标二的被解释变量\n（每个目标跨例句的人工 std）", fontsize=10.8)


    ax = axes[1]
    w, x = .26, np.arange(len(R))
    ax.bar(x - w, R.m1, w, label="指标1 组合性相关 ρ", color="#3C6E9E", edgecolor="white")
    ax.bar(x, [0] * len(R), w, color="#B03A2E", edgecolor="white", hatch="//",
           label="指标2 语境方差相关 ρ（恒为常数→未定义）")


    ax.bar(x + w, R.rmse, w, label="指标3 RMSE", color="#C2703D", edgecolor="white")
    for i, r in R.iterrows():
        ax.text(i - w, r.m1 + .02, f"{r.m1:.2f}", ha="center", fontsize=7.6)
        ax.text(i, .02, "nan", ha="center", fontsize=8.4, color="#B03A2E", fontweight="bold")
        ax.text(i + w, r.rmse + .02, f"{r.rmse:.2f}", ha="center", fontsize=7.6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r.subset}\n{r.col}" for _, r in R.iterrows()], fontsize=7.4)
    ax.set_ylabel("指标值"); ax.set_ylim(0, 1.02)
    ax.legend(fontsize=7.6, loc="upper center", bbox_to_anchor=(.5, -.13), ncol=1,
              handlelength=1.4)
    ax.set_title("(b) type-level「作弊」上界：它能拿满指标1，\n但指标2 直接未定义", fontsize=10.8)


    ax = axes[2]
    labs = [f"{n}\n{c}" for n, k, cs, lb, c in SPECS for c in cs]
    sds = [series[(n, c)] for n, k, cs, lb, c in SPECS for c in cs]
    ns = [len(v) for v in sds]




    n25 = [int((v > .25 * v.max()).sum()) if len(v) else 0 for v in sds]
    frac25 = [c / n if n else 0 for c, n in zip(n25, ns)]
    xx = np.arange(len(sds))



    ax.bar(xx - .19, ns, .36, color="#9EC3E2", edgecolor="white",
           label="可算 std 的目标数")
    ax.bar(xx + .19, n25, .36, color="#C2703D", edgecolor="white",
           label="其中 std 达到本子集最大值 25% 的目标数")
    for i, (n, m) in enumerate(zip(ns, n25)):

        ax.text(i - .19, n + max(ns) * .018, str(n), ha="center", fontsize=8)
        ax.text(i + .19, m + max(ns) * .018, str(m), ha="center", fontsize=8)
    ax.set_xticks(xx); ax.set_xticklabels(labs, fontsize=7.2)
    ax.set_ylabel("目标数"); ax.set_ylim(0, max(ns) * 1.34)

    ax.legend(fontsize=7.8, loc="lower center", bbox_to_anchor=(.5, 1.01), ncol=1,
              handlelength=1.3, borderaxespad=0.)
    ax.set_title("(c) 指标二的样本量与有效信号量", fontsize=10.8, pad=52)

    fig.suptitle("图12  官方三项指标的实测可达成性：指标二（语境方差相关）会彻底淘汰 type-level 系统",
                 y=1.05, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig12_eval_metrics.png")
    plt.close(fig)

    print(R.to_string(index=False))
    print("\n写入 scripts/context_variance.json 与 figures/fig12_eval_metrics.png")


if __name__ == "__main__":
    main()
