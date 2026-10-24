import os, re, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import seaborn as sns

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


TRAIN_COLS = {
    "nn": ["ContextID", "Compound", "Mod", "Head", "ModAvg", "ModStd", "HeadAvg", "HeadStd", "Context"],
    "pv": ["ContextID", "ParticleVerb", "Base", "Particle", "Avg", "Std", "Context"],
}

TEST_COLS = {k: [c for c in v if not re.search(r"(Avg|Std)$", c)] for k, v in TRAIN_COLS.items()}


def load(name):
    return pd.read_csv(os.path.join(ROOT, "data", "train", f"{name}-train.tsv"), sep="\t")


def main():
    out = {"schema": {"train": TRAIN_COLS, "test_inferred": TEST_COLS}, "subsets": {}}

    frames = {}
    for name, key, cols, lb, c in SPECS:
        df = load(name)
        frames[name] = df
        task = "nn" if name.endswith("nn") else "pv"


        colprof = {}
        for col in df.columns:
            s = df[col]
            colprof[col] = dict(
                dtype=str(s.dtype),
                n_unique=int(s.nunique()),
                n_null=int(s.isna().sum()),
                example=str(s.dropna().iloc[0])[:60] if len(s) else "",
                n_words_mean=round(float(s.astype(str).str.split().str.len().mean()), 2),
            )


        sizes = df.groupby(key).size()

        if task == "nn":

            en = (df.Mod + " " + df.Head == df.Compound)

            de = (df.Mod + df.Head.str[0].str.lower() + df.Head.str[1:] == df.Compound)
            rules = {"Mod+' '+Head": float(en.mean()),
                     "Mod+lower(Head[0])+Head[1:]": float(de.mean())}
            exact = max(rules.values())
        else:

            r = {"Base+' '+Particle": float((df.Base + " " + df.Particle == df.ParticleVerb).mean()),
                 "Particle+Base（无空格）": float((df.Particle + df.Base == df.ParticleVerb).mean()),
                 "Base+' '+Particle（无空格）": float((df.Base + df.Particle == df.ParticleVerb).mean())}
            rules = r
            exact = max(r.values())


        contig = []
        for _, g in df.groupby(key):
            ids = g.ContextID.astype(str)
            nums = [int(x) for x in ids if x.isdigit()]
            if len(nums) >= 3:
                contig.append((max(nums) - min(nums) + 1) == len(nums))
        frac_contig = round(float(np.mean(contig)), 3) if contig else None


        target_str = df[key]
        dup_str = float((target_str.map(target_str.value_counts()) > 1).mean())

        rec = dict(
            label=lb, task=task,
            n_rows=len(df), n_targets=int(sizes.size),
            target_key=key,
            rows_per_target=dict(mean=round(float(sizes.mean()), 2),
                                 median=float(sizes.median()),
                                 min=int(sizes.min()), max=int(sizes.max()),
                                 pct_lt_5=round(float((sizes < 5).mean()), 3),
                                 pct_ge_8=round(float((sizes >= 8).mean()), 3)),
            size_hist={str(k): int(v) for k, v in sizes.value_counts().sort_index().items()},
            composition_exact_ratio=round(exact, 3),
            composition_rules={k: round(v, 4) for k, v in rules.items()},
            contextid_contiguous_within_target=frac_contig,
            frac_rows_where_target_string_ambiguous=round(dup_str, 3),
            n_chars=int(df.Context.str.len().sum()),
            n_words=int(df.Context.astype(str).str.split().str.len().sum()),
            columns=colprof,
        )
        out["subsets"][name] = rec
        print(f"[{name}] 行 {rec['n_rows']} / 目标 {rec['n_targets']} | "
              f"每目标 {rec['rows_per_target']['mean']} (min {rec['rows_per_target']['min']}, "
              f"max {rec['rows_per_target']['max']}) | <5 条占 {rec['rows_per_target']['pct_lt_5']:.1%} | "
              f"成分拼接命中率 {exact:.1%} | ContextID 连续 {frac_contig}")

    en_nn = set(frames["en-nn"].Compound); de_nn = set(frames["de-nn"].Compound)
    en_pv = set(frames["en-pv"].ParticleVerb); de_pv = set(frames["de-pv"].ParticleVerb)
    out["cross_subset_overlap"] = {
        "en_nn_x_de_nn_identical": len(en_nn & de_nn),
        "en_nn_x_en_pv": len(en_nn & en_pv),
        "en_nn_x_de_pv": len(en_nn & de_pv),
        "de_nn_x_de_pv": len(de_nn & de_pv),
        "note": "不同语言的 MWE 几乎不重叠，属正常；同一语言 NN/PV 亦不重叠，说明两任务的词库独立。",
    }
    print("\n跨子集目标串重叠：", {k: v for k, v in out["cross_subset_overlap"].items() if k != "note"})



    out["five_plus_five"] = {
        "official_design": "每目标采 5 条当代 + 5 条历史；因历史数据稀疏部分不足",
        "observable_in_train": False,
        "reason": "训练集无年代列；ContextID 经验证不含年代序（Pearson +0.034/-0.015/+0.079/-0.128）",
        "practical_implication": "无法在本地构造「当代/历史」对照，只能用表层启发式近似，且实测信号稀疏（见 fig13）",
    }


    fig, axes = plt.subplots(1, 3, figsize=(15.4, 4.3))
    xs = np.arange(4)


    ax = axes[0]
    means = [out["subsets"][n]["rows_per_target"]["mean"] for n, *_ in SPECS]
    pcts = [out["subsets"][n]["rows_per_target"]["pct_lt_5"] * 100 for n, *_ in SPECS]
    ax.bar(xs, means, .55, color=[c for *_, c in SPECS], edgecolor="white")
    for x, m, p in zip(xs, means, pcts):

        if m > 9.3:
            ax.text(x, m - .18, f"{m:.2f}", ha="center", va="top", fontsize=9,
                    color="white", fontweight="bold")
        else:
            ax.text(x, m + .12, f"{m:.2f}", ha="center", fontsize=9)
        ax.text(x, .35, f"<5条\n{p:.0f}%", ha="center", fontsize=7.6, color="white", fontweight="bold")
    ax.axhline(10, color="#B03A2E", ls="--", lw=1.1)
    ax.text(3.42, 10.12, "官方目标 10 条", fontsize=7.8, color="#B03A2E", ha="right")
    ax.set_xticks(xs)
    ax.set_xticklabels([lb.replace("·", "\n") for _, _, _, lb, _ in SPECS], fontsize=8.2)
    ax.set_ylabel("每目标例句数（均值）")
    ax.set_ylim(0, 11.4)
    ax.set_title("(a) 官方目标每目标 10 条（5 当代 + 5 历史）\n实测均值与「不足 5 条」的目标占比", fontsize=10.5)


    ax = axes[1]
    allbins = list(range(1, 12))
    bottom = np.zeros(len(allbins))
    for name, key, cols, lb, c in SPECS:
        sizes = frames[name].groupby(key).size()
        cnt = np.array([int((sizes == b).sum()) for b in allbins], float)
        ax.bar(allbins, cnt, bottom=bottom, color=c, edgecolor="white", label=lb, width=.72)
        bottom += cnt
    for x, tot in zip(allbins, bottom):
        if tot > 0:
            ax.text(x, tot + 6, f"{int(tot)}", ha="center", fontsize=8, color="#33414E")
    ax.set_xlabel("每个目标的例句条数")
    ax.set_ylabel("目标个数（四个子集合计）")
    ax.set_xticks(allbins)
    ax.set_ylim(0, bottom.max() * 1.15)
    ax.legend(fontsize=7.2)
    ax.set_title("(b) 每目标例句条数的完整分布\n众数在 10 条与 5–8 条，长尾为数据稀疏的目标", fontsize=10.5)


    ax = axes[2]
    groups = [("训练集 NN", TRAIN_COLS["nn"], 0.00, True),
              ("训练集 PV", TRAIN_COLS["pv"], 1.42, True),
              ("测试集 NN", TEST_COLS["nn"], 2.84, False),
              ("测试集 PV", TEST_COLS["pv"], 4.26, False)]
    for title, cols, x0, is_train in groups:
        for j, cname in enumerate(cols):
            dropped = bool(re.search(r"(Avg|Std)$", cname))
            fc = "#C9D6DF" if dropped else "#4A7A9B"
            tc = "#93A3B0" if dropped else "white"
            ax.add_patch(plt.Rectangle((x0, -1.2 - j), 1.24, .80,
                                       facecolor=fc, edgecolor="white"))
            ax.text(x0 + .62, -1.2 - j + .40, cname, ha="center", va="center",
                    fontsize=7.6, color=tc, fontweight="bold")


        ax.text(x0 + .62, .44, title, ha="center", va="bottom", fontsize=8.6)
        ax.text(x0 + .62, .10, f"{len(cols)} 列", ha="center", va="bottom", fontsize=7.4,
                color="#7A8A99")

    ax.text(.62, -10.55, "浅灰 = 测试集不提供", ha="center", fontsize=7.4, color="#7A8A99")
    ax.set_xlim(-.12, 4.26 + 1.24 + .12)
    ax.set_ylim(-10.95, .95)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("(c) 列结构：标签列 `*Avg`/`*Std` 在测试集被移除\n"
                 "推理只需 `ContextID` + `Context` 即可对齐", fontsize=10.5, pad=22)

    fig.suptitle("图14  数据结构解剖：四个子集「一行 = 一个语境实例、一个目标 = 7–10 行」的两层嵌套，"
                 "外加标签列在测试集被移除", y=1.045, fontsize=12.8, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig14_data_structure.png")
    plt.close(fig)

    json.dump(out, open(os.path.join(ROOT, "scripts", "data_structure.json"), "w"),
              ensure_ascii=False, indent=1)
    print("\n写入 scripts/data_structure.json 与 figures/fig14_data_structure.png")


if __name__ == "__main__":
    main()
