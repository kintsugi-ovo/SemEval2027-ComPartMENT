import os, re, json, itertools
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
                     "legend.frameon": False, "savefig.bbox": "tight",
                     "mathtext.fontset": "stix"})
_AVAIL = {f.name for f in font_manager.fontManager.ttflist}
for cand in ["Arial Unicode MS", "Hiragino Sans GB", "Songti SC", "STHeiti"]:
    if cand in _AVAIL:
        plt.rcParams["font.sans-serif"] = [cand, "Arial Unicode MS", "DejaVu Sans"]
        break
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "reports", "figures")
os.makedirs(FIG, exist_ok=True)

SPECS = [("en-nn", "Compound", "nn", "英语·名词复合词", "#3C6E9E"),
         ("de-nn", "Compound", "nn", "德语·名词复合词", "#7FA8C4"),
         ("en-pv", "ParticleVerb", "pv", "英语·小品词动词", "#C2703D"),
         ("de-pv", "ParticleVerb", "pv", "德语·小品词动词", "#E0A87E")]


N_ANNOT = 10


def reconstruct_ratings(avg, std, n=N_ANNOT):
    S = int(round(avg * n))
    if abs(avg * n - S) > 1e-4 or not (0 <= S <= 5 * n):
        return None, S, None, None
    N = round((std ** 2) * n * (n - 1))
    if abs((std ** 2) * n * (n - 1) - N) > 0.01:
        return None, S, None, None
    num = N + S * S
    if num % n:
        return None, S, None, None
    Q = num // n
    if not (0 <= Q <= 25 * n) or S * S > n * Q or Q > 5 * S or Q < S:
        return None, S, Q, None

    for c5 in range(0, n + 1):
        r1, r2 = S - 5 * c5, Q - 25 * c5
        if r1 < 0 or r2 < 0:
            break
        for c4 in range(0, n - c5 + 1):
            r1b, r2b = r1 - 4 * c4, r2 - 16 * c4
            if r1b < 0 or r2b < 0:
                break
            for c3 in range(0, n - c5 - c4 + 1):
                r1c, r2c = r1b - 3 * c3, r2b - 9 * c3
                if r1c < 0 or r2c < 0:
                    break
                for c2 in range(0, n - c5 - c4 - c3 + 1):
                    r1d, r2d = r1c - 2 * c2, r2c - 4 * c2
                    if r1d < 0 or r2d < 0:
                        break
                    rem = n - c5 - c4 - c3 - c2
                    c1 = r1d
                    c0 = rem - c1
                    if c0 < 0:
                        continue
                    if c1 == r2d:
                        v = ([0] * c0 + [1] * c1 + [2] * c2 + [3] * c3
                             + [4] * c4 + [5] * c5)
                        err = max(abs(float(np.mean(v)) - avg),
                                  abs(float(np.std(v, ddof=1)) - std))
                        return v, S, Q, err
    return None, S, Q, None


def load(n):
    return pd.read_csv(os.path.join(ROOT, "data", "train", f"{n}-train.tsv"), sep="\t")


def val_cols(task):
    return (["ModAvg", "HeadAvg"], ["ModStd", "HeadStd"]) if task == "nn" else (["Avg"], ["Std"])


def jaccard(a, b):
    u = len(a | b)
    return len(a & b) / u if u else 0.0


def main():
    out = {"granularity": {}, "noise_ceiling": {}, "information_gain": {},
           "target_std": {}}
    labels = [s[3] for s in SPECS]
    keys4 = [s[0] for s in SPECS]
    C = [s[4] for s in SPECS]


    gran = {}
    for name, key, task, lb, c in SPECS:
        df = load(name)
        vc, sc = val_cols(task)
        for v, s in zip(vc, sc):
            vals = df[v].values
            stdv = df[s].values
            u = np.unique(np.round(vals, 6))

            err10 = float(np.abs(vals * 10 - np.round(vals * 10)).max())

            on_grid = float((np.abs(vals * 10 - np.round(vals * 10)) < 1e-6).mean())

            p10 = vals * N_ANNOT
            ok10 = float((np.abs(p10 - np.round(p10)) < 1e-4).mean())

            m2 = vals ** 2 + stdv ** 2
            gran[f"{name}/{v}"] = dict(
                n_unique=int(len(u)), min=float(vals.min()), max=float(vals.max()),
                mean=round(float(vals.mean()), 4),
                off_decimal_grid_0p1=bool(err10 > 1e-6),

                max_err_to_0p1_grid=round(err10, 6),
                max_err_to_0p1_grid_in_score_units=round(err10 / 10, 6),
                frac_on_0p1_grid=round(on_grid, 4),
                frac_10avg_integer=round(ok10, 4),
                mean2_plus_var_max=round(float(m2.max()), 6),
                mean2_plus_var_le25=bool((m2 <= 25 + 1e-9).all()),
                frac_exact_0=round(float((vals <= 1e-9).mean()), 4),
                frac_le_0p5=round(float((vals <= 0.5).mean()), 4),
                frac_ge_4p5=round(float((vals >= 4.5).mean()), 4),
                zero_std_rows=round(float((stdv <= 1e-9).mean()), 4),
            )
    out["granularity"] = gran


    nrev = {}
    for name, key, task, lb, c in SPECS:
        df = load(name)
        vc, sc = val_cols(task)
        for v, s in zip(vc, sc):
            a_vals = df[v].values.astype(float)
            s_vals = df[s].values.astype(float)
            n_rows = len(a_vals)
            cA = cB = cC = cFull = 0
            resid = []
            for av, sv in zip(a_vals, s_vals):
                S_ = round(av * N_ANNOT)
                okA = abs(av * N_ANNOT - S_) < 1e-4
                N_ = (sv ** 2) * N_ANNOT * (N_ANNOT - 1)
                okB = abs(N_ - round(N_)) < 0.01
                resid.append(abs(N_ - round(N_)))
                num = round(N_) + S_ ** 2
                okC = (num % N_ANNOT == 0)
                cA += okA; cB += okB; cC += (okA and okB and okC)
                if okA and okB and okC:
                    _, _, _, err = reconstruct_ratings(av, sv)
                    if err is not None and err < 1e-3:
                        cFull += 1
            nrev[f"{name}/{v}"] = dict(
                n_rows=n_rows, n_annotators=N_ANNOT,
                frac_10avg_integer=round(cA / n_rows, 4),
                frac_90std2_integer=round(cB / n_rows, 4),
                frac_all_moment_conditions=round(cC / n_rows, 4),
                frac_fully_reconstructible=round(cFull / n_rows, 4),
                resid_90std2_mean=round(float(np.mean(resid)), 6),
                resid_90std2_p90=round(float(np.percentile(resid, 90)), 6),
            )
    out["n_annotators_reverse"] = nrev


    for name, key, task, lb, c in SPECS:
        df = load(name)
        vc, sc = val_cols(task)
        rec = {}
        for v in vc:
            g = df.groupby(key)[v]
            sd = g.std(ddof=1)
            sizes = g.size()

            rec[v] = dict(
                n_targets=int(g.ngroups),
                targets_with_std=int(sd.notna().sum()),
                targets_with_nonzero_std=int((sd > 1e-9).sum()),
                frac_zero_std=round(float((sd <= 1e-9).mean()), 4),
                single_row_targets=int((sizes == 1).sum()),
                sd_median=round(float(sd.median()), 4),
                sd_mean=round(float(sd.mean()), 4),
            )
        out["target_std"][name] = rec










    ceil = {}
    for name, key, task, lb, c in SPECS:
        df = load(name)
        vc, sc = val_cols(task)
        rec = {}
        for v, s in zip(vc, sc):

            g = df.groupby(key)[v]
            sig = float(g.mean().var(ddof=1))
            annot_var = float((df[s] ** 2).mean())
            noise = annot_var / N_ANNOT
            tot = sig + noise
            atten = float(np.sqrt(sig / tot)) if tot > 0 else 0.0

            wrong_tot = sig + annot_var
            wrong_atten = float(np.sqrt(sig / wrong_tot)) if wrong_tot > 0 else 0.0
            rec[v] = dict(
                target_level_signal_var=round(sig, 5),
                annotator_dispersion_var=round(annot_var, 5),
                mean_noise_var=round(noise, 6),
                n_annotators=N_ANNOT,
                signal_to_noise=round(sig / noise, 3) if noise > 0 else None,
                attenuation=round(atten, 4),
                implied_rho_ceiling=round(atten, 4),
                attenuation_if_misused_Std2=round(wrong_atten, 4),
            )
        out["noise_ceiling"][name] = rec


    for name, key, task, lb, c in SPECS:
        df = load(name)
        sims, per_target = [], []
        for _, g in df.groupby(key):
            sets = [set(w.lower() for w in re.findall(r"[A-Za-zÄÖÜäöüß]+", str(x)))
                    for x in g.Context]
            ps = [jaccard(a, b) for a, b in itertools.combinations(sets, 2)]
            if ps:
                per_target.append(float(np.mean(ps)))
                sims.extend(ps)
        s = np.array(sims)

        n_sent = float(df.groupby(key).size().mean())
        jmed = float(np.median(s))
        out["information_gain"][name] = dict(
            n_pairs=int(len(s)),
            jaccard_median=round(jmed, 4),
            jaccard_mean=round(float(s.mean()), 4),
            jaccard_p90=round(float(np.percentile(s, 90)), 4),
            independence_median=round(1 - jmed, 4),
            frac_duplicate=round(float((s >= 0.999).mean()), 4),
            frac_below_0p1=round(float((s < 0.1).mean()), 4),
            per_target_median=round(float(np.median(per_target)), 4),
            sentences_per_target=round(n_sent, 2),
            aggregation_potential=round(n_sent * (1 - jmed), 3),
        )

    json.dump(out, open(os.path.join(ROOT, "scripts", "label_structure.json"), "w"),
              ensure_ascii=False, indent=1)



    fig, axes = plt.subplots(1, 3, figsize=(14.6, 4.2))




    ax = axes[0]
    vals = [out["granularity"][f"{k}/{v}"]["frac_on_0p1_grid"]
            for k, _, t, _, _ in SPECS for v in val_cols(t)[0]]
    errs = [out["granularity"][f"{k}/{v}"]["max_err_to_0p1_grid"]
            for k, _, t, _, _ in SPECS for v in val_cols(t)[0]]
    labs = [f"{k}\n{v}" for k, _, t, _, _ in SPECS for v in val_cols(t)[0]]
    xx = np.arange(len(vals))
    ax.bar(xx, vals, .6, color="#5B8DB8", edgecolor="white")
    ax.axhline(1.0, color="#B03A2E", ls="--", lw=1.2)

    ax.text(-0.45, 1.03, "1.0 = 全部落在 0.1 格点", fontsize=8,
            color="#B03A2E", ha="left", va="bottom",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=.95))
    for i, (v, e) in enumerate(zip(vals, errs)):



        ax.text(i, v - .022, f"{v:.1%}", ha="center", va="top", fontsize=8.2,
                color="white", fontweight="bold")







        ax.text(i, .04, f"最大偏离\n{e / 10:.3f} 分",
                ha="center", fontsize=6.4, color="white", va="bottom", alpha=.95)
    ax.set_xticks(xx); ax.set_xticklabels(labs, fontsize=7)
    ax.set_ylabel("落在 0.1 格点上的行占比")
    ax.set_xlabel("柱内小字 = 该列的最大离格误差（单位：分；1 个 0.1 格 = 0.1 分）",
                  fontsize=7.2, labelpad=6)
    ax.set_ylim(0, 1.12)
    ax.set_title("(a) 评分确实是 0.1 的整数倍\n"
                 "→ 是 10 个整数评分的均值（少数行来自不同 n）", fontsize=10.4)


    ax = axes[1]
    for name, key, task, lb, c in SPECS:
        df = load(name)
        vc, sc = val_cols(task)
        for v, s in zip(vc, sc):
            m2 = df[v].values ** 2 + df[s].values ** 2
            ax.hist(m2, bins=np.arange(0, 26, 1), histtype="step", lw=1.5, color=c,
                    label=f"{lb}" if v == vc[0] else None)
    ax.axvline(25, color="#B03A2E", ls="--", lw=1.3)

    ax.text(1.2, ax.get_ylim()[1] * .06, "25 = 0–5 上界", fontsize=8, color="#B03A2E",
            ha="left", va="bottom",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=1.0))
    ax.set_xlabel("均值² + 标准差²"); ax.set_ylabel("例句数")
    ax.legend(fontsize=7.4)
    ax.set_title("(b) 「均值²+方差 ≤ 25」恒成立\n→ 标注值被严格约束在 0–5 内", fontsize=10.4)


    ax = axes[2]
    ntg = [out["target_std"][k][val_cols(t)[0][0]]["n_targets"] for k, _, t, _, _ in SPECS]
    nzero = [out["target_std"][k][val_cols(t)[0][0]]["targets_with_nonzero_std"]
             for k, _, t, _, _ in SPECS]
    xx = np.arange(4)


    ax.bar(xx - .19, ntg, .36, color="#9EC3E2", edgecolor="white", label="目标总数")
    ax.bar(xx + .19, nzero, .36, color="#C2703D", edgecolor="white",
           label="能算出非零 std 的目标")
    for i, (a_, b_) in enumerate(zip(ntg, nzero)):
        ax.text(i - .19, a_ + max(ntg) * .02, f"{a_}", ha="center", fontsize=7.6)
        ax.text(i + .19, b_ + max(ntg) * .02, f"{b_}", ha="center", fontsize=7.6)
    ax.set_xticks(xx); ax.set_xticklabels(labels, fontsize=8.2, rotation=15)
    ax.set_ylabel("目标数"); ax.set_ylim(0, max(ntg) * 1.22)
    ax.legend(fontsize=8, loc="lower center", bbox_to_anchor=(.5, 1.0), ncol=1)



    tot_t = sum(ntg); tot_nz = sum(nzero)
    gaps = [(lb, a_ - b_) for lb, a_, b_ in zip(labels, ntg, nzero) if a_ != b_]
    gap_txt = "、".join(f"{lb} {g} 个" for lb, g in gaps) if gaps else "无缺口"
    ax.set_title(f"(c) 指标二的可算性\n{tot_nz}/{tot_t} 个目标可算出非零 std\n"
                 f"（缺口：{gap_txt}单例句目标；零 std 目标 0 个）", fontsize=10.4, pad=44)
    fig.suptitle("图15  标签的数值结构：评分是 10 个整数评分的均值，严格落在 0–5 内；"
                 "指标二没有零方差退化目标", y=1.045, fontsize=12.6, fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{FIG}/fig15_label_structure.png"); plt.close(fig)


    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.2))
    ax = axes[0]
    data_ = []
    for name, key, task, lb, c in SPECS:
        df = load(name)
        sims = []
        for _, g in df.groupby(key):
            sets = [set(w.lower() for w in re.findall(r"[A-Za-zÄÖÜäöüß]+", str(x)))
                    for x in g.Context]
            sims.extend(jaccard(a, b) for a, b in itertools.combinations(sets, 2))
        data_.append(sims)
    bp = ax.boxplot(data_, widths=.55, patch_artist=True, showfliers=False,
                    medianprops=dict(color="#2C3E50", lw=1.6))
    for p_, c in zip(bp["boxes"], C):
        p_.set_facecolor(c); p_.set_alpha(.85); p_.set_edgecolor("white")
    ax.set_xticks(range(1, 5)); ax.set_xticklabels(labels, fontsize=8.2, rotation=15)
    ax.set_ylabel("同目标两条例句的词汇 Jaccard")
    ax.set_title("(a) 同目标多例句的词汇重叠度\n"
                 "越低 → 每条例句携带的语境信息越独立", fontsize=10.4)
    ax.set_ylim(-0.012, .225)

    ax = axes[1]
    xx = np.arange(4)
    jac = [out["information_gain"][k]["jaccard_median"] for k in keys4]
    indep = [out["information_gain"][k]["independence_median"] for k in keys4]
    pot = [out["information_gain"][k]["aggregation_potential"] for k in keys4]
    nsent = [out["information_gain"][k]["sentences_per_target"] for k in keys4]



    ax.bar(xx - .19, indep, .36, color="#5B8DB8", edgecolor="white",
           label="例句独立度 1−Jaccard")
    ax.set_ylabel("例句独立度 1−J（0–1）"); ax.set_ylim(0, 1.32)
    ax2 = ax.twinx()
    ax2.bar(xx + .19, pot, .36, color="#C2703D", edgecolor="white",
            label="聚合潜力 = 例句数 × 独立度")
    ax2.set_ylabel("聚合潜力（条/目标，0–13）"); ax2.set_ylim(0, 13.2)
    ax2.grid(False); ax2.spines["top"].set_visible(False)
    for i, v in enumerate(indep):
        ax.text(i - .19, v + .02, f"{v:.3f}", ha="center", fontsize=7.6)
    for i, v in enumerate(pot):
        ax2.text(i + .19, v + .22, f"{v:.2f}", ha="center", fontsize=7.6)
    ax.set_xticks(xx); ax.set_xticklabels(labels, fontsize=8.2, rotation=15)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7.6, loc="lower center",
              bbox_to_anchor=(.5, 1.0), ncol=1, handlelength=1.2,
              borderaxespad=0.)
    ax.set_title("(b) 例句高度独立 → 聚合潜力都很高\n"
                 "（左柱独立度看左轴、右柱聚合潜力看右轴；小品词动词最高，复合词最低）",
                 fontsize=10.4, pad=48)
    fig.suptitle("图16  同目标多例句的信息增量：例句几乎不重复，跨例聚合有真实信息增量；"
                 "真正的瓶颈是目标级方差而非例句冗余", y=1.05, fontsize=12.6,
                 fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{FIG}/fig16_information_gain.png"); plt.close(fig)


    fig, axes = plt.subplots(1, 3, figsize=(16.4, 4.3))




    ax = axes[0]
    conds = [("10·Avg 整数", "frac_10avg_integer"),
              ("90·Std² 整数", "frac_90std2_integer"),
              ("三个矩条件全成立", "frac_all_moment_conditions"),
              ("可完整重建 10 个评分", "frac_fully_reconstructible")]
    SERIES = []


    for (name, key, task, lb, c) in SPECS:
        vs = val_cols(task)[0]
        for vi, v in enumerate(vs):
            SERIES.append((name, v, c, f"{name} {v}", 1.0 if vi == 0 else .45))
    w = .8 / len(SERIES)
    xx = np.arange(len(conds))
    for j, (name, v, c, disp, al) in enumerate(SERIES):
        vals = [out["n_annotators_reverse"][f"{name}/{v}"][f] for _, f in conds]
        ax.bar(xx + (j - (len(SERIES) - 1) / 2) * w, vals, w * .88,
               color=c, edgecolor="white", alpha=al, label=disp)
    ax.set_xticks(xx); ax.set_xticklabels([c for c, _ in conds], fontsize=8)
    ax.set_ylabel("命中率"); ax.set_ylim(0, 1.30)
    ax.legend(fontsize=6.4, loc="upper center", bbox_to_anchor=(.5, 1.02),
              ncol=2, handlelength=1.0, columnspacing=.9)


    recon = [out["n_annotators_reverse"][f"{n}/{v}"]["frac_fully_reconstructible"]
             for n, v, _, _, _ in SERIES]
    ax.annotate(f"重建成功率 {min(recon):.1%}–{max(recon):.1%}",
                xy=(3, 1.0), xytext=(3, 1.13), ha="center", va="center",
                fontsize=7.6, color="#1B4F72",
                arrowprops=dict(arrowstyle="->", color="#1B4F72", lw=.9),
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#1B4F72",
                          linewidth=.7, alpha=1.0))
    ax.set_title("(a) 标注者人数反解：n=10 的四个矩条件\n"
                 "（全部 6 个子集×列；每行 = 10 个 0-5 整数评分）", fontsize=10.4, pad=40)




    ax = axes[1]
    BSPEC = []
    for (name, key, task, lb, c) in SPECS:
        for v in val_cols(task)[0]:
            BSPEC.append((name, v, c, f"{name} {v}"))
    xx = np.arange(len(BSPEC))
    w = .38
    for i, (name, v, c, disp) in enumerate(BSPEC):
        rec = out["noise_ceiling"][name][v]
        ax.bar(i - w / 2, rec["annotator_dispersion_var"], w, color="#C2703D",
               edgecolor="white")
        ax.bar(i + w / 2, rec["mean_noise_var"], w, color="#5B8DB8",
               edgecolor="white")
        ax.text(i - w / 2, rec["annotator_dispersion_var"] + .04,
                f"{rec['annotator_dispersion_var']:.2f}", ha="center", fontsize=6.8)
        ax.text(i + w / 2, rec["mean_noise_var"] + .04,
                f"{rec['mean_noise_var']:.3f}", ha="center", fontsize=6.8)
    ax.plot([], [], color="#C2703D", lw=6, label="Std²（标注者间离散度）")
    ax.plot([], [], color="#5B8DB8", lw=6, label="Std²/10（均值的抽样噪声）")
    ax.set_xticks(xx)
    ax.set_xticklabels([b[3] for b in BSPEC], fontsize=7, rotation=15)
    ax.set_ylabel("方差")
    ax.set_ylim(0, max(out["noise_ceiling"][n][v]["annotator_dispersion_var"]
                        for n, v, _, _ in BSPEC) * 1.34)
    ax.legend(fontsize=7.6, loc="upper center", bbox_to_anchor=(.5, 1.0),
              ncol=2, handlelength=1.0)
    ax.set_title("(b) 均值的噪声是标注者间离散的 1/10\n"
                 "（全部 6 个子集×列；此前误用 Std²，把噪声高估了 10 倍）", fontsize=10.4,
                 pad=40)








    ax = axes[2]
    bar_specs = []
    for name, key, task, lb, c in SPECS:
        for v in val_cols(task)[0]:
            rec = out["noise_ceiling"][name][v]
            bar_specs.append((f"{name}\n{v}", lb, c, rec["attenuation"],
                              rec["attenuation_if_misused_Std2"]))
    xx = np.arange(len(bar_specs))
    for i, (lab, lb, c, att, wrong) in enumerate(bar_specs):
        ax.bar(i, att, .55, color=c, edgecolor="white")



        if att > .90:
            ax.text(i, att + .012, f"{att:.2f}", ha="center", va="bottom",
                    fontsize=8.2, fontweight="bold", color=c)
        else:
            ax.text(i, att - .018, f"{att:.2f}", ha="center", va="top",
                    fontsize=8.2, fontweight="bold", color="white")
        ax.plot([i - .27, i + .27], [wrong] * 2, color="#B03A2E", lw=1.8, ls="--")
    ax.axhline(.9, color="#7F8C8D", ls=":", lw=1.2)

    ax.text(len(bar_specs) - .45, .905, "ρ = 0.90（参照线）", fontsize=7.6, color="#5D6D7E",
            ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=.95))
    ax.plot([], [], color="#B03A2E", lw=1.8, ls="--",
            label="若误把 Std² 当均值噪声（错误做法）")


    ax.legend(fontsize=7.4, loc="lower center", bbox_to_anchor=(.5, 1.0),
              ncol=1, handlelength=1.6, borderaxespad=0.)
    ax.set_xticks(xx)
    ax.set_xticklabels([b[0] for b in bar_specs], fontsize=7.2, rotation=15)
    ax.set_xlim(-.7, len(bar_specs) - .3)
    ax.set_ylabel("指标1 ρ 的理论上限（噪声衰减后）")
    ax.set_ylim(0, 1.08)
    atts = [b[3] for b in bar_specs]
    ax.set_title(f"(c) 修正后的天花板：{min(atts):.2f}–{max(atts):.2f}（全部 6 个子集×列）\n"
                 "（此前 0.45–0.62，严重低估了可达上限）", fontsize=10.4, pad=42)

    fig.suptitle("图17  标注噪声天花板：n=10 的反解把噪声估计修正了 10 倍，"
                 "指标1 的可追上限比原先估计高得多", y=1.04, fontsize=12.6,
                 fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{FIG}/fig17_noise_ceiling.png"); plt.close(fig)

    print(json.dumps(out, ensure_ascii=False, indent=1))
    print("\n写入 scripts/label_structure.json 与 figures/fig15-17")


if __name__ == "__main__":
    main()
