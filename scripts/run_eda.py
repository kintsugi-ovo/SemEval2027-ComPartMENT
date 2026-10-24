import os, re, json, unicodedata, math
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
        plt.rcParams["font.sans-serif"] = [cand] + ["Arial Unicode MS", "DejaVu Sans"]
        break
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "train")
OUT  = os.path.join(ROOT, "scripts")
FIG  = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)

SPECS = [("en-nn", "Compound",     "en", "英语·名词复合词", "语料前三"),
         ("de-nn", "Compound",     "de", "德语·名词复合词", "缀合成词"),
         ("en-pv", "ParticleVerb", "en", "英语·小品词动词", "分写短语"),
         ("de-pv", "ParticleVerb", "de", "德语·小品词动词", "可分前缀")]


def fold(s, ascii_mode=False):
    s = unicodedata.normalize("NFKD", str(s).lower()).replace("ß", "ss")
    s = re.sub(r"[^a-z0-9äöüß']", "", s)
    if ascii_mode:
        for a, b in [("ä", "ae"), ("ö", "oe"), ("ü", "ue")]:
            s = s.replace(a, b)
    return s

def stem_de(b):
    for suf in ["ern", "eln", "en", "er", "e", "t", "ten", "n"]:
        if b.endswith(suf) and len(b) - len(suf) >= 4:
            return b[:-len(suf)]
    return b

def stem_en(w):
    for suf in ["ing", "edly", "ed", "es", "s", "y", "e"]:
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[:-len(suf)]
    return w

def stem_keys(base, lang):
    ks = set()
    for v in (fold(base), fold(base, True)):
        st = stem_de(v) if lang == "de" else stem_en(v)
        for L in (5, 4, 3):
            if len(st) >= L:
                ks.add(st[:L])
    return ks

WORD = re.compile(r"[a-zA-ZäöüÄÖÜß]+(?:[-'][a-zA-ZäöüÄÖÜß]+)*")

def locate(mwe, base, part, lang, ctx):
    low = str(ctx).lower()

    m = re.search(re.escape(mwe.lower()).replace(r"\-", r"[-']?").replace(r"\ ", r"[-']?\s"), low)
    if m:
        return "S0 精确匹配", (m.start(), m.end())
    toks = [(mm.group(0), mm.start(), mm.end()) for mm in WORD.finditer(low)]
    if not toks:
        return "S5 未定位", None

    if fold(mwe, True) and fold(mwe, True) in fold(low, True):
        return "S1 历史转写还原", None
    ks = stem_keys(base, lang); p = fold(part)
    pw = {fold(part), fold(part, True)}
    fw = [fold(t[0]) for t in toks]; fa = [fold(t[0], True) for t in toks]
    if lang == "de":

        for i, t in enumerate(toks):
            if not fw[i].startswith(p):
                continue
            r, ra = fw[i][len(p):], fa[i][len(p):]
            if any(k and (k in r or k in ra) for k in ks):
                return "S2a 德语前缀型(屈折)", (t[1], t[2])

        for i in range(len(toks)):
            if not any(k and (k in fw[i] or k in fa[i]) for k in ks):
                continue
            for j in range(i + 1, min(i + 4, len(toks))):
                if fw[j] in pw or fa[j] in pw:
                    return "S2b 德语后置型(动词分离)", (toks[i][1], toks[j][2])
        return "S5 未定位", None

    for i, t in enumerate(toks):
        if any(k and (k in fw[i] or k in fa[i]) for k in ks):
            return "S3 英文词干前缀", (t[1], t[2])

    for i in range(len(toks) - 1):
        two = fw[i] + fw[i + 1]
        if any(k and k in two for k in ks):
            return "S4 相邻窗口", (toks[i][1], toks[i + 1][2])
    return "S5 未定位", None


def load():
    out = {}
    for name, key, lang, cn, form in SPECS:
        df = pd.read_csv(os.path.join(DATA, f"{name}-train.tsv"), sep="\t")
        df["subset"], df["lang"], df["label"], df["form"] = name, lang, cn, form
        out[name] = (df, key)
    return out

def main():
    data = load()
    S = {}


    S["audit"] = {}
    for name, (df, key) in data.items():
        ctx = df.Context.astype(str)
        par = ctx.str.contains(r"\(", regex=True)
        dup_c, dup_id = int(df.Context.duplicated().sum()), int(df.ContextID.duplicated().sum())
        body = ctx.str.replace(r"\([^()]*\)", "", regex=True).str.strip()
        still = sum(1 for t, b in zip(df[key], body) if str(t).lower() in b.lower())
        S["audit"][name] = dict(n=len(df), targets=int(df[key].nunique()),
                                per_target=round(len(df) / df[key].nunique(), 2),
                                paren=int(par.sum()), paren_pct=round(100 * par.mean(), 1),
                                tail_paren=int(ctx.str.rstrip().str.endswith(")").sum()),
                                dup_context=dup_c, dup_id=dup_id, nulls=int(df.isna().sum().sum()),
                                survive_paren_strip=round(100 * still / len(df), 1))


    S["score"] = {}
    for name, (df, key) in data.items():
        cols = ["ModAvg", "HeadAvg"] if name.endswith("nn") else ["Avg"]
        v = df[cols].values.ravel()
        S["score"][name] = dict(mean=round(float(v.mean()), 3), sd=round(float(v.std()), 3),
                                mn=round(float(v.min()), 2), mx=round(float(v.max()), 2),
                                bins=[int(((v >= i) & (v < i + 1)).sum()) for i in range(5)],
                                p25=round(float(np.percentile(v, 25)), 2),
                                med=round(float(np.median(v)), 2),
                                p75=round(float(np.percentile(v, 75)), 2))


    S["spread"] = {}
    rng_series = {}
    for name, (df, key) in data.items():
        col = "ModAvg" if name.endswith("nn") else "Avg"
        g = df.groupby(key)[col]
        rng = g.max() - g.min()
        rng_series[name] = rng
        S["spread"][name] = dict(med_range=round(float(rng.median()), 2),
                                 mean_range=round(float(rng.mean()), 2),
                                 max_range=round(float(rng.max()), 2),
                                 pct_ge1=round(100 * (rng >= 1).mean(), 1),
                                 pct_ge2=round(100 * (rng >= 2).mean(), 1),
                                 within_sd=round(float(g.std().mean()), 3),
                                 n_std_missing=int(g.std().isna().sum()))


    S["role"] = {}
    for name in ["en-nn", "de-nn"]:
        df, key = data[name]
        r = float(df[["ModAvg", "HeadAvg"]].corr(method="spearman").iloc[0, 1])
        d = (df.ModAvg - df.HeadAvg)
        S["role"][name] = dict(spearman=round(r, 3), mean_mod=round(float(df.ModAvg.mean()), 2),
                               mean_head=round(float(df.HeadAvg.mean()), 2),
                               mean_gap=round(float(d.mean()), 3),
                               pct_head_higher=round(100 * (d < 0).mean(), 1),
                               sd_mean_mod=round(float(df.ModStd.mean()), 3),
                               sd_mean_head=round(float(df.HeadStd.mean()), 3),
                               mod_head_sd_corr=round(float(df[["ModStd", "HeadStd"]].corr().iloc[0, 1]), 3))


    S["annot"] = {}
    for name, (df, key) in data.items():
        cols = ["ModStd", "HeadStd"] if name.endswith("nn") else ["Std"]
        v = df[cols].values.ravel()
        S["annot"][name] = dict(mean=round(float(v.mean()), 3), med=round(float(np.median(v)), 3),
                                pct_gt1=round(100 * (v > 1).mean(), 1),
                                pct_gt1_5=round(100 * (v > 1.5).mean(), 1),
                                n=len(v))


    S["locate"] = {}
    order = ["S0 精确匹配", "S1 历史转写还原", "S2a 德语前缀型(屈折)", "S2b 德语后置型(动词分离)",
             "S3 英文词干前缀", "S4 相邻窗口", "S5 未定位"]
    for name, (df, key) in data.items():
        lang = "en" if name.startswith("en") else "de"
        labs = []
        for _, r in df.iterrows():
            if name.endswith("nn"):
                labs.append(locate(r[key], r[key], "", lang, r.Context)[0])
            else:
                labs.append(locate(r[key], r.Base, r.Particle, lang, r.Context)[0])
        cnt = pd.Series(labs).value_counts()
        S["locate"][name] = {k: int(cnt.get(k, 0)) for k in order}
        S["locate"][name + "_cum"] = {k: round(100 * (1 - cnt.reindex(order[order.index(k):]).fillna(0).sum() / len(df)), 1)
                                      for k in order[:-1]}


    S["style"] = {}
    for name, (df, key) in data.items():
        c = df.Context.astype(str)

        umlaut = np.mean([bool(re.search(r"[äöüÄÖÜ]", x)) for x in c])

        ascii_share = np.mean([bool(re.search(r"(ae|oe|ue|ss)", x)) and not re.search(r"[äöüÄÖÜß]", x) for x in c])
        sharp_s = np.mean([("ß" in x) or ("ss" in x and False) for x in c])
        us_quote = np.mean([('"' in x) or ("“" in x) for x in c])
        semi = np.mean([";" in x for x in c])
        digits = np.mean([bool(re.search(r"\d", x)) for x in c])
        S["style"][name] = dict(umlaut=round(float(umlaut), 4),
                                ascii_translit=round(float(ascii_share), 4),
                                sharp_s=round(float(sharp_s), 4),
                                us_quote=round(float(us_quote), 4),
                                semicolon=round(float(semi), 4),
                                digits=round(float(digits), 4),
                                len_mean=round(float(c.str.len().mean()), 1),
                                words_mean=round(float(c.str.split().str.len().mean()), 1))


    S["len"] = {}
    for name, (df, key) in data.items():
        wl = df.Context.astype(str).str.split().str.len()
        S["len"][name] = dict(words_med=int(wl.median()), words_p90=int(np.percentile(wl, 90)),
                              words_max=int(wl.max()),
                              n_per_target=S["audit"][name]["per_target"])


    S["leak"] = {}
    for name, (df, key) in data.items():

        dup = df.groupby("Context")[key].nunique()
        S["leak"][name] = dict(shared_ctx=int((dup > 1).sum()),
                               same_target_ctx_dup=int(df.duplicated(["Context", key]).sum()))

        tg = sorted(set(df[key].astype(str)))
        nested = 0
        for a in tg:
            for b in tg:
                if a != b and len(a) >= 4 and a.lower() in b.lower():
                    nested += 1; break
        S["leak"][name]["nested_targets"] = nested


    S["order"] = {}
    for name, (df, key) in data.items():
        cols = ["ModAvg", "HeadAvg"] if name.endswith("nn") else ["Avg"]
        rec = {}
        for c in cols:
            inc, rho = [], []
            for _, g in df.groupby(key):
                v = g.sort_values("ContextID")[c].values
                if len(v) > 3:
                    inc.append(float((np.diff(v) > 0).mean()))
                    rho.append(stats.spearmanr(g.ContextID, g[c])[0])
            rec[c] = dict(inc_frac=round(float(np.mean(inc)), 3),
                          spearman=round(float(np.mean(rho)), 3),
                          frac_gt05=round(float(np.mean(np.array(rho) > .5)), 3))

        ids, yrs = [], []
        for _, r in df.iterrows():
            mm = re.search(r"\b(1[6-9]\d\d|20[0-2]\d)\b", str(r.Context))
            if mm:
                ids.append(r.ContextID); yrs.append(int(mm.group(1)))
        rec["year_evidence"] = dict(n=len(yrs),
                                    pearson=round(float(np.corrcoef(ids, yrs)[0, 1]), 3)
                                    if len(set(yrs)) > 5 else None)
        def _contig(v):
            d = np.unique(np.diff(np.sort(v)))
            return len(d) == 1 and d[0] == 1
        rec["contiguous_within_mwe"] = bool(
            df.groupby(key).ContextID.apply(_contig).all())
        S["order"][name] = rec


    from scipy import stats as _st
    rng = np.random.default_rng(42)
    S["leakage"] = {}
    for name, (df, key) in data.items():
        d = df.copy()
        gid = d[key].values; n = len(d)
        d["rk"] = d.groupby(key).cumcount()
        sz = d.groupby(key)[key].transform("size")
        pred = (d["rk"] / (sz - 1).replace(0, np.nan)).fillna(.5).values * 5
        y = d["ModAvg"].values if name.endswith("nn") else d["Avg"].values
        idx = rng.permutation(n); te = idx[int(.8 * n):]
        ut = pd.unique(gid); perm = rng.permutation(len(ut))
        hold = set(ut[perm[:int(.2 * len(ut))]])
        te2 = np.array([i for i in range(n) if gid[i] in hold])
        S["leakage"][name] = dict(
            full=round(float(_st.spearmanr(pred, y)[0]), 3),
            random_row_split=round(float(_st.spearmanr(pred[te], y[te])[0]), 3),
            mwe_disjoint_split=round(float(_st.spearmanr(pred[te2], y[te2])[0]), 3))

    json.dump(S, open(os.path.join(OUT, "stats.json"), "w"), ensure_ascii=False, indent=1)


    labels = [s[3] for s in SPECS]; keys4 = [s[0] for s in SPECS]
    C = ["#3C6E9E", "#7FA8C4", "#C2703D", "#E0A87E"]


    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.9))
    ns = [S["audit"][k]["n"] for k in keys4]; ts = [S["audit"][k]["targets"] for k in keys4]
    x = np.arange(4)
    for a, t, v in zip(ax, ["例句数", "目标表达数", "每目标平均例句数"],
                       [ns, ts, [S["audit"][k]["per_target"] for k in keys4]]):
        b_ = a.bar(x, v, color=C, width=.62, edgecolor="white", linewidth=1.2)
        a.set_title(f"({chr(97 + list(ax).index(a))}) {t}")
        a.set_xticks(x); a.set_xticklabels(labels, fontsize=8.2, rotation=18)
        for r, vv in zip(b_, v):
            a.text(r.get_x() + r.get_width() / 2, vv, f"{vv:g}", ha="center", va="bottom", fontsize=8.5)
        a.set_ylim(0, max(v) * 1.18)
    fig.suptitle("图1  四个子集的数据规模", y=1.06, fontsize=13, fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{FIG}/fig1_overview.png"); plt.close(fig)




    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.3))
    for a, k, lb, c in zip(axes, keys4, labels, C):
        df, key = data[k]
        cols = ["ModAvg", "HeadAvg"] if k.endswith("nn") else ["Avg"]
        v = df[cols].values.ravel()
        a.hist(v, bins=np.arange(-0.25, 5.6, 0.5), color=c, edgecolor="white", linewidth=.8)
        a.axvline(v.mean(), color="#B03A2E", lw=1.6, ls="--")
        a.set_title(f"({chr(97 + list(axes).index(a))}) {lb}\n"
                    f"均值 {v.mean():.2f}（峰值 {int((v>=np.floor(v.mean())).sum())} 例）",
                    fontsize=9.6)
        a.set_xlabel("组合性评分 (0–5)"); a.set_xlim(-0.3, 5.3)
        a.set_ylabel("例句数")
    fig.suptitle("图2  人工组合性评分的分布（红虚线为均值；四面板各自独立纵轴，规模不可直接比高度）",
                 y=1.10, fontsize=13, fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{FIG}/fig2_score_dist.png"); plt.close(fig)


    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.1))
    data_ = [rng_series[k].values for k in keys4]
    bp = axes[0].boxplot(data_, widths=.55, patch_artist=True, showfliers=False,
                         medianprops=dict(color="#2C3E50", lw=1.6))
    for p_, c in zip(bp["boxes"], C):
        p_.set_facecolor(c); p_.set_alpha(.85); p_.set_edgecolor("white")
    for i, d in enumerate(data_, 1):
        axes[0].plot(np.random.default_rng(i).normal(i, .055, len(d)), d, "o", ms=1.6,
                     color="white", alpha=.35, mec="none")
    axes[0].set_xticks(range(1, 5)); axes[0].set_xticklabels(labels, fontsize=8.4, rotation=15)
    axes[0].set_ylabel("同一目标跨语境分数极差 (max–min)")
    axes[0].set_title("(a) 极差分布")
    p1 = [S["spread"][k]["pct_ge1"] for k in keys4]
    p2 = [S["spread"][k]["pct_ge2"] for k in keys4]
    w = .36; xx = np.arange(4)
    axes[1].bar(xx - w / 2, p1, w, label="极差 ≥ 1.0", color="#5B8DB8", edgecolor="white")
    axes[1].bar(xx + w / 2, p2, w, label="极差 ≥ 2.0", color="#C2703D", edgecolor="white")
    for i in range(4):
        axes[1].text(xx[i] - w / 2, p1[i] + 1.2, f"{p1[i]:.0f}%", ha="center", fontsize=8.5)
        axes[1].text(xx[i] + w / 2, p2[i] + 1.2, f"{p2[i]:.0f}%", ha="center", fontsize=8.5)
    axes[1].set_xticks(xx); axes[1].set_xticklabels(labels, fontsize=8.4, rotation=15)
    axes[1].set_ylabel("目标表达占比 (%)"); axes[1].set_ylim(0, 112)
    axes[1].legend(loc="upper left", fontsize=9)
    axes[1].set_title("(b) 跨语境波动显著的目标比例")
    fig.suptitle("图3  组合性的语境依赖性：同一表达在不同句子中的分数波动", y=1.03, fontsize=13, fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{FIG}/fig3_context_spread.png"); plt.close(fig)


    fig, axes = plt.subplots(4, 4, figsize=(15.5, 11.5), sharey=True)
    for r, (k, lb) in enumerate(zip(keys4, labels)):
        df, key = data[k]; col = "ModAvg" if k.endswith("nn") else "Avg"
        g = df.groupby(key)[col]
        rng = (g.max() - g.min()).sort_values(ascending=False)
        picks = list(rng.index[:2]) + list(rng.index[-2:]) if len(rng) > 6 else list(rng.index)
        for c_, mwe in enumerate(picks):
            a = axes[r, c_]; sub = df[df[key] == mwe].sort_values("ContextID")
            a.plot(range(len(sub)), sub[col], "o-", ms=4.2, lw=1.5,
                   color="#C2703D" if rng[mwe] >= 2 else "#3C6E9E")
            sdcol = "ModStd" if k.endswith("nn") else "Std"
            for xi, sdv in enumerate(sub[sdcol]):
                a.errorbar(xi, sub[col].iloc[xi], yerr=sdv, fmt="none",
                           ecolor="#8A8A8A", alpha=.55, capsize=2.2, lw=.9)
            a.set_ylim(-.4, 5.4)





            n_ex = len(sub)
            if n_ex >= 3:
                rho_s = stats.spearmanr(np.arange(n_ex), sub[col].values)[0]





                vals_ = sub[col].values.astype(float)
                lo_y, hi_y = -0.4, 5.4
                top_data = float(np.max(vals_)) if n_ex else 0.0

                y_lab = min(0.97, (top_data - lo_y) / (hi_y - lo_y) + 0.16)
                a.set_ylim(lo_y, lo_y + (hi_y - lo_y) / max(1e-9, y_lab))
                a.text(.03, y_lab - .055, f"ρ={rho_s:+.2f}", transform=a.transAxes,
                       fontsize=8.6, color="#2C3E50", ha="left", va="top",
                       bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#B0BEC5",
                                 linewidth=.6, alpha=.95), zorder=6)
            else:

                a.text(.03, .93, f"n={n_ex}，ρ 未定义", transform=a.transAxes,
                       fontsize=7.6, color="#7F8C8D", ha="left", va="top",
                       bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#B0BEC5",
                                 linewidth=.6, alpha=.95), zorder=6)
            a.set_title(f"({chr(97 + r * 4 + c_)}) {mwe}", fontsize=9.5)


            a.set_xlim(-.6, max(n_ex - 1, 0) + .6)
            a.set_xticks(range(n_ex))
            a.set_xlabel("例句（按 ContextID 排序）", fontsize=8)
            if c_ == 0: a.set_ylabel("组合性评分", fontsize=9)
        axes[r, 0].text(-0.28, .5, lb, transform=axes[r, 0].transAxes, rotation=90,
                        va="center", ha="center", fontsize=11, fontweight="bold")
        for c_ in range(len(picks), 4):
            axes[r, c_].axis("off")
    fig.suptitle("图4  波动最大（左二）与最稳定（右二）目标的逐例句评分\n"
                 "横轴按 ContextID 排序：面板内的 ρ = 「例句序号 vs 评分」的 Spearman 秩相关，"
                 "**小品词动词的 ρ 普遍接近 +1、英语复合词多为中等**——\n"
                 "行序本身与评分强相关（行序伪影，见图11），这不是真实的历时趋势",
                 y=1.02, fontsize=12.5, fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{FIG}/fig4_per_example.png"); plt.close(fig)


    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9))
    for a, k, lb in zip(axes[:2], ["en-nn", "de-nn"], ["英语·名词复合词", "德语·名词复合词"]):
        df, key = data[k]
        a.scatter(df.ModAvg, df.HeadAvg, s=5, alpha=.22, color="#3C6E9E", edgecolor="none")
        a.plot([0, 5], [0, 5], "--", color="#B03A2E", lw=1.1, label="对角线（两成分贡献相同）")
        a.set_xlabel("修饰语贡献 ModAvg"); a.set_ylabel("中心语贡献 HeadAvg")
        r = S["role"][k]["spearman"]
        a.set_title(f"({chr(97+list(axes[:2]).index(a))}) {lb}\nSpearman ρ = {r:.3f}")
        a.legend(fontsize=8, loc="upper left")
        below = S["role"][k]["pct_head_higher"]
        a.text(.97, .04, f"中心语贡献更高的比例：{below:.0f}%", transform=a.transAxes,
               ha="right", fontsize=8.5, color="#B03A2E")





    d_en = (data["en-nn"][0].ModAvg - data["en-nn"][0].HeadAvg).values
    d_de = (data["de-nn"][0].ModAvg - data["de-nn"][0].HeadAvg).values
    parts = axes[2].violinplot([d_en, d_de], positions=[0, 1], widths=.8,
                                showextrema=False, showmedians=False)
    for body, c in zip(parts["bodies"], ["#3C6E9E", "#C2703D"]):
        body.set_facecolor(c); body.set_alpha(.9); body.set_edgecolor("white")
        body.set_linewidth(.8)
    axes[2].axhline(0, color="#B03A2E", ls="--", lw=1.1)
    axes[2].set_ylabel("ModAvg − HeadAvg")
    axes[2].set_xlabel("语言")
    axes[2].set_xticks([0, 1]); axes[2].set_xticklabels(["英语", "德语"], fontsize=9)
    axes[2].set_xlim(-.6, 1.6)

    for i, (arr, c) in enumerate(zip([d_en, d_de], ["#3C6E9E", "#C2703D"])):
        m = float(np.mean(arr))
        axes[2].plot([i - .17, i + .17], [m, m], color="white", lw=2.8, zorder=5)
        axes[2].plot([i - .17, i + .17], [m, m], color=c, lw=1.8, zorder=6)
        axes[2].text(i, m + (.10 if m >= 0 else -.10), f"{m:+.2f}", ha="center",
                     va="bottom" if m >= 0 else "top", fontsize=8.6, color=c,
                     fontweight="bold", zorder=7,
                     bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=.85))
    axes[2].set_title("(c) 成分贡献差值分布（横线 = 均值差）")
    fig.suptitle("图5  名词复合词的成分角色不对称：修饰语 vs 中心语", y=1.05, fontsize=13, fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{FIG}/fig5_constituent_role.png"); plt.close(fig)


    fig, axes = plt.subplots(1, 2, figsize=(12.2, 3.9))
    for k, lb, c in zip(keys4, labels, C):
        df, key = data[k]
        cols = ["ModStd", "HeadStd"] if k.endswith("nn") else ["Std"]
        axes[0].hist(df[cols].values.ravel(), bins=np.arange(0, 3.3, .1), histtype="step",
                     lw=1.7, label=lb, color=c)
    axes[0].axvline(1.5, color="#B03A2E", ls="--", lw=1.3)
    axes[0].set_xlabel("人工标注标准差 Std"); axes[0].set_ylabel("例句数")
    axes[0].set_yscale("log")



    lo, hi = axes[0].get_ylim()
    axes[0].set_ylim(lo, hi * 3.0)
    axes[0].text(1.53, axes[0].get_ylim()[1] * .55, "Std=1.5\n（分歧显著区）", fontsize=8.5,
                 color="#B03A2E", va="top", ha="left",
                 bbox=dict(boxstyle="round,pad=0.30", fc="white", ec="#B03A2E",
                           linewidth=.8, alpha=1.0))
    axes[0].legend(fontsize=8.2, loc="upper right")
    axes[0].set_title("(a) 标注离散度分布")
    for i, (k, lb) in enumerate(zip(keys4, labels)):
        df, key = data[k]
        col = "ModAvg" if k.endswith("nn") else "Avg"
        sd = "ModStd" if k.endswith("nn") else "Std"
        axes[1].scatter(df[col], df[sd], s=4, alpha=.16, color=C[i], edgecolor="none", label=lb)
    axes[1].set_xlabel("平均组合性评分"); axes[1].set_ylabel("人工标注标准差")
    axes[1].set_title("(b) 评分越低 → 标注分歧越大", pad=30)

    axes[1].legend(fontsize=8.2, markerscale=4, loc="lower center",
                   bbox_to_anchor=(.5, 1.0), ncol=4, frameon=False,
                   borderaxespad=0.)
    fig.suptitle("图6  人工标注噪声结构（低分区的分歧不等于脏数据）", y=1.04, fontsize=13, fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{FIG}/fig6_annot_noise.png"); plt.close(fig)


    PLAN = {"en-nn": ["S0 精确匹配", "S1 历史转写还原", "S3 英文词干前缀"],
            "de-nn": ["S0 精确匹配", "S1 历史转写还原", "S2a 德语前缀型(屈折)"],
            "en-pv": ["S0 精确匹配", "S3 英文词干前缀", "S4 相邻窗口"],
            "de-pv": ["S0 精确匹配", "S1 历史转写还原", "S2a 德语前缀型(屈折)",
                      "S2b 德语后置型(动词分离)"]}
    SHORT = {"S0 精确匹配": "S0 精确", "S1 历史转写还原": "S1 转写还原",
             "S2a 德语前缀型(屈折)": "S2a 前缀屈折", "S2b 德语后置型(动词分离)": "S2b 动词分离",
             "S3 英文词干前缀": "S3 词干", "S4 相邻窗口": "S4 相邻窗口"}
    fig, axes = plt.subplots(1, 4, figsize=(15.2, 3.9))
    for a, k, lb, c in zip(axes, keys4, labels, C):
        plan = PLAN[k]; cnt = S["locate"][k]; n = S["audit"][k]["n"]
        cum, base = [], 0.0
        for st in plan:
            base += cnt.get(st, 0) / n * 100; cum.append(base)
        w = .58
        for j, st in enumerate(plan):
            inc = cnt.get(st, 0) / n * 100
            a.bar(j, inc, w, bottom=cum[j] - inc, color=c, edgecolor="white", linewidth=1.1)






            if inc >= 8.0:
                a.text(j, cum[j] - inc / 2, f"+{inc:.1f}", ha="center", va="center",
                       fontsize=8.4, color="white", fontweight="bold")
            else:
                y_lab = max(cum[j] - inc - 4.5, 1.5)

                a.plot([j, j], [cum[j] - inc, y_lab + 1.6], color=c, lw=.7,
                       ls="-", alpha=.65, zorder=5)
                a.text(j, y_lab, f"+{inc:.1f}", ha="center", va="center",
                       fontsize=7.8, color=c, fontweight="bold",
                       bbox=dict(boxstyle="round,pad=0.22", fc="white",
                                 ec=c, linewidth=.8, alpha=1.0),
                       zorder=6)
        gap = 100 - cum[-1]
        a.bar(len(plan), cum[-1], w, color=c, edgecolor="white", linewidth=1.1)
        a.bar(len(plan), gap, w, bottom=cum[-1], color="#D4D4D4", edgecolor="white",
              linewidth=1.1, hatch="//")
        a.axhline(100, color="#B03A2E", ls=":", lw=1.1)
        a.set_xticks(range(len(plan) + 1))
        a.set_xticklabels([SHORT[x] for x in plan] + ["累计"], fontsize=8, rotation=16)
        a.set_ylim(0, 126); a.set_yticks([0, 25, 50, 75, 100])
        a.set_title(f"({chr(97 + list(axes).index(a))}) {lb}\n"
                    f"累计定位 {cum[-1]:.1f}%　未定位 {gap:.1f}%", fontsize=10, color=c)
        if gap > 2:
            a.text(len(plan), cum[-1] + gap / 2, f"未定位\n{gap:.0f}%", ha="center",
                   va="center", fontsize=7.6, color="#555")
    axes[0].set_ylabel("占该子集例句比例 (%)")
    fig.suptitle("图7  预处理策略阶梯：每条规则的增量贡献\n"
                 "条内数字为该规则单独救回的例句占比；灰斜纹为规则失效、需引入形态分析器的部分",
                 y=1.14, fontsize=12.5)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig7_locate_cascade.png"); plt.close(fig)


    steps = ["S0 精确匹配", "S1 历史转写还原", "S2a 德语前缀型(屈折)",
             "S2b 德语后置型(动词分离)", "S3 英文词干前缀", "S4 相邻窗口"]
    fig, ax = plt.subplots(figsize=(11.2, 4.0))
    present = [s for s in steps]
    bottom = np.zeros(4)
    colors = ["#2E5A88", "#4E8CC0", "#7FB3D8", "#E59866", "#C2703D", "#8C4A2F"]
    for s, c in zip(present, colors):
        v = np.array([S["locate"][k].get(s, 0) / S["audit"][k]["n"] * 100 for k in keys4])
        ax.bar(np.arange(4), v, .6, bottom=bottom, label=s, color=c, edgecolor="white", linewidth=1)
        bottom += v
    v = np.array([S["locate"][k].get("S5 未定位", 0) / S["audit"][k]["n"] * 100 for k in keys4])
    ax.bar(np.arange(4), v, .6, bottom=bottom, label="S5 未定位", color="#B0B0B0",
           edgecolor="white", linewidth=1, hatch="//")
    for i in range(4):

        ax.text(i, bottom[i] + 1.8, f"{bottom[i]:.1f}%", ha="center", fontsize=9.5,
                fontweight="bold", zorder=5,
                bbox=dict(facecolor="white", edgecolor="none", alpha=.9, pad=1.8))
    ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=8.6)
    ax.set_ylabel("占该子集例句比例 (%)"); ax.set_ylim(0, 116)
    ax.legend(fontsize=8.4, ncol=4, loc="upper center", bbox_to_anchor=(.5, -.14))
    ax.set_title("图8  定位结果的构成分解：各类策略各贡献多少命中率", fontsize=12.5)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig8_locate_breakdown.png"); plt.close(fig)


    fig, axes = plt.subplots(1, 3, figsize=(13.8, 3.9))
    feats = [("sharp_s", "含 ß 的句子比例（德语历史拼写信号）", 100),
             ("digits", "含阿拉伯数字的比例（年代代理：战前文档用拼写数字）", 100),
             ("semicolon", "含分号的比例（现代标点规范信号）", 100)]



    ymax = 30
    for a, (f, t, sc) in zip(axes, feats):
        v = [S["style"][k][f] * sc for k in keys4]
        b = a.bar(range(4), v, .6, color=C, edgecolor="white")
        a.set_xticks(range(4)); a.set_xticklabels(labels, fontsize=8.2, rotation=18)
        a.set_title(f"({chr(97 + list(axes).index(a))}) {t}", fontsize=10.5)
        for r_, vv in zip(b, v):
            a.text(r_.get_x() + r_.get_width() / 2, vv, f"{vv:.1f}", ha="center",
                   va="bottom", fontsize=8)
        a.set_ylim(0, ymax)
        a.set_yticks([0, 10, 20, 30])
        a.set_ylabel("该子集例句占比 (%)", fontsize=9)
    fig.suptitle("图9  可用作「年代代理」的表层特征：历史文本的书写痕迹\n"
                 "（三面板共享同一纵轴 0–30%，柱高可直接横向比较）",
                 y=1.10, fontsize=13, fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{FIG}/fig9_style_features.png"); plt.close(fig)


    fig, axes = plt.subplots(1, 2, figsize=(12.2, 3.9))
    v = [S["audit"][k]["survive_paren_strip"] for k in keys4]
    b = axes[0].bar(range(4), v, .6, color="#B03A2E", edgecolor="white")
    b[0].set_color("#3C6E9E"); b[1].set_color("#3C6E9E")
    for i, (r_, vv, pp) in enumerate(zip(b, v, [S["audit"][k]["paren_pct"] for k in keys4])):
        axes[0].text(r_.get_x() + r_.get_width() / 2, vv, f"{vv:.1f}%", ha="center", va="bottom", fontsize=8.6)
        axes[0].text(i, 4, f"含括号 {pp:.1f}%", ha="center", fontsize=8, color="white")
    axes[0].set_ylim(0, 112); axes[0].set_xticks(range(4)); axes[0].set_xticklabels(labels, fontsize=8.4, rotation=15)
    axes[0].set_ylabel("剥离所有括号后仍能定位目标的比例 (%)")
    axes[0].set_title("(a) 全量括号剥离的副作用")
    for i, k in enumerate(keys4):
        df, key = data[k]
        cnt = df.groupby(key).size()
        axes[1].hist(cnt, bins=np.arange(.5, 11.5, 1), histtype="step", lw=1.8, color=C[i], label=labels[i])
    axes[1].set_xlabel("每个目标表达的例句数"); axes[1].set_ylabel("目标表达数")
    axes[1].set_xticks(range(1, 11)); axes[1].legend(fontsize=8)
    axes[1].set_title("(b) 例句覆盖不均衡")
    fig.suptitle("图10  数据质量与覆盖度检查", y=1.04, fontsize=13, fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{FIG}/fig10_quality.png"); plt.close(fig)



    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.2))
    ks_ = keys4
    base = [S["leakage"][k]["mwe_disjoint_split"] for k in ks_]
    rnd = [S["leakage"][k]["random_row_split"] for k in ks_]
    x = np.arange(4); w = .36
    axes[0].bar(x - w/2, rnd, w, label="逐行随机 8:2 划分", color="#C2703D", edgecolor="white")
    axes[0].bar(x + w/2, base, w, label="MWE 互斥 8:2 划分", color="#3C6E9E", edgecolor="white")
    for i in range(4):
        axes[0].text(x[i]-w/2, rnd[i]+.012, f"{rnd[i]:.2f}", ha="center", fontsize=8.6)
        axes[0].text(x[i]+w/2, base[i]+.012, f"{base[i]:.2f}", ha="center", fontsize=8.6)
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels, fontsize=8.4, rotation=15)
    axes[0].set_ylabel("Spearman ρ"); axes[0].set_ylim(0, .82)
    axes[0].axhline(.50, color="#B03A2E", ls="--", lw=1.2)
    axes[0].text(-0.42, .515, "参考线：ComposiGen 已发表最好系统 ρ = 0.50", fontsize=8,
                 color="#B03A2E", ha="left")
    axes[0].legend(fontsize=8.6); axes[0].set_title("(a) 零文本基线：预测 = 组内行号秩")
    inc_mod, inc_head, inc_avg = [], [], []
    for i, (k, lb) in enumerate(zip(ks_, labels)):
        if k.endswith("nn"):
            inc_mod.append(S["order"][k]["ModAvg"]["inc_frac"])
            inc_head.append(S["order"][k]["HeadAvg"]["inc_frac"])
        else:
            inc_avg.append(S["order"][k]["Avg"]["inc_frac"])
    axes[1].plot([0, 1], inc_mod, "o-", color="#7FA8C4", lw=2, ms=7, label="复合词 · 修饰语贡献")
    axes[1].plot([0, 1], inc_head, "s-", color="#2E5A88", lw=2, ms=7, label="复合词 · 中心语贡献")
    axes[1].plot([2, 3], inc_avg, "D-", color="#C2703D", lw=2, ms=7, label="小品词动词 · 整体")
    for xs, ys, c in [([0, 1], inc_mod, "#7FA8C4"), ([0, 1], inc_head, "#2E5A88"),
                      ([2, 3], inc_avg, "#C2703D")]:
        for xi, yi in zip(xs, ys):
            axes[1].text(xi, yi+.022, f"{yi:.2f}", ha="center", fontsize=8.4, color=c)
    axes[1].axhline(.5, color="#B03A2E", ls="--", lw=1.3)

    axes[1].text(2.62, .455, "0.5 = 完全随机顺序", fontsize=8, color="#B03A2E",
                 ha="right", va="bottom")
    axes[1].set_xticks(range(4)); axes[1].set_xticklabels(labels, fontsize=8.4, rotation=15)
    axes[1].set_xlim(-.35, 3.35); axes[1].set_ylim(.42, .95)
    axes[1].set_ylabel("相邻有序对中分数上升的比例"); axes[1].legend(fontsize=8.4)
    axes[1].set_title("(b) 行序是否编码金标评分")
    fig.suptitle("图11  行序伪影与泄漏：ContextID 在同一目标内近似按金标排序，"
                 "导致零文本基线也能拿到高分", y=1.04, fontsize=13, fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{FIG}/fig11_order_leakage.png"); plt.close(fig)

    print(json.dumps(S, ensure_ascii=False, indent=1))
    print("\n图已保存至", FIG)

if __name__ == "__main__":
    main()
