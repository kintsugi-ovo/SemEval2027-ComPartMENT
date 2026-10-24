import json, numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"
FIG.mkdir(exist_ok=True)
SET = ["en-nn/Mod", "en-nn/Head", "de-nn/Mod", "de-nn/Head", "en-pv", "de-pv"]
x = np.arange(len(SET))

dv = json.load(open(ROOT / "scripts" / "decomp_value.json"))
flat = {"en-nn/Mod": dv["en-nn"]["ModAvg"], "en-nn/Head": dv["en-nn"]["HeadAvg"],
        "de-nn/Mod": dv["de-nn"]["ModAvg"], "de-nn/Head": dv["de-nn"]["HeadAvg"],
        "en-pv": dv["en-pv"]["Avg"], "de-pv": dv["de-pv"]["Avg"]}
between = [flat[s]["between_var"] for s in SET]
noise = [flat[s]["within_noise_var"] for s in SET]
signal = [flat[s]["within_signal_var"] for s in SET]

mc = json.load(open(ROOT / "scripts" / "metric2_ceiling.json"))
mck = ["en-nn/ModAvg", "en-nn/HeadAvg", "de-nn/ModAvg", "de-nn/HeadAvg", "en-pv/Avg", "de-pv/Avg"]
ceil = [mc[k]["m2_perfect"] for k in mck]
real = [mc[k]["m2_realistic"] for k in mck]

pr = json.load(open(ROOT / "scripts" / "probe_input_channels.json"))
prk = {"en-nn/ModAvg": ("en-nn", "ModAvg"), "en-nn/HeadAvg": ("en-nn", "HeadAvg"),
        "de-nn/ModAvg": ("de-nn", "ModAvg"), "de-nn/HeadAvg": ("de-nn", "HeadAvg"),
        "en-pv/Avg": ("en-pv", "Avg"), "de-pv/Avg": ("de-pv", "Avg")}
mwe_only = [pr[prk[k][0]][prk[k][1]]["E"]["m1"] for k in mck]
with_ctx = [pr[prk[k][0]][prk[k][1]]["A"]["m1"] for k in mck]
with_seg = [pr[prk[k][0]][prk[k][1]]["B"]["m1"] for k in mck]

sp = json.load(open(ROOT / "scripts" / "mc_strategy_payoff.json"))
d_cost = [sp[k]["S3 +独立噪声"]["m1"] - sp[k]["S1 完美组内"]["m1"] for k in mck]
s2 = [sp[k]["S2 仅组均值"]["m1"] for k in mck]

fig, axes = plt.subplots(2, 2, figsize=(15, 9))
C = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]

ax = axes[0, 0]
w = 0.27
ax.bar(x - w, between, w, label="Between-target (real, type-level)", color=C[0])
ax.bar(x, signal, w, label="Within-target REAL signal", color=C[2])
ax.bar(x + w, noise, w, label="Within-target ANNOTATION NOISE", color=C[1])
ax.set_xticks(x); ax.set_xticklabels(SET, fontsize=9)
ax.set_ylabel("Variance (score^2)")
ax.set_title("(a) Where the label variance lives", fontsize=11)
ax.legend(fontsize=8); ax.axhline(0, color="k", lw=0.6)
ax.set_ylim(bottom=-0.05)

ax = axes[0, 1]
ax.bar(x - 0.2, ceil, 0.4, label="Ceiling: perfect context model", color=C[2])
ax.bar(x + 0.2, real, 0.4, label="Ceiling: realistic model", color=C[0])
ax.set_xticks(x); ax.set_xticklabels(SET, fontsize=9)
ax.set_ylabel("Metric 2 (Spearman of within-target std)")
ax.set_title("(b) Metric 2 is mostly unreachable in nn", fontsize=11)
ax.legend(fontsize=8); ax.axhline(0, color="k", lw=0.8)
ax.set_ylim(-0.05, 0.7)
for i, v in enumerate(ceil):
    ax.text(i - 0.2, v + 0.012, f"{v:.2f}", ha="center", fontsize=8)

ax = axes[1, 0]
ax.bar(x - 0.2, mwe_only, 0.4, label="MWE only (no context)", color=C[0])
ax.bar(x, with_ctx, 0.4, label="MWE + context", color=C[1])
ax.bar(x + 0.2, with_seg, 0.4, label="MWE + segment + context", color=C[2])
ax.set_xticks(x); ax.set_xticklabels(SET, fontsize=9)
ax.set_ylabel("Metric 1 (Spearman)")
ax.set_title("(c) Probe: does adding context help?", fontsize=11)
ax.legend(fontsize=8); ax.axhline(0, color="k", lw=0.8)
for i in range(len(SET)):
    d = with_ctx[i] - mwe_only[i]
    ax.text(i, min(mwe_only[i], with_ctx[i]) - 0.045, f"{d:+.2f}", ha="center", fontsize=8,
            color="red" if d < 0 else "green")

ax = axes[1, 1]
ax.bar(x - 0.2, s2, 0.4, label="Metric 1 if we predict target mean only", color=C[0])
ax.bar(x + 0.2, d_cost, 0.4, label="Cost of adding within-target spread", color=C[1])
ax.set_xticks(x); ax.set_xticklabels(SET, fontsize=9)
ax.set_ylabel("Metric 1 change")
ax.set_title("(d) Making within-target spread costs Metric 1", fontsize=11)
ax.legend(fontsize=8); ax.axhline(0, color="k", lw=0.8)
for i in range(len(SET)):
    ax.text(i + 0.2, d_cost[i] - 0.012, f"{d_cost[i]:+.3f}", ha="center", fontsize=8, color="red")

fig.suptitle("ComPartMENT diagnosis: type-level signal dominates, context signal is noise-limited",
             fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
out = FIG / "fig_diagnosis.png"
fig.savefig(out, dpi=140)
print("wrote", out)
