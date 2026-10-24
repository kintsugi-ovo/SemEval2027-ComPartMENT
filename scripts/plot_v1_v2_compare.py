import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

TRACES = json.loads((ROOT / "scripts" / "train_traces.json").read_text(encoding="utf-8"))
PARSED = json.loads((ROOT / "scripts" / "run_logs_parsed.json").read_text(encoding="utf-8"))

C_V1, C_V2 = "#4C72B0", "#C44E52"


def panel_train(ax, model, run, title):
    for ver, color, lab in (("v1", C_V1, "v1"), ("v2", C_V2, "v2")):
        s = TRACES.get(f"{ver}|{model}|{run}")
        if not s:
            continue
        xs = sorted(int(k) for k in s["per_round"])
        ys = [s["per_round"][str(k)]["loss_end"] if str(k) in s["per_round"] else s["per_round"][k]["loss_end"] for k in xs]
        ax.plot(xs, ys, "o-", color=color, lw=2.2, ms=6, label=lab, zorder=3)
    ax.set_title(title, fontsize=10.5, pad=7)
    ax.set_xlabel("training round", fontsize=8.5)
    ax.set_ylabel("train loss (end of round)", fontsize=8.5)
    ax.grid(alpha=0.25, ls=":")
    ax.legend(fontsize=8.5)


def panel_metric(ax, model, run, title, field="m1", ylab="metric-1 (Spearman)"):
    for ver, color, lab in (("v1", C_V1, "v1"), ("v2", C_V2, "v2")):
        k = f"{ver}|{model}|{run}"
        if k not in PARSED:
            continue
        dedup = {}
        for p in PARSED[k]["trajectory"]:
            dedup[p["epoch"]] = p
        xs = sorted(dedup)
        ax.plot(xs, [dedup[x][field] for x in xs], "o-", color=color, lw=2.2, ms=6, label=lab, zorder=3)
    ax.axhline(0, color="#999", lw=0.9, ls="--")
    ax.set_title(title, fontsize=10.5, pad=7)
    ax.set_xlabel("epoch", fontsize=8.5)
    ax.set_ylabel(ylab, fontsize=8.5)
    ax.grid(alpha=0.25, ls=":")
    ax.legend(fontsize=8.5)


def main():
    fig, axes = plt.subplots(3, 3, figsize=(14.5, 11.6))
    fig.suptitle(
        "v1 vs v2  |  DeBERTa: train loss itself rebounds = optimization collapse   |   Qwen: fits train better, metric flat",
        fontsize=12.5, y=0.992,
    )

    panel_train(axes[0][0], "deberta", "joint", "DeBERTa joint: train loss REBOUNDS in v2")
    panel_metric(axes[0][1], "deberta", "joint", "DeBERTa joint: metric-1 per epoch")
    panel_metric(axes[0][2], "deberta", "joint", "DeBERTa joint: metric-2 per epoch", "m2", "metric-2 (Spearman)")

    panel_train(axes[1][0], "deberta", "single_de-nn", "DeBERTa single_de-nn: loss stalls in v2")
    panel_metric(axes[1][1], "deberta", "single_de-nn", "DeBERTa single_de-nn: metric-1")
    panel_metric(axes[1][2], "deberta", "single_de-nn", "DeBERTa single_de-nn: metric-2", "m2", "metric-2 (Spearman)")

    panel_train(axes[2][0], "qwen", "joint", "Qwen joint: v2 fits train data BETTER")
    panel_metric(axes[2][1], "qwen", "joint", "Qwen joint: metric-1 saturates ~0.50")
    panel_metric(axes[2][2], "qwen", "joint", "Qwen joint: metric-2 saturates ~0.36", "m2", "metric-2 (Spearman)")

    for ax in axes.ravel():
        ax.tick_params(labelsize=8)

    fig.tight_layout(rect=(0, 0, 1, 0.978))
    p = OUT / "fig_v1_v2_compare.png"
    fig.savefig(p, dpi=170, bbox_inches="tight")
    print(f"→ {p}")


if __name__ == "__main__":
    main()
