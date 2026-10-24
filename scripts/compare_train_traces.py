import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RUNS = {
    "v1": {
        "deberta": ROOT / "logs" / "v1" / "compartment-deberta-baseline.log",
        "qwen": ROOT / "logs" / "v1" / "compartment-qwen-instruct-baseline.log",
    },
    "v2": {
        "deberta": ROOT / "logs" / "v2" / "compartment-deberta-baseline.log",
        "qwen": ROOT / "logs" / "v2" / "compartment-qwen-instruct-baseline.log",
    },
}

RE_RUN = re.compile(r"运行 (\d)/5：(\S+)")
RE_TRAIN = re.compile(
    r"\{'loss': '([-0-9.]+)', 'grad_norm': '([-0-9.]+)', 'learning_rate': '([-0-9.e+-]+)', 'epoch': '([0-9.]+)'\}"
)


def parse(path):
    runs, cur = {}, None
    if not path.exists():
        return runs
    for ln in path.read_text(encoding="utf-8").split("\n"):
        m = RE_RUN.search(ln)
        if m:
            cur = m.group(2)
            runs.setdefault(cur, [])
            continue
        if cur is None:
            continue
        m = RE_TRAIN.search(ln)
        if m:
            runs[cur].append(
                {
                    "loss": float(m.group(1)),
                    "grad_norm": float(m.group(2)),
                    "lr": float(m.group(3)),
                    "epoch": float(m.group(4)),
                }
            )
    return runs


def summarize(recs):
    if not recs:
        return {}
    lrs = [r["lr"] for r in recs]
    gns = [r["grad_norm"] for r in recs]
    losses = [r["loss"] for r in recs]
    n_ep = max(r["epoch"] for r in recs)

    buckets = {}
    for r in recs:
        buckets.setdefault(int(r["epoch"]), []).append(r["loss"])
    per_epoch_loss = {e: round(statistics.mean(v), 4) for e, v in sorted(buckets.items())}

    eb = {}
    for r in recs:
        eb.setdefault(int(r["epoch"]), []).append(r)
    per_epoch = {}
    for e, rs in sorted(eb.items()):
        rs = sorted(rs, key=lambda x: x["epoch"])
        per_epoch[e] = {
            "lr_max": round(max(x["lr"] for x in rs), 8),
            "lr_mean": round(statistics.mean(x["lr"] for x in rs), 8),



            "lr_end": round(rs[-1]["lr"], 8),
            "gnorm_max": round(max(x["grad_norm"] for x in rs), 2),
            "gnorm_mean": round(statistics.mean(x["grad_norm"] for x in rs), 2),
            "n": len(rs),
        }
    gns_sorted = sorted(gns)
    p95 = gns_sorted[min(len(gns_sorted) - 1, int(0.95 * len(gns_sorted)))]






    rounds = {}
    for r in recs:
        k = int(-(-r["epoch"] // 1))
        rounds.setdefault(k, []).append(r)
    per_round = {}
    for k, rs in sorted(rounds.items()):
        rs = sorted(rs, key=lambda x: x["epoch"])
        last = rs[-1]
        per_round[k] = {
            "loss_end": last["loss"],
            "lr_end": last["lr"],
            "gnorm_max": round(max(x["grad_norm"] for x in rs), 2),
            "gnorm_mean": round(statistics.mean(x["grad_norm"] for x in rs), 2),
            "n": len(rs),
            "epoch_span": [round(rs[0]["epoch"], 4), round(rs[-1]["epoch"], 4)],
        }
    return {
        "n_logs": len(recs),
        "lr_max": max(lrs),
        "lr_min": min(lrs),
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "loss_min": min(losses),
        "grad_norm_max": max(gns),
        "grad_norm_p95": p95,
        "grad_norm_median": statistics.median(gns),
        "n_grad_spikes": sum(1 for g in gns if g > 50),
        "n_epochs_seen": n_ep,
        "per_epoch_loss": per_epoch_loss,
        "per_epoch": per_epoch,
        "per_round": per_round,
    }


def main():
    out = {}
    for ver, models in RUNS.items():
        for model, path in models.items():
            for run, recs in parse(path).items():
                out[f"{ver}|{model}|{run}"] = summarize(recs)

    if "--json" in sys.argv:
        p = Path(sys.argv[sys.argv.index("--json") + 1])
        p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"→ {p}")
        return

    order = ["single_en-nn", "single_de-nn", "single_en-pv", "single_de-pv", "joint"]
    for model in ["deberta", "qwen"]:
        for run in order:
            print("=" * 100)
            print(f"{model.upper()} / {run}")
            for ver in ["v1", "v2"]:
                k = f"{ver}|{model}|{run}"
                s = out.get(k)
                if not s:
                    continue
                per = " ".join(f"e{e}:{v:.3f}" for e, v in s["per_epoch_loss"].items())
                print(
                    f"  {ver}: logs={s['n_logs']:4d} lr_max={s['lr_max']:.3e} "
                    f"loss {s['loss_first']:.3f}→{s['loss_last']:.3f} (min {s['loss_min']:.3f}) "
                    f"| gnorm med/p95/max = {s['grad_norm_median']:.1f}/{s['grad_norm_p95']:.1f}/{s['grad_norm_max']:.1f} "
                    f"spikes>50={s['n_grad_spikes']}"
                )
                print(f"      per-epoch loss: {per}")


if __name__ == "__main__":
    main()
