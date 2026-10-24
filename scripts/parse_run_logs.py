import json
import re
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
RE_EVAL = re.compile(r"\{'eval_loss'.*'epoch': '([0-9.]+)'\}")


METRIC_KEYS = [
    ("eval_合计|_指标1_ρ", "m1"),
    ("eval_合计|_指标2_ρ", "m2"),
    ("eval_合计|_指标3_RMSE", "m3"),
    ("eval_合计|_综合分", "comb"),
]
SETTING_RE = re.compile(r"eval_(en-nn|de-nn|en-pv|de-pv)\|(ModAvg|HeadAvg|Avg)_指标1_ρ': '([-0-9.]+)'")
TRAIN_MIN_RE = re.compile(r"训练完成，用时 ([0-9.]+) 分钟，loss=([-0-9.]+)")


def _f(pat, ln, default=None):
    m = re.search(pat, ln)
    return m.group(1) if m else default


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
        if cur is None or "'eval_loss'" not in ln or "eval_合计|_指标1_ρ" not in ln:
            continue
        rec = {
            "epoch": float(_f(r"'epoch': '([0-9.]+)'", ln, "0")),
            "eval_loss": float(_f(r"'eval_loss': '([-0-9.]+)'", ln, "nan")),
            "train_minutes": None,
            "settings": {},
        }
        for key, short in METRIC_KEYS:
            rec[short] = float(_f(re.escape(key) + r"': '([-0-9.]+)'", ln, "nan"))
        for sname, col, _v in SETTING_RE.findall(ln):
            for suffix, short in (("指标1_ρ", "m1"), ("指标2_ρ", "m2"), ("指标3_RMSE", "m3")):

                v = _f(re.escape(f"eval_{sname}|{col}_{suffix}") + r"': '([-0-9.]+)'", ln)
                if v is not None:
                    rec["settings"].setdefault(f"{sname}|{col}", {})[short] = float(v)
        runs[cur].append(rec)

    order = list(runs.keys())
    mins = []
    for ln in path.read_text(encoding="utf-8").split("\n"):
        m = TRAIN_MIN_RE.search(ln)
        if m:
            mins.append(float(m.group(1)))
    for name, mm in zip(order, mins):
        if runs[name]:
            runs[name][-1]["train_minutes"] = mm
    return runs


def best_of(recs):
    vals = [r for r in recs if r.get("comb") == r.get("comb")]
    return max(vals, key=lambda r: r["comb"]) if vals else (recs[-1] if recs else None)


def main():
    data = {}
    for ver, models in RUNS.items():
        for model, path in models.items():
            data[(ver, model)] = parse(path)

    out = {}
    for (ver, model), runs in data.items():
        print(f"\n{'='*100}\n{ver.upper()} · {model}\n{'='*100}")
        for name, recs in runs.items():
            if not recs:
                continue
            b, last = best_of(recs), recs[-1]
            tm = b.get("train_minutes")
            print(f"\n  {name}  （{len(recs)} 次 eval，训练 {tm if tm else '?'} 分钟）")
            print(f"    {'epoch':>6} {'指标1':>9} {'指标2':>9} {'RMSE':>9} {'综合分':>9}")
            for r in recs:
                mark = " <-best" if r is b else (" <-交出" if r is last and r is not b else "")
                print(f"    {r['epoch']:>6} {r['m1']:>9.4f} {r['m2']:>9.4f} "
                      f"{r['m3']:>9.4f} {r['comb']:>9.4f}{mark}")
            out[f"{ver}|{model}|{name}"] = {
                "best": b, "last": last, "n_evals": len(recs),
                "epochs": [r["epoch"] for r in recs],
                "trajectory": [{k: r[k] for k in ("epoch", "m1", "m2", "m3", "comb")} for r in recs],
            }

    if "--json" in sys.argv:
        p = Path(sys.argv[sys.argv.index("--json") + 1])
        p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n已保存: {p}")


if __name__ == "__main__":
    main()
