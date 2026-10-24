import json
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_JSON = ROOT / "scripts" / "three_gen_compare.json"

LINE = re.compile(r"^\s*([\d.]+)s\s+\d+\s(.*)$")
EVAL_RE = re.compile(r"^\{(.*)\}$")
KV_RE = re.compile(r"'([^']+)':\s*'([^']*)'")
EPOCH_RE = re.compile(r"'epoch':\s*'?(\d+)'?")
RUN_RE = re.compile(r"^运行\s+(\d+)/5：(\S+)")
TRAIN_RE = re.compile(r"^训练完成，用时\s+([\d.]+)\s+分钟，loss=([\d.]+)")
GN_RE = re.compile(r"^\{'loss': '([\d.]+)', 'grad_norm': '([-\d.e+]+)'")

GEN_DIRS = {
    "v1": ROOT / "v1",
    "v2": ROOT / "v2",
    "v3": ROOT / "logs" / "v3" / "round3",
}
FILES = {
    "deberta": "compartment-deberta-baseline.log",
    "qwen": "compartment-qwen-instruct-baseline.log",
}

SUFFIX = {"指标1_ρ": "m1", "指标2_ρ": "m2", "指标3_RMSE": "m3"}


def split_metric_key(s):
    for suf, short in SUFFIX.items():
        if s.endswith("_" + suf):
            return s[:-(len(suf) + 1)].rstrip("|"), short
    return s, None


def parse(path: Path):
    runs, cur = [], None
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = LINE.match(raw)
        if not m:
            continue
        body = m.group(2).rstrip()
        mr = RUN_RE.match(body)
        if mr:
            cur = dict(name=mr.group(2), curve=[], final=None, random=None,
                       train_minutes=None, train_loss=None, grad_norms=[])
            runs.append(cur)
            continue
        if cur is None:
            continue
        mt = TRAIN_RE.match(body)
        if mt:
            cur["train_minutes"] = float(mt.group(1))
            cur["train_loss"] = float(mt.group(2))
            continue
        mg = GN_RE.match(body)
        if mg:
            try:
                cur["grad_norms"].append(float(mg.group(2)))
            except ValueError:
                pass
            continue
        me = EVAL_RE.match(body)
        if me:
            inner = me.group(1)
            kv = dict(KV_RE.findall(inner))
            ep = EPOCH_RE.search(inner)
            rec = {}
            for k, v in kv.items():
                if k.startswith("eval_"):
                    st, short = split_metric_key(k[len("eval_"):])
                    if short:
                        rec[st + "|" + short] = float(v)
            if ep is not None and "合计|m1" in rec:
                cur["curve"].append(dict(epoch=int(ep.group(1)), **rec))
            continue
        if body.startswith(("en-nn|", "de-nn|", "en-pv|", "de-pv|", "合计|")):
            parts = body.split()
            if len(parts) >= 2:
                st, short = split_metric_key(parts[0])
                try:
                    val = float(parts[-1])
                except ValueError:
                    continue
                if cur["final"] is None:
                    cur["final"] = {}
                cur["final"][st + ("|" + short if short else "")] = val
    return runs


def main():
    out = {}
    for gen, d in GEN_DIRS.items():
        for model, fn in FILES.items():
            p = d / fn
            if not p.exists():
                print(f"⚠ 缺 {p}")
                continue
            out.setdefault(gen, {})[model] = parse(p)

    OUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写出 {OUT_JSON}\n")

    for model in FILES:
        print("=" * 100)
        print(f"{model}：三代对照（每格 = 指标1 / 指标2 / RMSE）")
        print("=" * 100)
        names = []
        for gen in ("v1", "v2", "v3"):
            if gen in out and model in out[gen]:
                names = [r["name"] for r in out[gen][model]]
                break
        hdr = f"{'运行':16s}" + "".join(f"{g:>30s}" for g in ("v1", "v2", "v3"))
        print(hdr)
        for nm in names:
            row = f"{nm:16s}"
            for gen in ("v1", "v2", "v3"):
                rs = out.get(gen, {}).get(model)
                r = next((x for x in (rs or []) if x["name"] == nm), None)
                f = (r or {}).get("final") or {}
                if f:
                    row += (f"{f.get('合计|m1', float('nan')):9.4f} /"
                            f"{f.get('合计|m2', float('nan')):8.4f} /"
                            f"{f.get('合计|m3', float('nan')):7.4f}").rjust(30)
                else:
                    row += "—".rjust(30)
            print(row)

        print(f"\n[{model}] 训练时长与梯度（v3）")
        for r in out.get("v3", {}).get(model, []):
            g = r["grad_norms"]
            if g:
                print(f"  {r['name']:16s} {r['train_minutes']:6.1f} 分钟 | "
                      f"loss {r['train_loss']:.4f} | grad_norm 中位 {statistics.median(g):6.2f} "
                      f"最大 {max(g):7.2f}")
        print()


if __name__ == "__main__":
    main()
