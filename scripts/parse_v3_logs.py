import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "logs" / "v3" / "round3"
OUT_JSON = ROOT / "scripts" / "v3_run_logs.json"


LINE_RE = re.compile(r"^\s*[\d.]+s\s+\d+\s(.*)$")


EVAL_RE = re.compile(r"^\{(.*)\}$")
KV_RE = re.compile(r"'([^']+)':\s*'([^']*)'")


EPOCH_RE = re.compile(r"'epoch':\s*'?(\d+)'?")

RUN_RE = re.compile(r"^运行\s+(\d+)/5：(\S+)\s+\|\s+参与子集\s+\[(.*)\]")
TRAIN_RE = re.compile(r"^训练完成，用时\s+([\d.]+)\s+分钟，loss=([\d.]+)")
PROBE_RE = re.compile(r"^t3 micro-batch=(\d+): 通过，峰值显存 ([\d.]+) GiB，每样本 ([\d.]+) ms")
RAND_RE = re.compile(r"^随机初始化模型基线（[\d.]+s）: 指标1 ρ=([-\d.nan]+) \| 指标2 ρ=([-\d.nan]+) \| RMSE=([\d.]+)")
DATALEN_RE = re.compile(r"^口径自检 ✓ 有效 batch = (\d+)×(\d+) = (\d+) \| 每轮批数 (\d+) \| 总更新步 (\d+) \| warmup (\d+) 步")
READBACK_RE = re.compile(r"^口径回读 ✓ 有效 batch=(\d+) \| 每轮批数=(\d+) \| 总更新步=(\d+) \| warmup=(\d+) 步")
SETTING_RE = re.compile(r"^(\w+)\s+=\s+(.+?)\s*(?:←.*)?$")

METRIC_SUFFIX = {
    "指标1_ρ": "m1",
    "指标2_ρ": "m2",
    "指标3_RMSE": "m3",
}


def split_metric_key(setting_and_metric: str):
    for suffix, short in METRIC_SUFFIX.items():
        if setting_and_metric.endswith("_" + suffix):
            setting = setting_and_metric[:-(len(suffix) + 1)].rstrip("|")
            return setting, short
    return setting_and_metric, None



def parse_log(path: Path):
    runs = []
    cur = None
    setup = {}
    probe = None
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = LINE_RE.match(raw)
        if not m:
            continue
        body = m.group(1).rstrip()

        m = RUN_RE.match(body)
        if m:
            cur = dict(idx=int(m.group(1)), name=m.group(2),
                       subsets=[s.strip().strip("'") for s in m.group(3).split(",") if s.strip()],
                       epochs=[], final=None, random=None,
                       train_minutes=None, train_loss=None,
                       datalen=None, readback=None)
            runs.append(cur)
            continue

        if cur is None:
            continue

        m = PROBE_RE.match(body)
        if m and probe is None:
            probe = dict(micro_batch=int(m.group(1)), peak_gib=float(m.group(2)),
                         ms_per_sample=float(m.group(3)))
            continue

        m = DATALEN_RE.match(body)
        if m:
            cur["datalen"] = dict(effective_batch=int(m.group(3)), steps_per_epoch=int(m.group(4)),
                                  max_steps=int(m.group(5)), warmup=int(m.group(6)))
            continue

        m = READBACK_RE.match(body)
        if m:
            cur["readback"] = dict(effective_batch=int(m.group(1)), steps_per_epoch=int(m.group(2)),
                                   max_steps=int(m.group(3)), warmup=int(m.group(4)))
            continue

        m = TRAIN_RE.match(body)
        if m:
            cur["train_minutes"] = float(m.group(1))
            cur["train_loss"] = float(m.group(2))
            continue

        m = RAND_RE.match(body)
        if m:
            cur["random"] = dict(m1=float(m.group(1)), m2=float(m.group(2)), m3=float(m.group(3)))
            continue

        m = EVAL_RE.match(body)
        if m:
            kv = dict(KV_RE.findall(m.group(1)))
            ep = EPOCH_RE.search(m.group(1))
            epoch = int(ep.group(1)) if ep else None
            rec = dict(epoch=epoch)
            for k, v in kv.items():
                if k.startswith("eval_"):
                    setting, short = split_metric_key(k[len("eval_"):])
                    if short is not None:
                        rec[setting + "|" + short] = float(v)
                    elif setting.startswith("合计|"):
                        rec[setting] = v


            if epoch is not None and "合计|m1" in rec:
                cur["epochs"].append(rec)
            elif epoch is not None and "eval_loss" in kv:
                cur.setdefault("final_eval", rec)
            continue


        if body.startswith("en-nn|") or body.startswith("de-nn|") or body.startswith("en-pv|") \
           or body.startswith("de-pv|") or body.startswith("合计|"):
            parts = body.split()
            if len(parts) >= 2:
                setting, short = split_metric_key(parts[0])
                try:
                    val = float(parts[-1])
                except ValueError:
                    continue
                if cur["final"] is None:
                    cur["final"] = {}
                cur["final"][setting + ("|" + short if short else "")] = val

    return dict(model=path.stem, probe=probe, runs=runs)


def main():
    if not LOG_DIR.exists():
        sys.exit(f"日志目录不存在：{LOG_DIR}")
    out = {}
    for name, fname in [("deberta", "compartment-deberta-baseline.log"),
                        ("qwen", "compartment-qwen-instruct-baseline.log")]:
        p = LOG_DIR / fname
        if not p.exists():
            print(f"⚠ 缺日志：{p}")
            continue
        out[name] = parse_log(p)

    OUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写出 {OUT_JSON}")

    for model, data in out.items():
        print(f"\n{'='*74}\n{model}   探针峰值显存 {data['probe']['peak_gib']} GiB / "
              f"{data['probe']['ms_per_sample']} ms 每样本\n{'='*74}")
        for r in data["runs"]:
            dl = r.get("datalen") or {}
            rb = r.get("readback") or {}
            print(f"\n[{r['idx']}/5] {r['name']}  子集={r['subsets']}")
            print(f"  训练 {r['train_minutes']} 分钟 | loss {r['train_loss']} | "
                  f"每轮批数 {dl.get('steps_per_epoch')} | 总步 {dl.get('max_steps')} | "
                  f"warmup {dl.get('warmup')}（回读 {rb.get('warmup')}）")
            print("  逐 epoch 指标1 / 指标2 / RMSE:")
            for e in r["epochs"]:
                t1 = e.get("合计|m1"); t2 = e.get("合计|m2"); t3 = e.get("合计|m3")
                print(f"    epoch {e['epoch']}: m1={t1}  m2={t2}  rmse={t3}")
            f = r.get("final") or {}
            print(f"  ★ 最终: m1={f.get("合计|m1")}  m2={f.get("合计|m2")}  "
                  f"rmse={f.get("合计|m3")}  综合分={f.get("合计|综合分")}")
            rnd = r.get("random") or {}
            print(f"  随机初始化: m1={rnd.get('m1')}  m2={rnd.get('m2')}  rmse={rnd.get('m3')}")


if __name__ == "__main__":
    main()
