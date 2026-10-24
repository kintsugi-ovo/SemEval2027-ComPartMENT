import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "logs" / "v3" / "round3"
OUT_JSON = ROOT / "scripts" / "v3_crosscheck.json"


THRESH = {"m1": 0.065, "m2": 0.153, "m3": 0.046}


SETTING_LINE = re.compile(r"^(\w+)\s+=\s+(.+?)\s*(?:←.*)?$")
SECTION_MUST = "【必须与另一条基线逐字相同】"
SECTION_MAY = "【允许不同（各有明确理由，不影响可比性）】"


def grab_settings(path: Path):
    must, may = {}, {}
    mode = None
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        body = raw.split(" ", 2)[-1].rstrip() if re.match(r"^\s*[\d.]+s\s+\d+\s", raw) else raw.rstrip()
        if SECTION_MUST in body:
            mode = "must"; continue
        if SECTION_MAY in body:
            mode = "may"; continue
        if mode is None:
            continue
        m = SETTING_LINE.match(body)
        if not m:
            if body.strip() == "":
                continue
            mode = None
            continue
        key, val = m.group(1), m.group(2).strip()
        (must if mode == "must" else may)[key] = val
    return must, may


def main():
    data = json.loads((ROOT / "scripts" / "v3_run_logs.json").read_text(encoding="utf-8"))


    settings = {}
    for model, fname in [("deberta", "compartment-deberta-baseline.log"),
                         ("qwen", "compartment-qwen-instruct-baseline.log")]:
        p = LOG_DIR / fname
        if p.exists():
            settings[model] = grab_settings(p)

    diff_report = {}
    if len(settings) == 2:
        d, q = settings["deberta"][0], settings["qwen"][0]
        must_keys = sorted(set(d) | set(q))
        must_diff = [(k, d.get(k), q.get(k))
                     for k in must_keys if d.get(k) != q.get(k)]
        dm, qm = settings["deberta"][1], settings["qwen"][1]
        may_keys = sorted(set(dm) | set(qm))
        may_diff = [(k, dm.get(k), qm.get(k))
                    for k in may_keys if dm.get(k) != qm.get(k)]
        diff_report = dict(
            must_total=len(must_keys), must_diff=must_diff,
            may_total=len(may_keys), may_diff=may_diff,
        )


    best_epochs = {}
    for model, d in data.items():
        for r in d["runs"]:


            f = r.get("final") or {}
            fm1 = f.get("合计|m1")
            cands = [e["epoch"] for e in r["epochs"]
                     if e.get("合计|m1") is not None and abs(e["合计|m1"] - fm1) < 1e-9]
            best_epochs.setdefault(model, {})[r["name"]] = dict(
                final_m1=fm1, final_m2=f.get("合计|m2"), final_m3=f.get("合计|m3"),
                best_epoch=cands[0] if cands else None,
                curve=[dict(epoch=e["epoch"], m1=e.get("合计|m1"), m2=e.get("合计|m2"),
                            m3=e.get("合计|m3")) for e in r["epochs"]],
            )





    v12 = {
        ("deberta", "single_en-nn"): (0.4659, 0.3446, 0.8210),
        ("deberta", "single_de-nn"): (0.4017, 0.2792, 0.7935),
        ("deberta", "single_en-pv"): (0.5574, 0.6538, 0.9229),
        ("deberta", "single_de-pv"): (0.0776, 0.3775, 1.0890),
        ("deberta", "joint"):         (0.3790, 0.3402, 0.9162),
        ("qwen", "single_en-nn"): (0.4692, 0.1925, 0.8423),
        ("qwen", "single_de-nn"): (0.5035, 0.2512, 0.7707),
        ("qwen", "single_en-pv"): (0.4989, 0.6574, 0.9525),
        ("qwen", "single_de-pv"): (0.3795, 0.3317, 1.2004),
        ("qwen", "joint"):         (0.4950, 0.3864, 0.8429),
    }

    rows = []
    for model, d in data.items():
        for r in d["runs"]:
            f = r.get("final") or {}
            rnd = r.get("random") or {}
            key = (model, r["name"])
            v3 = (f.get("合计|m1"), f.get("合计|m2"), f.get("合计|m3"))
            prev = v12.get(key)
            rec = dict(model=model, run=r["name"],
                       v3_m1=v3[0], v3_m2=v3[1], v3_m3=v3[2],
                       rnd_m1=rnd.get("m1"), rnd_m2=rnd.get("m2"), rnd_m3=rnd.get("m3"),
                       best_epoch=best_epochs[model][r["name"]]["best_epoch"],
                       train_minutes=r.get("train_minutes"),
                       max_steps=(r.get("datalen") or {}).get("max_steps"))
            if prev:
                rec.update(v1_m1=prev[0], v1_m2=prev[1], v1_m3=prev[2],
                           d_m1=round(v3[0] - prev[0], 4),
                           d_m2=round(v3[1] - prev[1], 4),
                           d_m3=round(v3[2] - prev[2], 4))

            rec["vs_random_d_m1"] = round(v3[0] - rnd.get("m1"), 4)
            rec["vs_random_d_m2"] = round(v3[1] - rnd.get("m2"), 4)
            rec["vs_random_d_m3"] = round(v3[2] - rnd.get("m3"), 4)
            rows.append(rec)

    out = dict(口径_diff=diff_report, best_epoch=best_epochs, runs=rows,
               thresholds=THRESH)
    OUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写出 {OUT_JSON}\n")


    print("=" * 96)
    print("① 口径一致性（25 项「必须逐字相同」+ 13 项「允许不同」）")
    print("=" * 96)
    if diff_report:
        print(f"「必须逐字相同」共 {diff_report['must_total']} 项，"
              f"不同项：{diff_report['must_diff'] if diff_report['must_diff'] else '**无** ✓'}")
        print(f"「允许不同」共 {diff_report['may_total']} 项，不同项：")
        for k, a, b in diff_report["may_diff"]:
            print(f"    {k:26s} deberta={a}   qwen={b}")
    else:
        print("（日志缺失，无法 diff）")

    print("\n" + "=" * 96)
    print("② best checkpoint 选中了第几个 epoch（最终指标 = 该 epoch 的回读）")
    print("=" * 96)
    for model in best_epochs:
        print(f"\n[{model}]")
        for name, b in best_epochs[model].items():
            curve = " → ".join(f"e{e['epoch']}:{e['m1']:.4f}" for e in b["curve"][:4])
            print(f"  {name:16s} best=epoch {b['best_epoch']}   m1 曲线 {curve}")

    print("\n" + "=" * 96)
    print("③ v1 → v3 对比（阈值：指标1 0.065 / 指标2 0.153 / RMSE 0.046）")
    print("=" * 96)
    hdr = f"{'基线':8s} {'运行':16s} {'v1 m1':>8s} {'v3 m1':>8s} {'Δm1':>8s} {'判定':>6s} | " \
          f"{'v1 m2':>8s} {'v3 m2':>8s} {'Δm2':>8s} {'判定':>6s} | {'vs随机Δm1':>10s}"
    print(hdr)
    for r in rows:
        def judge(d, t):
            return "✅显著" if abs(d) > t else "⚠噪声内"
        print(f"{r['model']:8s} {r['run']:16s} {r.get('v1_m1', float('nan')):8.4f} "
              f"{r['v3_m1']:8.4f} {r.get('d_m1', float('nan')):8.4f} "
              f"{judge(r.get('d_m1', 0), THRESH['m1']):>6s} | "
              f"{r.get('v1_m2', float('nan')):8.4f} {r['v3_m2']:8.4f} {r.get('d_m2', float('nan')):8.4f} "
              f"{judge(r.get('d_m2', 0), THRESH['m2']):>6s} | {r['vs_random_d_m1']:10.4f}")


if __name__ == "__main__":
    main()
