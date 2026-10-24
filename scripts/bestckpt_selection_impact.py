import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "scripts" / "bestckpt_selection_impact.json"
DATA = json.loads((ROOT / "scripts" / "three_gen_compare.json").read_text(encoding="utf-8"))


THR = {"m1": 0.065, "m2": 0.153, "m3": 0.046}


def score(rec, rule):
    m1 = rec.get("合计|m1")
    m2 = rec.get("合计|m2")
    m3 = rec.get("合计|m3")
    if m1 is None:
        return None
    if rule == "m1":
        return m1
    if rule == "m1+m2":
        c2 = m2 if (m2 is not None and m2 == m2) else -1.0
        return m1 + c2
    if rule == "m1+m2-m3/5":
        c2 = m2 if (m2 is not None and m2 == m2) else -1.0
        return m1 + c2 - (m3 or 0.0) / 5.0
    if rule == "z":

        m2v = m2 if (m2 is not None and m2 == m2) else -1.0
        return m1 / 0.023 + m2v / 0.055 - (m3 or 0.0) / 0.046
    raise ValueError(rule)


RULES = ["m1", "m1+m2", "m1+m2-m3/5", "z"]


def main():
    result = {}
    print("=" * 108)
    print("每代日志：用四种选择准则重算「会选中哪个 epoch、交出什么指标」")
    print("（实际交出 = 日志里的『训练后指标』；epoch 0 是随机初始化，不参与竞争）")
    print("=" * 108)

    for gen in ("v1", "v2", "v3"):
        for model in ("deberta", "qwen"):
            if gen not in DATA or model not in DATA.get(gen, {}):
                continue
            print(f"\n【{gen} · {model}】")
            for r in DATA[gen][model]:

                curve = [e for e in r["curve"] if e["epoch"] >= 1]

                seen, uniq = set(), []
                for e in curve:
                    if e["epoch"] in seen:
                        continue
                    seen.add(e["epoch"])
                    uniq.append(e)
                curve = uniq
                actual = (r.get("final") or {}).get("合计|m1")
                line = f"  {r['name']:16s} 实际交出 m1={actual if actual is None else round(actual,4)}"
                picks = {}
                for rule in RULES:
                    scored = [(score(e, rule), e) for e in curve]
                    scored = [(s, e) for s, e in scored if s is not None]
                    if not scored:
                        continue
                    best_s, best_e = max(scored, key=lambda t: t[0])
                    picks[rule] = dict(epoch=best_e["epoch"], m1=best_e.get("合计|m1"),
                                       m2=best_e.get("合计|m2"), m3=best_e.get("合计|m3"))
                result.setdefault(gen, {}).setdefault(model, {})[r["name"]] = dict(
                    actual_m1=actual, picks=picks)
                for rule in RULES:
                    p = picks.get(rule)
                    if not p:
                        continue
                    mark = ""
                    if actual is not None and p["m1"] is not None:
                        d = p["m1"] - actual
                        mark = "  ←与实际一致" if abs(d) < 1e-6 else f"  ⚠与实际差 {d:+.4f}"
                    print(f"      {rule:16s} → epoch {p['epoch']}  m1={p['m1']:8.4f}{mark}")

    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写出 {OUT}")

    print("\n" + "=" * 108)
    print("★ 关键结论：现行准则（m1+m2）比「只看指标1」多交出了多少指标1")
    print("=" * 108)
    print(f"{'代':4s} {'基线':9s} {'运行':16s} {'现行(m1+m2)':>13s} {'只看m1':>10s} {'差':>9s} {'阈值':>7s}")
    for gen in ("v1", "v2", "v3"):
        for model in ("deberta", "qwen"):
            for nm, d in (result.get(gen, {}).get(model, {}) or {}).items():
                cur = d["picks"].get("m1+m2")
                only = d["picks"].get("m1")
                if not cur or not only:
                    continue
                delta = cur["m1"] - only["m1"]
                flag = "★现行准则吃掉了更好的 checkpoint" if delta < -THR["m1"] else ""
                print(f"{gen:4s} {model:9s} {nm:16s} {cur['m1']:13.4f} {only['m1']:10.4f} "
                      f"{delta:+9.4f} {THR['m1']:7.3f}  {flag}")


if __name__ == "__main__":
    main()
