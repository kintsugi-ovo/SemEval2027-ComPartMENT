import pandas as pd, numpy as np, collections, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "train"
SPEC = {
    "en-nn": dict(key="Compound", labs=["ModAvg", "HeadAvg"], stds=["ModStd", "HeadStd"], task="nn"),
    "de-nn": dict(key="Compound", labs=["ModAvg", "HeadAvg"], stds=["ModStd", "HeadStd"], task="nn"),
    "en-pv": dict(key="ParticleVerb", labs=["Avg"], stds=["Std"], task="pv"),
    "de-pv": dict(key="ParticleVerb", labs=["Avg"], stds=["Std"], task="pv"),
}
out = {}
for name, sp in SPEC.items():
    df = pd.read_csv(DATA / f"{name}-train.tsv", sep="\t")
    g = df.groupby(sp["key"])
    r = dict(rows=len(df), targets=df[sp["key"]].nunique(),
             rows_per_target=round(len(df) / df[sp["key"]].nunique(), 2))

    r["label_const_within_target"] = {c: bool(g[c].nunique().max() == 1) for c in sp["labs"]}
    r["std_const_within_target"] = {c: bool(g[c].nunique().max() == 1) for c in sp["stds"]}

    r["label_nunique_max"] = {c: int(g[c].nunique().max()) for c in sp["labs"]}

    if sp["task"] == "nn":
        exact, lower_head, fugen = [], [], collections.Counter()
        for _, row in df.iterrows():
            comp, mod, head = str(row["Compound"]), str(row["Mod"]), str(row["Head"])
            exact.append(comp == mod + head)
            lower_head.append(comp.lower() == (mod + head[0].lower() + head[1:]).lower())
            if comp.lower() != (mod + head[0].lower() + head[1:]).lower():
                m = re.match(re.escape(mod.lower()) + r"(.+?)" + re.escape(head.lower()) + "$", comp.lower())
                fugen[m.group(1) if m else "??"] += 1
        r["compound_eq_mod_plus_head"] = round(np.mean(exact), 4)
        r["compound_eq_mod_lowerhead"] = round(np.mean(lower_head), 4)
        r["fugen_top"] = fugen.most_common(8)

        r["head_is_suffix_of_compound"] = round(float(np.mean([
            str(row["Compound"]).lower().endswith(str(row["Head"]).lower()) for _, row in df.iterrows()])), 4)
        r["mod_is_prefix_of_compound"] = round(float(np.mean([
            str(row["Compound"]).lower().startswith(str(row["Mod"]).lower()) for _, row in df.iterrows()])), 4)
        r["mod_example"] = [str(x) for x in df["Mod"].drop_duplicates().head(6)]
        r["head_example"] = [str(x) for x in df["Head"].drop_duplicates().head(6)]
    else:
        r["pv_eq_base_plus_particle"] = round(float(np.mean([
            str(row["ParticleVerb"]).lower() == (str(row["Base"]) + str(row["Particle"])).lower()
            for _, row in df.iterrows()])), 4)
        r["pv_eq_particle_plus_base"] = round(float(np.mean([
            str(row["ParticleVerb"]).lower() == (str(row["Particle"]) + str(row["Base"])).lower()
            for _, row in df.iterrows()])), 4)
        r["particle_is_prefix"] = round(float(np.mean([
            str(row["ParticleVerb"]).lower().startswith(str(row["Particle"]).lower())
            for _, row in df.iterrows()])), 4)
        r["base_example"] = [str(x) for x in df["Base"].drop_duplicates().head(6)]
        r["particle_example"] = [str(x) for x in df["Particle"].drop_duplicates().head(8)]
    out[name] = r
    print(f"=== {name} ===")
    for k, v in r.items():
        print("  ", k, "=", v)

Path(ROOT / "scripts" / "input_gap.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print("\nwrote scripts/input_gap.json")
