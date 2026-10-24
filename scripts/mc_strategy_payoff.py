import numpy as np, json
from pathlib import Path
from scipy import stats as st

ROOT = Path(__file__).resolve().parents[1]
rng = np.random.RandomState(11)
N_MC = 800
PAR = {
    "en-nn/ModAvg":  dict(nt=444, pt=8,  tau2=0.0581, sig2=0.2418, oracle_m1=0.790),
    "en-nn/HeadAvg": dict(nt=444, pt=8,  tau2=0.0745, sig2=0.2146, oracle_m1=0.771),
    "de-nn/ModAvg":  dict(nt=461, pt=7,  tau2=0.0,   sig2=0.1264, oracle_m1=0.780),
    "de-nn/HeadAvg": dict(nt=461, pt=7,  tau2=0.0,   sig2=0.1783, oracle_m1=0.918),
    "en-pv/Avg":     dict(nt=158, pt=10, tau2=0.3821, sig2=0.2508, oracle_m1=0.702),
    "de-pv/Avg":     dict(nt=158, pt=9,  tau2=0.2102, sig2=0.2382, oracle_m1=0.798),
}
STRATS = ["S0 抄金标", "S1 完美组内", "S2 仅组均值", "S3 +独立噪声",
          "S4 收缩λ=0.5", "S4b 收缩λ=0.2", "S5 抖动c=0.3"]

def metrics(p, g):
    m1 = st.spearmanr(p.ravel(), g.ravel()).statistic
    ps = p.std(axis=1, ddof=1); gs = g.std(axis=1, ddof=1)
    ok = (ps > 0) & (gs > 0)
    m2 = float(st.spearmanr(ps[ok], gs[ok]).statistic) if ok.sum() >= 5 else float("nan")
    return float(m1), m2

res = {}
for name, pr in PAR.items():
    acc = {s: {"m1": [], "m2": []} for s in STRATS}
    for _ in range(N_MC):
        nt, pt = pr["nt"], pr["pt"]
        tau, sig = np.sqrt(pr["tau2"]), np.sqrt(pr["sig2"])
        mu = rng.normal(3.5, 1.0, nt)
        s = rng.normal(0, tau, (nt, pt))
        eta = rng.normal(0, sig, (nt, pt))
        eta2 = rng.normal(0, sig, (nt, pt))
        g = mu[:, None] + s + eta
        z = rng.normal(0, 1, (nt, pt))
        cands = {
            "S0 抄金标": g,
            "S1 完美组内": mu[:, None] + s,
            "S2 仅组均值": np.repeat(mu[:, None], pt, axis=1),
            "S3 +独立噪声": mu[:, None] + s + eta2,
            "S4 收缩λ=0.5": mu[:, None] + 0.5 * s + 0.5 * eta2,
            "S4b 收缩λ=0.2": mu[:, None] + 0.2 * s + 0.8 * eta2,
            "S5 抖动c=0.3": mu[:, None] + 0.3 * z * np.sqrt(pr["tau2"] + pr["sig2"]),
        }
        for k in STRATS:
            m1, m2 = metrics(cands[k], g)
            acc[k]["m1"].append(m1); acc[k]["m2"].append(m2)
    res[name] = {k: dict(m1=round(float(np.nanmean(v["m1"])), 4),
                         m2=round(float(np.nanmean(v["m2"])), 4)) for k, v in acc.items()}
    got = res[name]["S2 仅组均值"]["m1"]; want = pr["oracle_m1"]
    d_noise = res[name]["S3 +独立噪声"]["m1"] - res[name]["S1 完美组内"]["m1"]
    print(f"=== {name} === 校准 S2_m1={got:.4f} vs 实测 oracle {want:.3f} (差 {abs(got-want):.3f}) | "
          f"制造离散度对m1的代价={d_noise:+.4f}")
    for k in STRATS:
        print(f"   {k:16s} m1={res[name][k]['m1']:+.4f}   m2={res[name][k]['m2']:+.4f}")
Path(ROOT / "scripts" / "mc_strategy_payoff.json").write_text(
    json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
print("\nwrote scripts/mc_strategy_payoff.json (v2)")
