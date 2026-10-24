import numpy as np, pandas as pd, json
from pathlib import Path
from scipy import stats as st

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "train"
rng = np.random.RandomState(0)
N_MC = 4000

def simulate(n_targets, per_target, tau, sigma, a=1.0, b=0.0, c=0.0):
    out = []
    for _ in range(N_MC):
        mu = rng.normal(0, 1.0, n_targets)
        eps = rng.normal(0, tau, (n_targets, per_target))
        eta = rng.normal(0, sigma, (n_targets, per_target))
        g = mu[:, None] + eps + eta
        p = mu[:, None] + a * eps + c * rng.normal(0, 1.0, (n_targets, per_target))
        ps = p.std(axis=1, ddof=1); gs = g.std(axis=1, ddof=1)
        ok = (ps > 0) & (gs > 0)
        if ok.sum() < 5:
            continue
        out.append(st.spearmanr(ps[ok], gs[ok]).statistic)
    return float(np.nanmean(out)), float(np.nanstd(out))


PARAMS = {
    "en-nn/ModAvg":  dict(nt=444, pt=8,  tau2=0.0581, sig2=0.2418),
    "en-nn/HeadAvg": dict(nt=444, pt=8,  tau2=0.0745, sig2=0.2146),
    "de-nn/ModAvg":  dict(nt=461, pt=7,  tau2=0.0,   sig2=0.1264),
    "de-nn/HeadAvg": dict(nt=461, pt=7,  tau2=0.0,   sig2=0.1783),
    "en-pv/Avg":     dict(nt=158, pt=10, tau2=0.3821, sig2=0.2508),
    "de-pv/Avg":     dict(nt=158, pt=9,  tau2=0.2102, sig2=0.2382),
}
res = {}
print(f"{'设置':18s} {'τ²(真语境)':>12s} {'σ²(噪声)':>10s} {'信噪比':>8s} "
      f"{'完美系统m2':>12s} {'现实系统m2':>12s}")
for name, pr in PARAMS.items():
    tau = np.sqrt(pr["tau2"]); sig = np.sqrt(pr["sig2"])
    snr = pr["tau2"] / (pr["tau2"] + pr["sig2"])
    m2_perfect, sd1 = simulate(pr["nt"], pr["pt"], tau, sig, a=1.0, c=0.0)

    m2_real, sd2 = simulate(pr["nt"], pr["pt"], tau, sig, a=0.4, c=0.15)
    res[name] = dict(tau2=pr["tau2"], sigma2=pr["sig2"], snr=round(float(snr), 4),
                      m2_perfect=round(m2_perfect, 4), m2_perfect_sd=round(sd1, 4),
                      m2_realistic=round(m2_real, 4), m2_realistic_sd=round(sd2, 4))
    print(f"{name:18s} {pr['tau2']:12.4f} {pr['sig2']:10.4f} {snr:8.3f} "
          f"{m2_perfect:12.4f} {m2_real:12.4f}")

Path(ROOT / "scripts" / "metric2_ceiling.json").write_text(
    json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
print("\nwrote scripts/metric2_ceiling.json")
