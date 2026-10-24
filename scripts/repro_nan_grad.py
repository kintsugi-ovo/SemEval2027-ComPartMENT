import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

NB = ROOT / "ComPartMENT-CoMWE基线(DeBERTa-v2-xlarge-回归)/ComPartMENT基线_DeBERTa-v2-xlarge.ipynb"


import nbformat
from torch.utils.data import Sampler
from transformers import AutoModel, AutoTokenizer, Trainer

nb = nbformat.read(str(NB), as_version=4)
sources = [c.source for c in nb.cells]


def find_cell(marker):
    for s in sources:
        if marker in s:
            return s
    raise RuntimeError(f"找不到含 {marker!r} 的 cell")


NS = dict(
    torch=torch, np=np, pd=pd, F=F, nn=nn, Trainer=Trainer,
    SUBSET_ORDER=["en-nn", "de-nn", "en-pv", "de-pv"],
    SUBSET_LOSS_PROFILE={
        "en-nn": dict(order=0.08, mse=0.05, guard=0.06, xrank=0.00, kappa=0.15),
        "de-nn": dict(order=0.08, mse=0.05, guard=0.06, xrank=0.00, kappa=0.15),
        "en-pv": dict(order=0.08, mse=0.05, guard=0.06, xrank=0.06, kappa=0.50),
        "de-pv": dict(order=0.08, mse=0.05, guard=0.06, xrank=0.06, kappa=0.50),
    },
    N_OUTPUTS=2, HUBER_DELTA=1.0, ORDER_TAU=1.0,
    XRANK_MARGIN=0.05, XRANK_TAU=2.0, BALANCE_EMA_BETA=0.99, AUTO_BALANCE=True,
    SEED=42, Sampler=Sampler,
)
exec(compile(find_cell("def profile_vectors"), "<parts>", "exec"), NS)
exec(compile(find_cell("class GroupStats"), "<assembly>", "exec"), NS)

profile_vectors = NS["profile_vectors"]
order_parts = NS["order_parts"]
mse_parts = NS["mse_parts"]
guard_parts = NS["guard_parts"]
xrank_parts = NS["xrank_parts"]
std_from_moments = NS["std_from_moments"]
GroupStats = NS["GroupStats"]
N_SUB = 4



class M(nn.Module):
    def __init__(self, bb, hidden):
        super().__init__()
        self.backbone = bb
        self.head = nn.Linear(hidden, 2)

    def forward(self, input_ids, attention_mask):
        h = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        m = attention_mask.unsqueeze(-1).to(h.dtype)
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1.0)
        return {"logits": 5.0 * torch.sigmoid(self.head(pooled))}


def build(device):
    name = "microsoft/deberta-v3-base"
    tok = AutoTokenizer.from_pretrained(name)
    bb = AutoModel.from_pretrained(name, dtype=torch.float32)
    model = M(bb, bb.config.hidden_size).to(device=device, dtype=torch.float32)
    return tok, model


def make_batch(tok, subset="en-nn", n_targets=8, batch=8, device="cpu"):
    fn = {"en-nn": "en-nn-train.tsv", "de-nn": "de-nn-train.tsv",
          "en-pv": "en-pv-train.tsv", "de-pv": "de-pv-train.tsv"}[subset]
    df = pd.read_csv(ROOT / "data" / "train" / fn, sep="\t")
    df.columns = [c.strip() for c in df.columns]
    key = "Compound" if subset.endswith("nn") else "ParticleVerb"
    cols = ["ModAvg", "HeadAvg"] if subset.endswith("nn") else ["Avg"]

    rng = np.random.default_rng(42)
    keys = list(df[key].unique())
    rng.shuffle(keys)
    flat = []
    for k in keys:
        sub = df[df[key] == k]
        idx = list(range(len(sub)))
        rng.shuffle(idx)
        flat.extend([(k, i) for i in idx])

    take = []
    for k, i in flat:
        if len(take) >= batch:
            break
        take.append((k, i))
    rows = []
    for k, i in take:
        r = df[df[key] == k].iloc[i]
        rows.append(r)

    gkeys = [f"{subset}||{r[key]}" for r in rows]
    codes = {k: i for i, k in enumerate(sorted(set(gkeys)))}
    gidx = torch.tensor([codes[k] for k in gkeys], dtype=torch.long)
    sidx = torch.tensor([0 if subset == "en-nn" else (1 if subset == "de-nn" else (2 if subset == "en-pv" else 3))] * len(rows),
                         dtype=torch.long)
    label_rows = []
    for r in rows:
        row = [float(r[cols[0]])]
        if len(cols) > 1:
            row.append(float(r[cols[1]]))
        else:
            row.append(0.0)
        label_rows.append(row)
    labels = torch.tensor(label_rows, dtype=torch.float32)
    mask = torch.tensor([[1.0] + ([1.0] if len(cols) > 1 else [0.0])] * len(rows), dtype=torch.float32)
    text = [f"MWE: {r[key]} | Context: {r['Context']}" for r in rows]
    enc = tok(text, truncation=True, max_length=160, padding=True, return_tensors="pt")
    return enc, labels, mask, gidx, sidx, rows


def check_grad(name, loss, params, model):
    loss.backward(retain_graph=True)
    bad_nan, bad_inf, gmax = [], [], 0.0
    for n, p in params:
        if p.grad is None:
            continue
        if torch.isnan(p.grad).any():
            bad_nan.append(n)
        if torch.isinf(p.grad).any():
            bad_inf.append(n)
        gmax = max(gmax, float(p.grad.abs().max()))
    print(f"  {name:8s} loss={float(loss):+.6f} | NaN 参数数={len(bad_nan)} | Inf 参数数={len(bad_inf)} | max|g|={gmax:.3e}")
    if bad_nan:
        print(f"      → NaN 参数（前 5）: {bad_nan[:5]}")
    if bad_inf:
        print(f"      → Inf 参数（前 5）: {bad_inf[:5]}")
    model.zero_grad(set_to_none=True)
    return len(bad_nan), len(bad_inf)


def main():
    device = "cpu"
    torch.manual_seed(42)
    tok, model = build(device)
    model.train()
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]

    for subset in ["en-nn", "de-nn", "en-pv"]:
        print("=" * 90)
        print(f"【{subset}】第 1 个 batch（grouped sampler 顺序）")
        enc, labels, mask, gidx, sidx, rows = make_batch(tok, subset=subset, device=device)
        enc = {k: v.to(device) for k, v in enc.items()}
        print(f"  batch={len(rows)} | 目标组数={len(set(gidx.tolist()))} | group_index={gidx.tolist()}")
        enc = {k: v for k, v in enc.items() if k in ("input_ids", "attention_mask")}
        logits = model(**enc)["logits"]
        print(f"  预测值: {logits[:, 0].tolist()}")
        for j in range(2):
            v = mask[:, j]
            if v.sum() == 0:
                print(f"  列{j}: 无标签（跳过）")
                continue
            g = logits[:, j][v > 0]
            gg = labels[:, j][v > 0]
            print(f"  列{j}: 预测 std={float(g.std(unbiased=True)):.6f} | gold std={float(gg.std(unbiased=True)):.6f}")

        prof = profile_vectors(torch.device(device), torch.float32)

        per = F.huber_loss(logits, labels, delta=1.0, reduction="none")
        main_loss = (per * mask).sum() / mask.sum().clamp(min=1.0)
        print("  —— 逐项梯度体检 ——")
        check_grad("huber", main_loss, params, model)


        j = 0
        valid = mask[:, j]
        N, D = order_parts(logits[:, j], labels[:, j], gidx, sidx, 1.0, valid, prof["order"], N_SUB)
        tv = N / D.clamp(min=1.0)
        print(f"    L_order 分项值 = {tv.tolist()} | D = {D.tolist()}")
        check_grad("order", (tv * (D > 0).float()).sum(), params, model)


        N, D = mse_parts(logits[:, j], labels[:, j], valid, sidx, prof["mse"], N_SUB)
        tv = N / D.clamp(min=1.0)
        check_grad("mse", (tv * (D > 0).float()).sum(), params, model)


        n_groups = int(gidx.max()) + 1
        gs = GroupStats(n_groups, torch.device(device), torch.float32)
        cnt, s1, s2, gcnt, gs1, gs2 = gs.stats(logits[:, j], labels[:, j], gidx, valid)
        std_p = std_from_moments(cnt, s1, s2)
        std_g = std_from_moments(gcnt, gs1, gs2)
        gsub = torch.zeros(n_groups, dtype=torch.long)
        for gi, si in zip(gidx.tolist(), sidx.tolist()):
            gsub[gi] = si
        print(f"    cnt={cnt.tolist()} | std_p={std_p.tolist()} | std_g={std_g.tolist()}")
        N, D = guard_parts(std_p, std_g, cnt, gsub, prof["kappa"], N_SUB)
        tv = N / D.clamp(min=1.0)
        print(f"    L_guard 分项值 = {tv.tolist()} | D = {D.tolist()}")
        check_grad("guard", (tv * (D > 0).float()).sum(), params, model)


        hist = torch.zeros(n_groups, dtype=torch.bool)
        N, D = xrank_parts(std_p, std_g, torch.unique(gidx), gsub, prof["xrank"], 0.05, 2.0, hist, N_SUB)
        tv = N / D.clamp(min=1.0)
        print(f"    L_xrank 分项值 = {tv.tolist()} | D = {D.tolist()}")
        check_grad("xrank", (tv * (D > 0).float()).sum(), params, model)


if __name__ == "__main__":
    main()
