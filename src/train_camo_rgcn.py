"""Camo R-GCN（主模型）：标签无关的伪装边抑制。

边权 = exp(β·(cos(x_u, x_v) − 1))，跨群体（疑似伪装）边权趋近 0。
聚合用每关系一套无 bias 线性变换 + 加权平均，关系间用 softmax 注意力组合。
不依赖标签信号，1% 标注下仍工作。
"""
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

import dgl.function as fn

import config
from load_data import load_dataset
from train_rgcn import (DROPOUT, EPOCHS, HIDDEN_DIM, LR, N_LAYERS, PATIENCE,
                        WEIGHT_DECAY, predict_proba, set_seed, subsample_train)
from utils import risk_metrics

DATASET = "yelp"
SEEDS = [42, 43, 44, 45, 46]
LABEL_FRACTIONS = [0.01, 0.05, 0.10, 1.0]
TOPK_RATIOS = (0.01, 0.03, 0.05, 0.10, 0.15, 0.20)
CAMO_BETA = 3.0

METRIC_KEYS = ["auc", "pr_auc", "ks", "recall@10%", "recall@15%", "recall@20%"]


class WeightedRelConv(nn.Module):
    """每关系一套线性变换 + 加权平均聚合 + softmax 关系注意力。"""

    def __init__(self, in_dim, out_dim, rel_names):
        super().__init__()
        self.rel_names = rel_names
        self.weight = nn.ModuleDict(
            {r: nn.Linear(in_dim, out_dim, bias=False) for r in rel_names})
        self.rel_attn = nn.Parameter(torch.zeros(len(rel_names)))

    def forward(self, g, h, edge_weight):
        ntype = g.ntypes[0]
        attn = torch.softmax(self.rel_attn, dim=0)
        out = 0
        for i, r in enumerate(self.rel_names):
            with g.local_scope():
                g.nodes[ntype].data["h"] = self.weight[r](h)
                g.edges[r].data["w"] = edge_weight[r]
                g.update_all(fn.u_mul_e("h", "w", "m"), fn.sum("m", "agg"),
                             etype=r)
                g.update_all(fn.copy_e("w", "mw"), fn.sum("mw", "wsum"),
                             etype=r)
                agg = g.nodes[ntype].data["agg"]
                wsum = g.nodes[ntype].data["wsum"].clamp(min=1e-6)
                out = out + attn[i] * (agg / wsum)
        return out


class CamoRGCN(nn.Module):
    def __init__(self, in_dim, hidden, rel_names, beta=CAMO_BETA):
        super().__init__()
        self.rel_names = rel_names
        self.beta = beta
        self.input_proj = nn.Linear(in_dim, hidden)
        self.layers = nn.ModuleList(
            [WeightedRelConv(hidden, hidden, rel_names)
             for _ in range(N_LAYERS)])
        self.classifier = nn.Linear(hidden, 2)

    def _edge_weights(self, g, feat):
        fn_ = F.normalize(feat, dim=1)
        ew = {}
        for et in g.canonical_etypes:
            src, dst = g.edges(etype=et)
            sim = (fn_[src] * fn_[dst]).sum(dim=1)
            ew[et[1]] = torch.exp(self.beta * (sim - 1.0)).unsqueeze(-1)
        return ew

    def forward(self, g, feat):
        ew = self._edge_weights(g, feat)
        h = self.input_proj(feat)
        for i, layer in enumerate(self.layers):
            h = layer(g, h, ew) + h
            if i != len(self.layers) - 1:
                h = F.dropout(F.relu(h), DROPOUT, self.training)
        return self.classifier(h)


def fit_camo(g, feat, y_np, train_mask, val_mask, device):
    rel_names = [et[1] for et in g.canonical_etypes]
    model = CamoRGCN(feat.shape[1], HIDDEN_DIM, rel_names).to(device)
    y_tensor = torch.tensor(y_np, device=device)
    tr = torch.tensor(train_mask, device=device)

    n_pos = int(y_np[train_mask].sum())
    n_neg = int(train_mask.sum()) - n_pos
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, n_neg / max(n_pos, 1)], device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR,
                                 weight_decay=WEIGHT_DECAY)

    best_auc, best_state, wait = 0.0, None, 0
    for _ in range(EPOCHS):
        model.train()
        logits = model(g, feat)
        loss = criterion(logits[tr], y_tensor[tr])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        proba = predict_proba(model, g, feat)
        val_auc = roc_auc_score(y_np[val_mask], proba[val_mask])
        if val_auc > best_auc:
            best_auc, wait = val_auc, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            wait += 1
        if wait >= PATIENCE:
            break

    model.load_state_dict(best_state)
    return model


def train_and_save(dataset: str = DATASET, seed: int = 42) -> None:
    """训练满标注主模型并保存到 outputs/models/rgcn_camo_<dataset>.pt。"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    g = load_dataset(dataset)
    X = g.ndata["feature"].numpy().astype(np.float32)
    y = g.ndata["label"].numpy().astype(np.int64)
    train_mask = g.ndata["train_mask"].numpy().astype(bool)
    val_mask = g.ndata["val_mask"].numpy().astype(bool)
    feat = torch.tensor(
        StandardScaler().fit_transform(X).astype(np.float32), device=device)
    g = g.to(device)

    set_seed(seed)
    model = fit_camo(g, feat, y, train_mask, val_mask, device)
    path = config.MODELS_DIR / f"rgcn_camo_{dataset}.pt"
    torch.save(model.state_dict(), path)
    print(f"主模型已保存: {path}")


def run_low_label(dataset: str = DATASET) -> pd.DataFrame:
    """5 种子 × 4 标注比例，写 data/processed/camo_raw.csv。"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    g = load_dataset(dataset)
    X = g.ndata["feature"].numpy().astype(np.float32)
    y = g.ndata["label"].numpy().astype(np.int64)
    masks = {m: g.ndata[f"{m}_mask"].numpy().astype(bool)
             for m in ("train", "val", "test")}
    feat = torch.tensor(
        StandardScaler().fit_transform(X).astype(np.float32), device=device)
    g = g.to(device)

    records = []
    for si, seed in enumerate(SEEDS, 1):
        for frac in LABEL_FRACTIONS:
            tr = subsample_train(masks["train"], y, frac, seed)
            set_seed(seed)
            model = fit_camo(g, feat, y, tr, masks["val"], device)
            proba = predict_proba(model, g, feat)
            m = risk_metrics(y[masks["test"]], proba[masks["test"]], TOPK_RATIOS)
            records.append({"seed": seed, "frac": frac, "model": "camo", **m})
            print(f"  [seed {si}/{len(SEEDS)}={seed}] {frac:.0%}  "
                  f"AUC={m['auc']:.4f}")
        pd.DataFrame(records).to_csv(
            config.PROCESSED_DIR / "camo_raw.csv", index=False, encoding="utf-8")
    return pd.DataFrame(records)


def main():
    config.ensure_dirs()
    # 1) 满标注训一份主模型保存，供下游 detect_communities / explain / pipeline 用
    train_and_save()
    # 2) 5 种子 × 4 标注比例完整对比
    df = run_low_label()
    agg = df.groupby("frac")[METRIC_KEYS].agg(["mean", "std"])
    lines = [
        f"# Camo R-GCN 多种子（{DATASET}）",
        f"\n生成时间: {datetime.now():%Y-%m-%d %H:%M:%S}　种子: {SEEDS}\n",
        "| 标注 | " + " | ".join(METRIC_KEYS) + " |",
        "| --- " * (len(METRIC_KEYS) + 1) + "|",
    ]
    for frac in LABEL_FRACTIONS:
        cells = [f"{agg.loc[frac, (k, 'mean')]:.4f} ± "
                 f"{agg.loc[frac, (k, 'std')]:.4f}" for k in METRIC_KEYS]
        lines.append(f"| {frac:.0%} | " + " | ".join(cells) + " |")
    (config.REPORTS_DIR / "camo_report.md").write_text(
        "\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
