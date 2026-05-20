"""标准多关系 R-GCN：每关系一套 GraphConv，按和聚合，残差 + dropout。

也是 Camo R-GCN 的对照模型。在 YelpChi/Amazon 满标注 + 100/10/5/1% 低标注下
跑全套对比。
"""
from datetime import datetime

import lightgbm as lgb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

import dgl
from dgl.nn import GraphConv, HeteroGraphConv

import config
from load_data import LOADERS, load_dataset
from utils import risk_metrics

HIDDEN_DIM = 64
N_LAYERS = 2
DROPOUT = 0.3
EPOCHS = 300
LR = 0.01
WEIGHT_DECAY = 5e-4
PATIENCE = 30
SEED = 42

LABEL_FRACTIONS = [1.0, 0.1, 0.05, 0.01]
TOPK_RATIOS = (0.01, 0.05, 0.10)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    dgl.seed(seed)


class RGCN(nn.Module):
    def __init__(self, in_dim, hidden, canonical_etypes):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden)
        self.layers = nn.ModuleList([
            HeteroGraphConv(
                {et: GraphConv(hidden, hidden, norm="right",
                               allow_zero_in_degree=True)
                 for et in canonical_etypes},
                aggregate="sum")
            for _ in range(N_LAYERS)
        ])
        self.classifier = nn.Linear(hidden, 2)

    def forward(self, g, feat):
        ntype = g.ntypes[0]
        h = {ntype: self.input_proj(feat)}
        for i, layer in enumerate(self.layers):
            h_in = h
            h = layer(g, h)
            # HeteroGraphConv 不含自环，残差补回节点自身
            h = {k: v + h_in[k] for k, v in h.items()}
            if i != len(self.layers) - 1:
                h = {k: F.dropout(F.relu(v), DROPOUT, self.training)
                     for k, v in h.items()}
        return self.classifier(h[ntype])

    def embed(self, g, feat):
        ntype = g.ntypes[0]
        h = {ntype: self.input_proj(feat)}
        for i, layer in enumerate(self.layers):
            h_in = h
            h = layer(g, h)
            h = {k: v + h_in[k] for k, v in h.items()}
            if i != len(self.layers) - 1:
                h = {k: F.dropout(F.relu(v), DROPOUT, self.training)
                     for k, v in h.items()}
        return h[ntype]


@torch.no_grad()
def predict_proba(model, g, feat) -> np.ndarray:
    model.eval()
    out = model(g, feat)
    if isinstance(out, tuple):  # CamoRGCN 返回 (logits, score) — 已废弃，保留兼容
        out = out[0]
    return torch.softmax(out, dim=1)[:, 1].cpu().numpy()


def fit_rgcn(g, feat, y_np, train_mask, val_mask, device):
    model = RGCN(feat.shape[1], HIDDEN_DIM, g.canonical_etypes).to(device)
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


def train_lgbm(X, y, train_mask, val_mask) -> np.ndarray:
    spw = float((y[train_mask] == 0).sum()) / max(
        float((y[train_mask] == 1).sum()), 1.0)
    model = lgb.LGBMClassifier(
        objective="binary", metric="auc", n_estimators=2000,
        learning_rate=0.05, num_leaves=31, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, scale_pos_weight=spw, random_state=SEED,
        n_jobs=-1, verbose=-1)
    model.fit(X[train_mask], y[train_mask],
              eval_set=[(X[val_mask], y[val_mask])], eval_metric="auc",
              callbacks=[lgb.early_stopping(100, first_metric_only=True),
                         lgb.log_evaluation(0)])
    return model.predict_proba(X)[:, 1]


def subsample_train(train_mask, y, fraction, seed) -> np.ndarray:
    """对训练 mask 做分层下采样，返回缩小后的布尔掩码。"""
    if fraction >= 1.0:
        return train_mask.copy()
    rng = np.random.RandomState(seed)
    idx = np.where(train_mask)[0]
    keep = []
    for cls in (0, 1):
        cls_idx = idx[y[idx] == cls]
        n_keep = max(1, int(round(len(cls_idx) * fraction)))
        keep.append(rng.choice(cls_idx, n_keep, replace=False))
    mask = np.zeros_like(train_mask)
    mask[np.concatenate(keep)] = True
    return mask


def run_dataset(name: str, device: str) -> dict:
    g = load_dataset(name)
    X = g.ndata["feature"].numpy().astype(np.float32)
    y = g.ndata["label"].numpy().astype(np.int64)
    train_mask = g.ndata["train_mask"].numpy().astype(bool)
    val_mask = g.ndata["val_mask"].numpy().astype(bool)
    test_mask = g.ndata["test_mask"].numpy().astype(bool)

    feat_std = StandardScaler().fit_transform(X).astype(np.float32)
    g = g.to(device)
    feat = torch.tensor(feat_std, device=device)

    rows = []
    for frac in LABEL_FRACTIONS:
        set_seed(SEED)
        tr = subsample_train(train_mask, y, frac, SEED)
        lgb_proba = train_lgbm(X, y, tr, val_mask)
        gnn = fit_rgcn(g, feat, y, tr, val_mask, device)
        gnn_proba = predict_proba(gnn, g, feat)

        lgb_m = risk_metrics(y[test_mask], lgb_proba[test_mask], TOPK_RATIOS)
        gnn_m = risk_metrics(y[test_mask], gnn_proba[test_mask], TOPK_RATIOS)
        rows.append({"frac": frac, "n_label": int(tr.sum()),
                     "lgb": lgb_m, "gnn": gnn_m})
        print(f"  [{name}] {frac:>5.0%}  LGB={lgb_m['auc']:.4f}  "
              f"R-GCN={gnn_m['auc']:.4f}  Δ={gnn_m['auc'] - lgb_m['auc']:+.4f}")

        if frac == 1.0:
            torch.save(gnn.state_dict(), config.MODELS_DIR / f"rgcn_{name}.pt")

    return {"name": name, "rows": rows}


def write_report(results: list) -> None:
    lines = [
        "# 标准 R-GCN（满标注 + 低标注）",
        f"\n生成时间: {datetime.now():%Y-%m-%d %H:%M:%S}\n",
    ]
    for res in results:
        lines += [f"## {res['name']}",
                  "\n| 标签量 | 模型 | AUC | PR-AUC | KS | recall@1% | recall@5% |",
                  "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
        for row in res["rows"]:
            tag = f"{row['frac']:.0%} ({row['n_label']:,})"
            for mdl in ("lgb", "gnn"):
                m = row[mdl]
                lines.append(
                    f"| {tag if mdl == 'lgb' else ''} | "
                    f"{'LightGBM' if mdl == 'lgb' else 'R-GCN'} | "
                    f"{m['auc']:.4f} | {m['pr_auc']:.4f} | {m['ks']:.4f} | "
                    f"{m['recall@1%']:.4f} | {m['recall@5%']:.4f} |")
        lines.append("")
    (config.REPORTS_DIR / "rgcn_report.md").write_text(
        "\n".join(lines), encoding="utf-8")


def main():
    set_seed(SEED)
    config.ensure_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}")
    results = [run_dataset(name, device) for name in LOADERS]
    write_report(results)


if __name__ == "__main__":
    main()
