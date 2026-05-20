"""YelpChi 上的 5 种子 × 4 标注比例三方对比（LightGBM / 标准 R-GCN / Camo R-GCN）。

随机性来源：训练标签下采样、GNN 权重初始化、LightGBM bagging。
train/val/test 用 DGL 内置 mask，固定不变。
"""
from datetime import datetime

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

import config
from load_data import load_dataset
from train_camo_rgcn import fit_camo
from train_rgcn import fit_rgcn, predict_proba, set_seed, subsample_train
from utils import risk_metrics

DATASET = "yelp"
SEEDS = [42, 43, 44, 45, 46]
LABEL_FRACTIONS = [0.01, 0.05, 0.10, 1.0]
TOPK_RATIOS = (0.01, 0.03, 0.05, 0.10, 0.15, 0.20)
METRIC_KEYS = ["auc", "pr_auc", "ks", "recall@1%", "recall@3%",
               "recall@5%", "recall@10%", "recall@15%", "recall@20%"]
MODELS = [("lgb", "LightGBM"), ("rgcn", "标准 R-GCN"), ("camo", "Camo R-GCN")]

RAW_CSV = config.PROCESSED_DIR / "multi_seed_raw.csv"
REPORT_PATH = config.REPORTS_DIR / "multi_seed_report.md"


def train_lgbm_seed(X, y, tr, va, seed) -> np.ndarray:
    spw = float((y[tr] == 0).sum()) / max(float((y[tr] == 1).sum()), 1.0)
    model = lgb.LGBMClassifier(
        objective="binary", metric="auc", n_estimators=2000,
        learning_rate=0.05, num_leaves=31, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, scale_pos_weight=spw, random_state=seed,
        n_jobs=-1, verbose=-1)
    model.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], eval_metric="auc",
              callbacks=[lgb.early_stopping(100, first_metric_only=True),
                         lgb.log_evaluation(0)])
    return model.predict_proba(X)[:, 1]


def run_one(g, feat, X, y, masks, frac, seed, device) -> list:
    tr = subsample_train(masks["train"], y, frac, seed)
    va, te = masks["val"], masks["test"]

    lgb_proba = train_lgbm_seed(X, y, tr, va, seed)
    set_seed(seed)
    rgcn_proba = predict_proba(fit_rgcn(g, feat, y, tr, va, device), g, feat)
    set_seed(seed)
    camo_proba = predict_proba(fit_camo(g, feat, y, tr, va, device), g, feat)

    return [
        {"seed": seed, "frac": frac, "model": "lgb",
         **risk_metrics(y[te], lgb_proba[te], TOPK_RATIOS)},
        {"seed": seed, "frac": frac, "model": "rgcn",
         **risk_metrics(y[te], rgcn_proba[te], TOPK_RATIOS)},
        {"seed": seed, "frac": frac, "model": "camo",
         **risk_metrics(y[te], camo_proba[te], TOPK_RATIOS)},
    ]


def write_report(df: pd.DataFrame) -> None:
    agg = df.groupby(["frac", "model"])[METRIC_KEYS].agg(["mean", "std"])
    name_map = dict(MODELS)

    lines = [
        "# 多种子稳健性（YelpChi）",
        f"\n生成时间: {datetime.now():%Y-%m-%d %H:%M:%S}　种子: {SEEDS}\n",
        "单元格为 `均值 ± 标准差`（test 集）\n",
    ]
    for frac in LABEL_FRACTIONS:
        lines += [
            f"## {frac:.0%} 标注",
            "\n| 模型 | " + " | ".join(METRIC_KEYS) + " |",
            "| --- " * (len(METRIC_KEYS) + 1) + "|",
        ]
        for key, cn in MODELS:
            cells = [f"{agg.loc[(frac, key), (k, 'mean')]:.4f} ± "
                     f"{agg.loc[(frac, key), (k, 'std')]:.4f}"
                     for k in METRIC_KEYS]
            lines.append(f"| {cn} | " + " | ".join(cells) + " |")
        lines.append("")

    # 1% 标注下 Camo R-GCN 较 LightGBM 的 AUC 增量
    sub_camo = df[(df["frac"] == 0.01) & (df["model"] == "camo")]["auc"]
    sub_lgb = df[(df["frac"] == 0.01) & (df["model"] == "lgb")]["auc"]
    diff = sub_camo.values - sub_lgb.values
    lines += [
        "## 结论\n",
        f"- 1% 标注下，Camo R-GCN 较 LightGBM AUC 增量: "
        f"**{diff.mean():+.4f} ± {diff.std():.4f}**"
        + ("（各种子均为正）" if (diff > 0).all() else "（存在种子为负）"),
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main():
    config.ensure_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    g = load_dataset(DATASET)
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
            records += run_one(g, feat, X, y, masks, frac, seed, device)
            print(f"  [seed {si}/{len(SEEDS)}={seed}] {frac:.0%} done")
        pd.DataFrame(records).to_csv(RAW_CSV, index=False, encoding="utf-8")

    write_report(pd.DataFrame(records))
    print(f"raw -> {RAW_CSV}\nreport -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
