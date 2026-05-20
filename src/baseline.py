"""LightGBM 基线，仅用节点特征。

注意：开 scale_pos_weight 后 binary_logloss 不再可比，早停必须只看 AUC
（first_metric_only=True），否则会被早停拉去优化跟实际表现脱钩的指标。
"""
from datetime import datetime

import lightgbm as lgb

import config
from load_data import LOADERS, load_dataset
from utils import risk_metrics

SEED = 42
TOPK_RATIOS = (0.01, 0.05, 0.10)


def train_one(name: str) -> dict:
    g = load_dataset(name)
    X = g.ndata["feature"].numpy()
    y = g.ndata["label"].numpy().astype(int)
    tr = g.ndata["train_mask"].numpy().astype(bool)
    va = g.ndata["val_mask"].numpy().astype(bool)
    te = g.ndata["test_mask"].numpy().astype(bool)

    spw = float((y[tr] == 0).sum()) / max(float((y[tr] == 1).sum()), 1.0)
    model = lgb.LGBMClassifier(
        objective="binary", metric="auc", n_estimators=2000,
        learning_rate=0.05, num_leaves=31, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, scale_pos_weight=spw, random_state=SEED,
        n_jobs=-1, verbose=-1,
    )
    model.fit(
        X[tr], y[tr], eval_set=[(X[va], y[va])], eval_metric="auc",
        callbacks=[lgb.early_stopping(100, first_metric_only=True),
                   lgb.log_evaluation(0)],
    )
    proba = model.predict_proba(X[te])[:, 1]
    m = risk_metrics(y[te], proba, TOPK_RATIOS)
    model.booster_.save_model(
        str(config.MODELS_DIR / f"baseline_{name}.txt"),
        num_iteration=model.best_iteration_)
    print(f"[{name}] best_iter={model.best_iteration_}  "
          f"AUC={m['auc']:.4f}  PR-AUC={m['pr_auc']:.4f}  KS={m['ks']:.4f}")
    return {"name": name, "metrics": m, "best_iter": model.best_iteration_,
            "n_train": int(tr.sum()), "n_test": int(te.sum())}


def write_report(results: list) -> None:
    keys = ["auc", "pr_auc", "ks"] + [f"recall@{int(r*100)}%" for r in TOPK_RATIOS]
    lines = [
        "# LightGBM 基线",
        f"\n生成时间: {datetime.now():%Y-%m-%d %H:%M:%S}\n",
    ]
    for r in results:
        m = r["metrics"]
        lines += [
            f"## {r['name']}",
            f"\ntrain/test = {r['n_train']:,}/{r['n_test']:,}　best_iter = {r['best_iter']}\n",
            "| 指标 | 数值 |", "| --- | ---: |",
        ]
        for k in keys:
            lines.append(f"| {k} | {m[k]:.4f} |")
        lines.append("")
    (config.REPORTS_DIR / "baseline_report.md").write_text(
        "\n".join(lines), encoding="utf-8")


def main():
    config.ensure_dirs()
    results = [train_one(name) for name in LOADERS]
    write_report(results)


if __name__ == "__main__":
    main()
