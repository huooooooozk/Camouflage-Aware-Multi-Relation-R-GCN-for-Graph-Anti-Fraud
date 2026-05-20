"""风控指标：AUC / PR-AUC / KS / Recall@TopK，统一口径供所有模型调用。"""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def topk_recall(y_true, y_score, top_ratio: float = 0.01) -> float:
    """按分数取前 top_ratio 比例，返回命中正例占全体正例的比。"""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    total_pos = int((y_true == 1).sum())
    if total_pos == 0:
        return 0.0
    k = max(1, int(np.ceil(len(y_score) * top_ratio)))
    top = np.argsort(y_score)[::-1][:k]
    return int((y_true[top] == 1).sum()) / total_pos


def ks_statistic(y_true, y_score) -> float:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if len(np.unique(y_true)) < 2:
        return 0.0
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


def risk_metrics(y_true, y_score, topk_ratios=(0.01, 0.05, 0.10)) -> dict:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    out = {
        "auc": roc_auc_score(y_true, y_score),
        "pr_auc": average_precision_score(y_true, y_score),
        "ks": ks_statistic(y_true, y_score),
    }
    for r in topk_ratios:
        out[f"recall@{int(r * 100)}%"] = topk_recall(y_true, y_score, r)
    return out
