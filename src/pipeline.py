"""分级处置 + 监控评估。

按风险分把节点分到 高危拦截 / 中危复核 / 低危观察 / 放行 四档；
在 test 集上算各档精度与作弊覆盖率；可选 LLM 整体策略建议。
"""
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

import config
from llm_attribution import call_llm
from load_data import load_dataset
from train_camo_rgcn import CamoRGCN
from train_rgcn import HIDDEN_DIM, TOPK_RATIOS, predict_proba
from utils import risk_metrics

DATASET = "yelp"
MODEL_PATH = config.MODELS_DIR / "rgcn_camo_yelp.pt"
RINGS_PARQUET = config.PROCESSED_DIR / "spam_rings_yelp.parquet"
DISPOSITION_PARQUET = config.PROCESSED_DIR / "disposition_yelp.parquet"
REPORT_PATH = config.REPORTS_DIR / "pipeline_report.md"

# (档位, 动作, 下界, 上界)
TIERS = [
    ("高危拦截", "自动拦截 / 下架",       0.90, 1.01),
    ("中危复核", "进入人工复核队列",       0.70, 0.90),
    ("低危观察", "降权 / 持续观察",         0.50, 0.70),
    ("放行",     "正常放行",                0.00, 0.50),
]

LLM_SYSTEM = (
    "你是内容生态反作弊的策略负责人。下面是「图模型识别 + 团伙挖掘 + 分级处置」"
    "系统的监控面板。请用中文输出：\n"
    "【运营总结】3-4 句；\n"
    "【策略建议】2-3 条，针对阈值、人工复核投入、团伙批量处置。\n"
    "只基于给定数据，不要编造数字。"
)


def assign_tier(p: float) -> tuple:
    for name, action, lo, hi in TIERS:
        if lo <= p < hi:
            return name, action
    return TIERS[-1][0], TIERS[-1][1]


def score_nodes(g, device) -> np.ndarray:
    if not MODEL_PATH.exists():
        sys.exit(f"缺少主模型 {MODEL_PATH}，请先 python src/train_camo_rgcn.py")
    X = g.ndata["feature"].numpy().astype(np.float32)
    feat = torch.tensor(StandardScaler().fit_transform(X).astype(np.float32),
                        device=device)
    rel_names = [et[1] for et in g.canonical_etypes]
    model = CamoRGCN(X.shape[1], HIDDEN_DIM, rel_names).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    return predict_proba(model, g.to(device), feat)


def disposition_table(proba, label, test_mask) -> list:
    total_spam = int(label[test_mask].sum())
    rows = []
    for name, action, lo, hi in TIERS:
        in_tier = (proba >= lo) & (proba < hi) & test_mask
        n = int(in_tier.sum())
        n_spam = int(label[in_tier].sum())
        rows.append({"tier": name, "action": action, "n": n,
                     "precision": n_spam / n if n else 0.0,
                     "coverage": n_spam / total_spam if total_spam else 0.0})
    return rows


def ring_disposition() -> dict:
    if not RINGS_PARQUET.exists():
        return {}
    members = pd.read_parquet(RINGS_PARQUET)
    ring_risk = members.groupby("ring_id")["risk"].mean()
    counts = {name: 0 for name, *_ in TIERS}
    for r in ring_risk:
        counts[assign_tier(r)[0]] += 1
    return counts


def panel_text(det, disp_rows, ring_counts, n_test) -> str:
    lines = [
        f"检测层（test {n_test:,}）：AUC={det['auc']:.4f}  "
        f"PR-AUC={det['pr_auc']:.4f}  KS={det['ks']:.4f}",
        "分级处置：",
    ]
    for r in disp_rows:
        lines.append(f"  {r['tier']}（{r['action']}）：{r['n']} 条，"
                     f"精度 {r['precision']:.2%}，覆盖 {r['coverage']:.2%}")
    lines.append("团伙处置：")
    for tier, cnt in ring_counts.items():
        lines.append(f"  {tier}：{cnt} 个团伙")
    return "\n".join(lines)


def write_report(det, disp_rows, ring_counts, n_test, llm_text):
    intercept = next(r for r in disp_rows if r["tier"] == "高危拦截")
    review = next(r for r in disp_rows if r["tier"] == "中危复核")

    lines = [
        "# 处置 + 监控闭环",
        f"\n生成时间: {datetime.now():%Y-%m-%d %H:%M:%S}\n",
        "## 检测层（test 集）\n",
        "| 指标 | 数值 |", "| --- | ---: |",
        f"| AUC | {det['auc']:.4f} |",
        f"| PR-AUC | {det['pr_auc']:.4f} |",
        f"| KS | {det['ks']:.4f} |",
    ]
    for r in TOPK_RATIOS:
        lines.append(f"| recall@{int(r*100)}% | {det[f'recall@{int(r*100)}%']:.4f} |")
    lines += [
        "",
        "## 分级处置\n",
        "| 档位 | 动作 | n | 档内精度 | 作弊覆盖 |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for r in disp_rows:
        lines.append(f"| {r['tier']} | {r['action']} | {r['n']:,} | "
                     f"{r['precision']:.2%} | {r['coverage']:.2%} |")
    lines += [
        "",
        f"> 高危档自动处置覆盖 {intercept['coverage']:.1%} 作弊；"
        f"叠加中危复核 {review['n']:,} 条共覆盖 "
        f"{intercept['coverage'] + review['coverage']:.1%}。",
        "",
        "## 团伙处置（按平均风险归档）\n",
        "| 档位 | 团伙数 |", "| --- | ---: |",
    ]
    for tier, cnt in ring_counts.items():
        lines.append(f"| {tier} | {cnt} |")
    lines += ["", "## LLM 策略建议\n", llm_text, ""]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main():
    config.ensure_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    g = load_dataset(DATASET)
    label = g.ndata["label"].numpy().astype(int)
    test_mask = g.ndata["test_mask"].numpy().astype(bool)

    proba = score_nodes(g, device)
    det = risk_metrics(label[test_mask], proba[test_mask], TOPK_RATIOS)
    disp_rows = disposition_table(proba, label, test_mask)
    ring_counts = ring_disposition()
    n_test = int(test_mask.sum())

    tiers = [assign_tier(p) for p in proba]
    pd.DataFrame({
        "node_id": np.arange(len(proba)),
        "risk": proba,
        "tier": [t[0] for t in tiers],
        "action": [t[1] for t in tiers],
    }).to_parquet(DISPOSITION_PARQUET, index=False)

    panel = panel_text(det, disp_rows, ring_counts, n_test)
    env = config.load_dotenv()
    try:
        llm_text = call_llm(env["ANTHROPIC_API_KEY"],
                            env["ANTHROPIC_BASE_URL"], panel, LLM_SYSTEM)
    except Exception as e:  # noqa: BLE001
        llm_text = f"（LLM 失败：{e}）"
        print(f"LLM 失败: {e}")

    write_report(det, disp_rows, ring_counts, n_test, llm_text)
    for r in disp_rows:
        print(f"  {r['tier']}: {r['n']:,}  精度 {r['precision']:.2%}  "
              f"覆盖 {r['coverage']:.2%}")
    print(f"disposition -> {DISPOSITION_PARQUET}\nreport      -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
