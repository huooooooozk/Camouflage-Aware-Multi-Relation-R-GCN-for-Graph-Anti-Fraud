"""Camo R-GCN 的可解释性：关系注意力 / 伪装边抑制 / 节点级 / 特征显著性。"""
import sys
from datetime import datetime

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

import config
from load_data import load_dataset
from train_camo_rgcn import CamoRGCN
from train_rgcn import HIDDEN_DIM, predict_proba

DATASET = "yelp"
MODEL_PATH = config.MODELS_DIR / "rgcn_camo_yelp.pt"
REPORT_PATH = config.REPORTS_DIR / "explain_report.md"

SUPPRESS_THRESHOLD = 0.5
N_CASES = 6
N_TOP_FEATURES = 10


def load_model(device):
    if not MODEL_PATH.exists():
        sys.exit(f"缺少主模型 {MODEL_PATH}，请先 python src/train_camo_rgcn.py")
    g = load_dataset(DATASET)
    label = g.ndata["label"].numpy().astype(int)
    X = g.ndata["feature"].numpy().astype(np.float32)
    feat = torch.tensor(StandardScaler().fit_transform(X).astype(np.float32),
                        device=device)
    rel_names = [et[1] for et in g.canonical_etypes]
    model = CamoRGCN(X.shape[1], HIDDEN_DIM, rel_names).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return g.to(device), feat, label, model, rel_names


def relation_attention(model, rel_names) -> list:
    out = []
    for layer in model.layers:
        attn = torch.softmax(layer.rel_attn.detach(), dim=0).cpu().numpy()
        out.append(dict(zip(rel_names, attn.tolist())))
    return out


@torch.no_grad()
def edge_weight_analysis(g, model, feat, label) -> dict:
    ew = model._edge_weights(g, feat)
    result = {}
    for et in g.canonical_etypes:
        rel = et[1]
        src, dst = g.edges(etype=et)
        src, dst = src.cpu().numpy(), dst.cpu().numpy()
        w = ew[rel].squeeze(-1).cpu().numpy()
        same = label[src] == label[dst]
        cross = ~same
        result[rel] = {
            "same_label_w": float(w[same].mean()) if same.any() else 0.0,
            "cross_label_w": float(w[cross].mean()) if cross.any() else 0.0,
            "cross_ratio": float(cross.mean()),
        }
    return result


@torch.no_grad()
def node_explanations(g, model, feat, label, proba, node_ids) -> list:
    ew = model._edge_weights(g, feat)
    rel_edges = {}
    for et in g.canonical_etypes:
        src, dst = g.edges(etype=et)
        rel_edges[et[1]] = (src.cpu().numpy(), dst.cpu().numpy(),
                            ew[et[1]].squeeze(-1).cpu().numpy())

    cases = []
    for v in node_ids:
        n_nb, n_supp = 0, 0
        for src, dst, w in rel_edges.values():
            inc = dst == v
            n_nb += int(inc.sum())
            n_supp += int((w[inc] < SUPPRESS_THRESHOLD).sum())
        cases.append({"node": int(v), "risk": float(proba[v]),
                      "label": int(label[v]), "n_neighbor": n_nb,
                      "n_suppressed": n_supp,
                      "supp_ratio": n_supp / n_nb if n_nb else 0.0})
    return cases


def feature_saliency(g, model, feat, high_mask) -> list:
    feat_g = feat.clone().detach().requires_grad_(True)
    logits = model(g, feat_g)
    mask = torch.tensor(high_mask, device=feat.device)
    logits[mask, 1].sum().backward()
    sal = feat_g.grad.abs().mean(dim=0).cpu().numpy()
    order = np.argsort(sal)[::-1][:N_TOP_FEATURES]
    return [(int(i), float(sal[i])) for i in order]


def write_report(rel_attn, edge_stats, cases, saliency, n_high, n_total):
    lines = [
        "# 可解释性（Camo R-GCN, YelpChi）",
        f"\n生成时间: {datetime.now():%Y-%m-%d %H:%M:%S}\n",
        "## 关系注意力（softmax 后的每层关系权重）\n",
        "| 层 | net_rsr | net_rtr | net_rur |",
        "| --- | ---: | ---: | ---: |",
    ]
    for i, attn in enumerate(rel_attn, 1):
        lines.append(f"| 第 {i} 层 | {attn['net_rsr']:.3f} | "
                     f"{attn['net_rtr']:.3f} | {attn['net_rur']:.3f} |")
    lines += [
        "",
        "## 伪装边抑制",
        "\n边权 = exp(β·(cos(x_u, x_v) − 1))。跨标签边即疑似伪装边。\n",
        "| 关系 | 同标签边权 | 跨标签边权 | 比值 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for rel, s in edge_stats.items():
        ratio = s["cross_label_w"] / s["same_label_w"] if s["same_label_w"] else 0.0
        lines.append(f"| {rel} | {s['same_label_w']:.4f} | "
                     f"{s['cross_label_w']:.4f} | {ratio:.2f} |")
    lines += [
        "",
        "## 节点级解释（风险最高的 6 个节点）\n",
        "| node | risk | 真实 | 邻居 | 抑制邻居 | 抑制占比 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for c in cases:
        lbl = "作弊" if c["label"] == 1 else "正常"
        lines.append(f"| {c['node']} | {c['risk']:.4f} | {lbl} | "
                     f"{c['n_neighbor']:,} | {c['n_suppressed']:,} | "
                     f"{c['supp_ratio']:.2%} |")
    lines += [
        "",
        "## 特征显著性（高风险节点的作弊 logit 梯度幅值 Top-10）\n",
        "| 排名 | 特征下标 | 平均梯度 |",
        "| --- | ---: | ---: |",
    ]
    for rank, (idx, val) in enumerate(saliency, 1):
        lines.append(f"| {rank} | #{idx} | {val:.4f} |")
    lines.append(f"\n高风险节点共 {n_high:,} / {n_total:,}。")
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main():
    config.ensure_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    g, feat, label, model, rel_names = load_model(device)
    proba = predict_proba(model, g, feat)
    high_mask = proba >= 0.5
    n_high, n_total = int(high_mask.sum()), len(label)

    rel_attn = relation_attention(model, rel_names)
    edge_stats = edge_weight_analysis(g, model, feat, label)
    cases = node_explanations(g, model, feat, label, proba,
                              np.argsort(proba)[::-1][:N_CASES])
    saliency = feature_saliency(g, model, feat, high_mask)

    write_report(rel_attn, edge_stats, cases, saliency, n_high, n_total)
    print(f"report -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
