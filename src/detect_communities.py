"""Louvain 团伙挖掘。

主模型给所有节点打分 -> 取高风险节点 + 两端均高风险的边构成子图 ->
Louvain 切社区 -> 规模 >= MIN_RING_SIZE 视为一个团伙。

net_rsr 等关系太稠密，直接取连通分量会塌成一个巨型簇，所以走 Louvain。
"""
import sys
from collections import defaultdict
from datetime import datetime

import community as community_louvain
import networkx as nx
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

import config
from load_data import load_dataset
from train_camo_rgcn import CamoRGCN
from train_rgcn import HIDDEN_DIM, predict_proba

DATASET = "yelp"
MODEL_PATH = config.MODELS_DIR / "rgcn_camo_yelp.pt"
RINGS_PARQUET = config.PROCESSED_DIR / "spam_rings_yelp.parquet"
REPORT_PATH = config.REPORTS_DIR / "community_report.md"

RISK_THRESHOLD = 0.5
MIN_RING_SIZE = 3
SEED = 42


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


def build_high_risk_graph(g, proba):
    high_mask = proba >= RISK_THRESHOLD
    high_nodes = np.where(high_mask)[0]

    G = nx.Graph()
    G.add_nodes_from(int(n) for n in high_nodes)
    rel_edges = {}
    for et in g.canonical_etypes:
        src, dst = g.edges(etype=et)
        src, dst = src.numpy(), dst.numpy()
        keep = high_mask[src] & high_mask[dst] & (src != dst)
        rel_edges[et[1]] = int(keep.sum())
        G.add_edges_from(np.stack([src[keep], dst[keep]], axis=1).tolist())
    return G, high_nodes, rel_edges


def extract_rings(G, proba, label):
    partition = community_louvain.best_partition(G, random_state=SEED)
    groups = defaultdict(list)
    for node, comm in partition.items():
        groups[comm].append(node)
    comps = sorted((v for v in groups.values() if len(v) >= MIN_RING_SIZE),
                   key=len, reverse=True)

    rings, members = [], []
    for ring_id, comp in enumerate(comps):
        nodes = sorted(int(n) for n in comp)
        rings.append({"ring_id": ring_id, "size": len(nodes),
                      "mean_risk": float(proba[nodes].mean()),
                      "spam_rate": float(label[nodes].mean())})
        for n in nodes:
            members.append({"node_id": n, "ring_id": ring_id,
                            "risk": float(proba[n]), "label": int(label[n])})
    return rings, pd.DataFrame(members)


def write_report(rings, members, n_high, rel_edges, n_total, base_rate):
    n_in_rings = len(members)
    ring_spam = members["label"].mean() if n_in_rings else 0.0
    sizes = sorted((r["size"] for r in rings), reverse=True)

    lines = [
        "# 团伙挖掘（YelpChi）",
        f"\n生成时间: {datetime.now():%Y-%m-%d %H:%M:%S}",
        f"风险阈值: {RISK_THRESHOLD}　最小团伙: {MIN_RING_SIZE}\n",
        f"- 全部节点: {n_total:,}",
        f"- 高风险节点: {n_high:,} ({n_high / n_total:.2%})",
        f"- 团伙数: {len(rings):,}　落入团伙的评论: {n_in_rings:,}",
        f"- 团伙成员真实作弊率: {ring_spam:.2%}　基准 {base_rate:.2%}　"
        f"提升 {(ring_spam / base_rate if base_rate else 0):.2f}x",
        f"- 最大团伙: {sizes[0] if sizes else 0}　中位规模: "
        f"{int(np.median(sizes)) if sizes else 0}",
        "",
        "## 高风险子图各关系边数\n",
        "| 关系 | 边数 |", "| --- | ---: |",
    ]
    for rel, n in rel_edges.items():
        lines.append(f"| {rel} | {n:,} |")
    lines += [
        "",
        "## 风险最高的 10 个团伙\n",
        "| ring_id | size | mean_risk | spam_rate |",
        "| --- | ---: | ---: | ---: |",
    ]
    for r in sorted(rings, key=lambda r: r["mean_risk"], reverse=True)[:10]:
        lines.append(f"| #{r['ring_id']} | {r['size']:,} | "
                     f"{r['mean_risk']:.4f} | {r['spam_rate']:.2%} |")
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main():
    config.ensure_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    g = load_dataset(DATASET)
    label = g.ndata["label"].numpy().astype(int)
    n_total = g.num_nodes()
    base_rate = float(label.mean())

    print("打分 ...")
    proba = score_nodes(g, device)
    n_high = int((proba >= RISK_THRESHOLD).sum())
    print(f"高风险节点: {n_high:,} / {n_total:,}")

    print("建子图 + Louvain ...")
    G, _, rel_edges = build_high_risk_graph(g, proba)
    rings, members = extract_rings(G, proba, label)
    members.to_parquet(RINGS_PARQUET, index=False)

    write_report(rings, members, n_high, rel_edges, n_total, base_rate)
    ring_spam = members["label"].mean() if len(members) else 0.0
    print(f"团伙数 {len(rings):,}　成员作弊率 {ring_spam:.2%}　基准 {base_rate:.2%}")
    print(f"members -> {RINGS_PARQUET}\nreport  -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
