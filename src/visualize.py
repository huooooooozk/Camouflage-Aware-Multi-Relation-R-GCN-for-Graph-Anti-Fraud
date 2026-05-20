"""三张主图：低标注交叉、单个团伙子图、分级处置分布。

数值取自 multi_seed 与 pipeline 的最终结果，按需更新。
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

import config
from load_data import load_dataset

RINGS_PARQUET = config.PROCESSED_DIR / "spam_rings_yelp.parquet"

# YelpChi，5 种子均值
FRAC_LABELS = ["1%", "5%", "10%", "100%"]
AUC_LGB = [0.7404, 0.8517, 0.8864, 0.9647]
AUC_RGCN = [0.7996, 0.8521, 0.8723, 0.9278]
AUC_CAMO = [0.8176, 0.8651, 0.8849, 0.9426]

# YelpChi test 集分级（来自 pipeline_report）
TIERS = [
    ("High-risk / auto-block",   932, 0.8262, 0.5923, "#d73027"),
    ("Mid-risk / manual review", 716, 0.3729, 0.2054, "#fc8d59"),
    ("Low-risk / monitor",       597, 0.1910, 0.0877, "#fee090"),
    ("Pass",                    6947, 0.0214, 0.1146, "#91cf60"),
]


def plot_crossover(path):
    x = np.arange(len(FRAC_LABELS))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(x, AUC_LGB, "o-", color="#555555", label="LightGBM (tree baseline)")
    ax.plot(x, AUC_RGCN, "s--", color="#4575b4", label="Vanilla R-GCN")
    ax.plot(x, AUC_CAMO, "^-", color="#d73027",
            label="Camouflage-aware R-GCN", linewidth=2)
    ax.set_xticks(x)
    ax.set_xticklabels(FRAC_LABELS)
    ax.set_xlabel("Training label fraction")
    ax.set_ylabel("Test AUC")
    ax.set_title("Detection AUC vs Label Fraction (YelpChi)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    ax.annotate("GNN > tree model\n(scarce labels)",
                xy=(0, AUC_CAMO[0]), xytext=(0.3, 0.83),
                fontsize=9, color="#d73027",
                arrowprops=dict(arrowstyle="->", color="#d73027"))
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_spam_ring(path):
    g = load_dataset("yelp")
    label = g.ndata["label"].numpy()
    members = pd.read_parquet(RINGS_PARQUET)

    sizes = members.groupby("ring_id").size()
    risk = members.groupby("ring_id")["risk"].mean()
    cand = sizes[(sizes >= 12) & (sizes <= 35)].index
    chosen = int(risk[cand].idxmax())
    nodes = members[members["ring_id"] == chosen]["node_id"].tolist()
    nodes_arr = np.array(nodes)

    G = nx.Graph()
    G.add_nodes_from(nodes)
    rel_color = {"net_rsr": "#e41a1c", "net_rtr": "#377eb8",
                 "net_rur": "#4daf4a"}
    rel_edges = {}
    for et in g.canonical_etypes:
        s, d = g.edges(etype=et)
        s, d = s.numpy(), d.numpy()
        m = np.isin(s, nodes_arr) & np.isin(d, nodes_arr) & (s < d)
        es = list(zip(s[m].tolist(), d[m].tolist()))
        rel_edges[et[1]] = es
        G.add_edges_from(es)

    pos = nx.spring_layout(G, seed=42, k=0.6)
    fig, ax = plt.subplots(figsize=(7, 6))
    for rel, es in rel_edges.items():
        if es:
            nx.draw_networkx_edges(G, pos, edgelist=es, ax=ax,
                                   edge_color=rel_color[rel], alpha=0.5,
                                   width=1.2)
    spam = [n for n in nodes if label[n] == 1]
    normal = [n for n in nodes if label[n] == 0]
    nx.draw_networkx_nodes(G, pos, nodelist=spam, ax=ax,
                           node_color="#d73027", node_size=130,
                           label="spam review")
    nx.draw_networkx_nodes(G, pos, nodelist=normal, ax=ax,
                           node_color="#4575b4", node_size=130,
                           label="normal review")
    for rel, color in rel_color.items():
        ax.plot([], [], color=color, label=rel)
    spam_rate = label[nodes_arr].mean()
    ax.set_title(f"Detected Spam Ring #{chosen}  "
                 f"(size={len(nodes)}, spam rate={spam_rate:.0%})")
    ax.legend(loc="upper right", fontsize=8)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_disposition(path):
    labels = [t[0] for t in TIERS]
    counts = [t[1] for t in TIERS]
    colors = [t[4] for t in TIERS]
    y = np.arange(len(TIERS))[::-1]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(y, counts, color=colors, edgecolor="#333333")
    for yi, t in zip(y, TIERS):
        _, n, prec, cov, _ = t
        ax.text(n + 80, yi,
                f"n={n}   precision={prec:.0%}   fraud-coverage={cov:.0%}",
                va="center", fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Number of reviews (test set)")
    ax.set_title("Risk-tiered Disposition (YelpChi test set)")
    ax.set_xlim(0, max(counts) * 2.1)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    config.ensure_dirs()
    plot_crossover(config.FIGURES_DIR / "lowlabel_crossover.png")
    plot_spam_ring(config.FIGURES_DIR / "spam_ring.png")
    plot_disposition(config.FIGURES_DIR / "disposition.png")
    print(f"figures -> {config.FIGURES_DIR}")


if __name__ == "__main__":
    main()
