"""YelpChi / Amazon 数据加载。首次调用从 data.dgl.ai 自动下载到 data/raw/。"""
import dgl

import config

LOADERS = {
    "yelp": dgl.data.FraudYelpDataset,
    "amazon": dgl.data.FraudAmazonDataset,
}


def load_dataset(name: str):
    key = name.lower()
    if key not in LOADERS:
        raise ValueError(f"未知数据集 '{name}', 可选: {list(LOADERS)}")
    return LOADERS[key](raw_dir=str(config.RAW_DIR))[0]


if __name__ == "__main__":
    config.ensure_dirs()
    for name in LOADERS:
        g = load_dataset(name)
        label = g.ndata["label"]
        n_pos = int((label == 1).sum())
        print(f"[{name}] 节点 {g.num_nodes():,}  关系 "
              f"{[e[1] for e in g.canonical_etypes]}  "
              f"作弊 {n_pos}/{g.num_nodes()} ({n_pos / g.num_nodes():.2%})")
