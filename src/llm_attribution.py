"""LLM 团伙归因：把团伙画像交给 LLM 生成自然语言风险说明。

走 Anthropic 兼容端点。配置从 .env 读 ANTHROPIC_API_KEY / ANTHROPIC_BASE_URL，
默认配 deepseek-chat（DeepSeek 的兼容端点），换成 Claude 把 MODEL 改一下即可。
"""
import json
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import requests

import config
from load_data import load_dataset

DATASET = "yelp"
RINGS_PARQUET = config.PROCESSED_DIR / "spam_rings_yelp.parquet"
ANALYSIS_JSON = config.PROCESSED_DIR / "ring_llm_analysis.json"
REPORT_PATH = config.REPORTS_DIR / "llm_attribution_report.md"

MODEL = "deepseek-chat"
TOP_N_RINGS = 15
MAX_TOKENS = 800
REQUEST_TIMEOUT = 60

REL_MEANING = {
    "net_rsr": "同商户且同星级的评论相连",
    "net_rtr": "同商户且同月的评论相连",
    "net_rur": "同一用户发布的评论相连",
}

SYSTEM_PROMPT = (
    "你是内容生态反作弊的风险分析师。下面给你一个由图神经网络识别、"
    "并经社区发现聚成的「疑似水军团伙」的结构画像。\n"
    "三种关系的含义：net_rsr=同商户且同星级的评论相连；"
    "net_rtr=同商户且同月的评论相连；net_rur=同一用户发布的评论相连。\n"
    "基于画像输出三部分，用中文：\n"
    "【风险归因】2-4 句，说明该团伙为何可疑；\n"
    "【团伙摘要】一句话概括作弊模式；\n"
    "【处置建议】1-2 句，给出处置动作建议。\n"
    "只基于给定画像，不要编造商户名或用户名。"
)


def build_ring_profiles() -> list:
    if not RINGS_PARQUET.exists():
        sys.exit(f"缺少团伙文件 {RINGS_PARQUET}，请先 python src/detect_communities.py")

    members = pd.read_parquet(RINGS_PARQUET)
    g = load_dataset(DATASET)
    n_nodes = g.num_nodes()

    ring_of = np.full(n_nodes, -1, dtype=np.int64)
    ring_of[members["node_id"].to_numpy()] = members["ring_id"].to_numpy()

    rel_internal = {}
    for et in g.canonical_etypes:
        src, dst = g.edges(etype=et)
        src, dst = src.numpy(), dst.numpy()
        both = (ring_of[src] >= 0) & (ring_of[src] == ring_of[dst])
        rel_internal[et[1]] = np.bincount(
            ring_of[src][both],
            minlength=int(members["ring_id"].max()) + 1)

    profiles = []
    for ring_id, grp in members.groupby("ring_id"):
        rel_edges = {rel: int(rel_internal[rel][ring_id]) // 2
                     for rel in REL_MEANING}
        profiles.append({
            "ring_id": int(ring_id),
            "size": int(len(grp)),
            "mean_risk": float(grp["risk"].mean()),
            "spam_rate": float(grp["label"].mean()),
            "rel_edges": rel_edges,
            "dominant_rel": max(rel_edges, key=rel_edges.get),
        })
    profiles.sort(key=lambda p: p["mean_risk"], reverse=True)
    return profiles


def build_user_prompt(p: dict) -> str:
    rel = p["rel_edges"]
    return (
        f"团伙 #{p['ring_id']} 结构画像：\n"
        f"- 评论数：{p['size']}\n"
        f"- 图模型平均风险分：{p['mean_risk']:.3f}\n"
        f"- 团伙内部关系边：net_rsr {rel['net_rsr']} / "
        f"net_rtr {rel['net_rtr']} / net_rur {rel['net_rur']}\n"
        f"- 主导关系：{p['dominant_rel']}（{REL_MEANING[p['dominant_rel']]}）"
    )


def call_llm(api_key: str, base_url: str, user_prompt: str,
             system: str) -> str:
    """走 Anthropic 兼容端点的 /v1/messages。"""
    url = base_url.rstrip("/") + "/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "system": system,
        "messages": [{"role": "user", "content": user_prompt}],
    }
    last_err = None
    for _ in range(2):
        try:
            resp = requests.post(url, headers=headers, json=payload,
                                 timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            return resp.json()["content"][0]["text"].strip()
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(2)
    raise RuntimeError(f"LLM 调用失败: {last_err}")


def write_report(results: list) -> None:
    lines = [
        "# LLM 团伙归因",
        f"\n生成时间: {datetime.now():%Y-%m-%d %H:%M:%S}",
        f"模型: {MODEL}　归因团伙数: {len(results)}\n",
    ]
    for r in results:
        p = r["profile"]
        rel = p["rel_edges"]
        lines += [
            f"## 团伙 #{p['ring_id']}",
            f"\n评论数 {p['size']}　平均风险 {p['mean_risk']:.3f}　"
            f"net_rsr/net_rtr/net_rur = "
            f"{rel['net_rsr']}/{rel['net_rtr']}/{rel['net_rur']}　"
            f"主导 {p['dominant_rel']}\n",
            r["llm_output"], "",
        ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main():
    config.ensure_dirs()
    env = config.load_dotenv()
    api_key = env.get("ANTHROPIC_API_KEY")
    base_url = env.get("ANTHROPIC_BASE_URL")
    if not api_key or not base_url:
        sys.exit(".env 缺少 ANTHROPIC_API_KEY 或 ANTHROPIC_BASE_URL")

    profiles = build_ring_profiles()
    top = profiles[:TOP_N_RINGS]
    print(f"团伙总数 {len(profiles)}，对前 {len(top)} 个做归因")

    results = []
    for i, p in enumerate(top, 1):
        prompt = build_user_prompt(p)
        out = call_llm(api_key, base_url, prompt, SYSTEM_PROMPT)
        results.append({"profile": p, "llm_output": out})
        print(f"  [{i}/{len(top)}] #{p['ring_id']} done")

    write_report(results)
    ANALYSIS_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2),
                             encoding="utf-8")
    print(f"json   -> {ANALYSIS_JSON}\nreport -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
