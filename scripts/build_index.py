"""把 bullet_corpus.jsonl 向量化 → 存成 numpy 索引。

用法：
    python scripts/build_index.py

输出：
    data/bullet_index.npz
        - embeddings: np.ndarray (N, 1536)  float32，已做 L2 归一化，检索时直接点积=cosine
        - meta: np.ndarray (N,) object   每条 bullet 的元数据 dict

为什么不用 ChromaDB / FAISS：
    当前语料 ~75 条，numpy 点积检索延迟 < 1ms，零依赖，索引文件可 git 追踪。
    规模扩展路径：1K+ 切 FAISS，10万+ 才上向量数据库。
"""
import json
import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "bullet_corpus.jsonl"
INDEX = ROOT / "data" / "bullet_index.npz"

EMBED_MODEL = "text-embedding-3-small"  # 1536 维，$0.02/1M tokens


def load_corpus() -> list[dict]:
    items = []
    with open(CORPUS, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def embed_batch(client: OpenAI, texts: list[str]) -> np.ndarray:
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
    # L2 归一化 → 后续点积即 cosine similarity
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


def main():
    client = OpenAI()
    items = load_corpus()
    print(f"📚 加载语料：{len(items)} 条 bullet")

    # 拼接"role_tag + skill_tags + bullet"作为 embedding 输入，提升同域召回
    texts = [
        f"[{it['role_tag']}] {' / '.join(it.get('skill_tags', []))} :: {it['bullet']}"
        for it in items
    ]

    print(f"🔗 调用 {EMBED_MODEL} 生成向量...")
    vecs = embed_batch(client, texts)
    print(f"✅ 向量形状：{vecs.shape}")

    np.savez(INDEX, embeddings=vecs, meta=np.array(items, dtype=object))
    print(f"💾 写入索引：{INDEX}  ({INDEX.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
