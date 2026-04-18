"""RAG 检索层：从 bullet 案例库召回 top-K 相似优质 bullet。

设计要点：
1. 单例索引（_SINGLETON），避免每次改写都 reload 474KB 的 npz
2. 查询 embedding 也做 L2 归一化，与库内向量点积 = cosine 相似度
3. role_filter：按岗位类别做硬过滤，避免"PM 岗匹到算法 bullet"的串味
4. 为什么不用 ChromaDB：75 条规模 numpy 点积 < 1ms，零依赖，详见 scripts/build_index.py
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
INDEX_PATH = ROOT / "data" / "bullet_index.npz"
EMBED_MODEL = "text-embedding-3-small"


class BulletRetriever:
    def __init__(self, index_path: Path = INDEX_PATH):
        data = np.load(index_path, allow_pickle=True)
        self.embeddings: np.ndarray = data["embeddings"]  # (N, 1536) 已 L2 归一
        self.meta: list[dict] = list(data["meta"])         # list of dict
        self._client = OpenAI()

    def _embed_query(self, query: str) -> np.ndarray:
        vec = self._client.embeddings.create(model=EMBED_MODEL, input=[query]).data[0].embedding
        v = np.array(vec, dtype=np.float32)
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        role_filter: Optional[str] = None,
    ) -> list[dict]:
        """返回 top_k 条 bullet，每条附 score。
        role_filter 为角色中文标签（如 "AI产品经理"），None 则全库检索。
        """
        q = self._embed_query(query)
        sims = self.embeddings @ q  # (N,)

        # 构造候选索引
        if role_filter:
            mask = np.array([m.get("role_tag") == role_filter for m in self.meta])
            if mask.sum() == 0:
                # 过滤空集合 → 退化为全库
                mask = np.ones(len(self.meta), dtype=bool)
        else:
            mask = np.ones(len(self.meta), dtype=bool)

        candidate_idx = np.where(mask)[0]
        candidate_sims = sims[candidate_idx]
        top = candidate_idx[np.argsort(-candidate_sims)[:top_k]]

        return [{**self.meta[i], "score": float(sims[i])} for i in top]


# 模块级单例（首次调用时加载）
_SINGLETON: Optional[BulletRetriever] = None


def get_retriever() -> BulletRetriever:
    global _SINGLETON
    if _SINGLETON is None:
        _SINGLETON = BulletRetriever()
    return _SINGLETON


if __name__ == "__main__":
    # demo：模拟 Rewriter 会用的 query
    r = get_retriever()

    cases = [
        ("字节 AI 产品经理 负责大模型应用 / Prompt 工程 / 评测体系", "AI产品经理"),
        ("电商 产品经理 需要数据分析和 AB 测试经验", "产品经理"),
        ("推荐算法工程师 熟悉精排模型 多目标学习", "算法工程师"),
        ("用户增长运营 负责裂变与留存", None),
    ]
    for q, role in cases:
        print(f"\n🔍 Query: {q}")
        print(f"   role_filter: {role}")
        for i, hit in enumerate(r.retrieve(q, top_k=3, role_filter=role), 1):
            print(f"   [{i}] score={hit['score']:.3f}  [{hit['role_tag']}] {hit['bullet'][:60]}...")
