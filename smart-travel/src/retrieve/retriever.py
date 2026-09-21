"""
检索接口模块。

职责（大白话）：
1. 加载 Chroma 向量库
2. 提供 search() 接口：传 query 和过滤条件，返回相关 POI
3. 返回带 score 和 rank 的结果
4. 对结果做"同名去重"：同一个主景点最多出现 2 条

设计原则：
- 只负责"检索"，不负责"生成"
- city 是必传过滤（防止跨城市串结果）
- 同名去重：黄鹤楼、黄鹤楼红墙、黄鹤楼公园算同一组
"""

import argparse
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.config import PROJECT_ROOT, CHROMA_DIR
from src.logger import get_logger

load_dotenv(PROJECT_ROOT / ".env")

logger = get_logger("retrieve")


# ============================================================
# 检索结果结构
# ============================================================
@dataclass
class SearchResult:
    """
    一条检索结果。

    - document：LangChain Document（page_content + metadata）
    - score：相似度分数（Chroma 默认 L2 距离，越小越相似）
    - rank：排名，从 1 开始
    """
    document: Document
    score: float
    rank: int

    @property
    def name(self) -> str:
        return self.document.metadata.get("name", "未知")

    @property
    def city(self) -> str:
        return self.document.metadata.get("city", "")

    @property
    def tags(self) -> str:
        return self.document.metadata.get("tags", "")

    @property
    def rating(self) -> str:
        return self.document.metadata.get("rating", "")


# ============================================================
# 同名去重
# ============================================================
def dedupe_by_containment(results: list, max_per_group: int = 2, top_k: int = 10) -> list:
    """
    同名去重：按名字前 3 个字分组，每组最多保留 max_per_group 条。

    例子：
      "黄鹤楼" / "黄鹤楼红墙" / "黄鹤楼公园" → 前 3 字都是 "黄鹤楼" → 同组
      "晴川阁" → 前 3 字是 "晴川阁" → 独立组

    为什么用前 3 字：
      - "包含关系"方案有个 bug：黄鹤楼红墙 和 黄鹤楼故址 互不包含
      - 前 3 字方案简单、可靠，覆盖了大部分主景点的命名习惯
      - 代价是 "湖北省博物馆" 和 "湖北省美术馆" 会被归为同组
        （但它们本来就是同类，可接受）

    Args:
        results: 原始检索结果（已按 score 排序）
        max_per_group: 同一组最多保留几条
        top_k: 最终返回几条

    Returns:
        去重后的结果列表
    """
    PREFIX_LEN = 3
    group_count = {}
    deduped = []

    for r in results:
        name = r.name
        key = name[:PREFIX_LEN]

        count = group_count.get(key, 0)
        if count >= max_per_group:
            continue

        group_count[key] = count + 1
        deduped.append(r)

        if len(deduped) >= top_k:
            break

    return deduped


# ============================================================
# Retriever 类
# ============================================================
class Retriever:
    """
    检索器。

    单例模式：同一个 collection 只加载一次。
    """

    _instances = {}

    def __new__(cls, collection_name: str):
        if collection_name not in cls._instances:
            instance = super().__new__(cls)
            instance._initialized = False
            cls._instances[collection_name] = instance
        return cls._instances[collection_name]

    def __init__(self, collection_name: str):
        if self._initialized:
            return

        self.collection_name = collection_name
        persist_dir = CHROMA_DIR / collection_name

        if not persist_dir.exists():
            raise FileNotFoundError(
                f"向量库不存在：{persist_dir}\n"
                f"请先跑 build 阶段：python -m src.build.kb_builder --city <city>"
            )

        logger.info(f"加载 Chroma：{persist_dir}")
        embeddings = DashScopeEmbeddings(model="text-embedding-v2")
        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=str(persist_dir),
        )
        self._initialized = True

    def search(
        self,
        query: str,
        city: Optional[str] = None,
        tags: Optional[list] = None,
        top_k: int = 10,
        max_per_group: int = 2,
    ) -> list:
        """
        检索接口。

        Args:
            query: 自然语言 query
            city: 城市过滤，如"武汉"
            tags: 兴趣标签过滤（暂不实现）
            top_k: 返回条数
            max_per_group: 同一主景点最多返回几条

        Returns:
            SearchResult 列表，按 score 升序
        """
        # 构造 metadata 过滤
        where = {}
        if city:
            where["city"] = city

        # 多召回：因为后面要去做重，多捞一点备用
        raw_k = top_k * 3

        raw_results = self.vectorstore.similarity_search_with_score(
            query,
            k=raw_k,
            filter=where if where else None,
        )

        # 包装成 SearchResult
        results = [
            SearchResult(document=doc, score=float(score), rank=i + 1)
            for i, (doc, score) in enumerate(raw_results)
        ]

        # 同名去重
        results = dedupe_by_containment(
            results, max_per_group=max_per_group, top_k=top_k
        )

        # 重排 rank（去重后 rank 会乱）
        for i, r in enumerate(results, start=1):
            r.rank = i

        return results


# ============================================================
# 命令行测试入口
# ============================================================
def main():
    """手动测试检索接口"""
    parser = argparse.ArgumentParser(description="检索测试")
    parser.add_argument("--city", required=True, help="城市 key，如 wuhan")
    parser.add_argument("--query", required=True, help="查询语句")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--max_per_group", type=int, default=2,
                        help="同一主景点最多返回几条")
    args = parser.parse_args()

    collection_name = f"{args.city}_v1"
    retriever = Retriever(collection_name)

    city_name_map = {"wuhan": "武汉", "chongqing": "重庆", "nanjing": "南京"}
    city_name = city_name_map.get(args.city, args.city)

    results = retriever.search(
        args.query,
        city=city_name,
        top_k=args.top_k,
        max_per_group=args.max_per_group,
    )

    print("\n" + "=" * 60)
    print(f"Query: {args.query}")
    print(f"City:  {city_name}")
    print(f"max_per_group: {args.max_per_group}")
    print("=" * 60)

    for r in results:
        print(f"[{r.rank:2d}] score={r.score:.4f}  {r.name}  ({r.tags})  {r.rating}")


if __name__ == "__main__":
    main()