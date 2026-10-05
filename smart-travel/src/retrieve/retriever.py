"""
检索接口模块。

职责：
1. 加载 Chroma 向量库 + BM25 索引
2. 提供 search() 接口：支持 vector / bm25 / hybrid 三种模式
3. 返回带 score 和 rank 的结果
4. 直接按 score 截断 top_k（同名收敛已在 build 阶段完成）

设计要点：
- 只负责"检索"，不负责"生成"
- city 是唯一的硬过滤（防止跨城市串结果）
- tags / behaviors 都不过滤，全走语义匹配
- BM25 + 向量 + 加权 RRF，语义为主、BM25 为辅
- query 里的城市名会被剥掉，防止 BM25 被城市名刷分

Chroma 过滤语法备忘：
- 标量字段「值在列表里」  → {"$in": [...]}
- 数组字段「包含某元素」  → {"$contains": "某元素"}
- 顶层只能有一个操作符，多条件必须用 $and / $or 显式包起来
"""

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import jieba
from dotenv import load_dotenv
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.config import PROJECT_ROOT, CHROMA_DIR, DATA_DIR
from src.logger import get_logger

load_dotenv(PROJECT_ROOT / ".env")

logger = get_logger("retrieve")


# ============================================================
# 常量
# ============================================================
# RRF 融合的 k 值（标准值 60）
# score(d) = Σ w_i / (k + rank_i(d))
RRF_K = 60

# RRF 权重：语义为主，BM25 为辅
RRF_WEIGHT_VECTOR = 1.0
RRF_WEIGHT_BM25 = 0.3

# 多召回的倍数：向量多捞一些，BM25 少捞一些
RECALL_MULTIPLIER_VECTOR = 5
RECALL_MULTIPLIER_BM25 = 2

# 无效坐标标记（和 build 阶段一致）
INVALID_COORD = -999.0


# ============================================================
# 检索结果结构
# ============================================================
@dataclass
class SearchResult:
    """
    一条检索结果。

    - document：LangChain Document（page_content + metadata）
    - score：相似度分数。向量模式是 L2 距离（越小越相似），
             BM25 模式是 BM25 分数（越大越相关），
             混合模式是加权 RRF 分数（越大越相关）。
             不同模式分数不可直接比较。
    - rank：排名，从 1 开始
    """
    document: Document
    score: float
    rank: int

    @property
    def id(self) -> str:
        return self.document.metadata.get("id", "")

    @property
    def name(self) -> str:
        return self.document.metadata.get("name", "未知")

    @property
    def city(self) -> str:
        return self.document.metadata.get("city", "")

    @property
    def district(self) -> str:
        return self.document.metadata.get("district", "")

    @property
    def tags(self) -> str:
        """
        tags 在 metadata 里是数组，展示时拼回逗号分隔字符串。
        注意：没 tags 字段的 POI 返回空字符串。
        """
        raw = self.document.metadata.get("tags", [])
        if isinstance(raw, list):
            return ",".join(raw)
        return str(raw)

    @property
    def behaviors(self) -> str:
        """
        behaviors 在 metadata 里是数组，展示时拼回逗号分隔字符串。
        规则同 tags。
        """
        raw = self.document.metadata.get("behaviors", [])
        if isinstance(raw, list):
            return ",".join(raw)
        return str(raw)

    @property
    def aliases(self) -> list:
        """
        同主景点变体名列表（build 阶段归组时写入）。
        如 ["户部巷风情街", "户部巷小吃一条街"]；无则空列表。
        """
        raw = self.document.metadata.get("aliases", [])
        return raw if isinstance(raw, list) else []

    @property
    def level(self) -> str:
        """清洗后的景区等级：5A / 4A / 3A，无则空串。"""
        return str(self.document.metadata.get("level", "") or "")

    @property
    def photos(self) -> list:
        """图片 URL 列表（metadata 里存的是 JSON 字符串）。"""
        raw = self.document.metadata.get("photos", "")
        if isinstance(raw, list):
            return raw
        try:
            v = json.loads(raw) if raw else []
            return v if isinstance(v, list) else []
        except (json.JSONDecodeError, TypeError):
            return []

    @property
    def description(self) -> str:
        """一句话描述。"""
        return str(self.document.metadata.get("description", "") or "")

    @property
    def display_name(self) -> str:
        """清洗后的展示名（去掉括号后缀），无则回退 name。"""
        raw = self.document.metadata.get("display_name", "")
        return str(raw) if raw else self.name

    @property
    def rating(self) -> float:
        """评分，float。-1.0 表示无评分。"""
        raw = self.document.metadata.get("rating", -1.0)
        try:
            return float(raw)
        except (ValueError, TypeError):
            return -1.0

    @property
    def lat(self) -> float:
        """纬度，无效返回 -999.0。"""
        raw = self.document.metadata.get("lat", INVALID_COORD)
        try:
            return float(raw)
        except (ValueError, TypeError):
            return INVALID_COORD

    @property
    def lng(self) -> float:
        """经度，无效返回 -999.0。"""
        raw = self.document.metadata.get("lng", INVALID_COORD)
        try:
            return float(raw)
        except (ValueError, TypeError):
            return INVALID_COORD

    @property
    def has_valid_coord(self) -> bool:
        """坐标是否有效（用于空间排序时过滤）"""
        return self.lat != INVALID_COORD and self.lng != INVALID_COORD


# ============================================================
# 分词（和 build 阶段保持一致）
# ============================================================
def tokenize(text: str) -> list[str]:
    """
    中文分词。必须和 build_bm25_index 用同一套分词逻辑，
    否则 query 和文档的 token 空间不一致，BM25 分数会失真。
    """
    return [w.lower() for w in jieba.cut(text) if w.strip()]


# ============================================================
# query 预处理：剥城市名
# ============================================================
def strip_city_from_query(query: str, city: Optional[str]) -> str:
    """
    把 query 里的城市名剥掉。

    为什么：
        所有 POI 的 text 里都含城市名（"武汉大学，位于武汉武昌区..."）。
        如果 query 里也带"武汉"，BM25 会被这个词刷分，
        把和用户意图无关、但名字里带城市的点顶到前面。

    例：
        "武汉适合了解近代史" + city="武汉" → "适合了解近代史"
        "武汉黄鹤楼" + city="武汉" → "黄鹤楼"
        "适合了解近代史" + city="武汉" → 不变

    注意：
        只剥 query 文本，city 参数本身还要用于 Chroma 硬过滤。
    """
    if not city or not query:
        return query

    # 直接替换，不管位置
    stripped = query.replace(city, "").strip()

    # 如果剥完空了（用户就打了城市名），退回原 query
    if not stripped:
        return query

    return stripped


# ============================================================
# 加权 RRF 融合
# ============================================================
def rrf_fuse(
    vector_results: list[SearchResult],
    bm25_results: list[SearchResult],
    k: int = RRF_K,
    w_vector: float = RRF_WEIGHT_VECTOR,
    w_bm25: float = RRF_WEIGHT_BM25,
) -> list[SearchResult]:
    """
    加权 Reciprocal Rank Fusion：把两路检索结果融合成一个排序。

    公式：
        score(d) = w_v / (k + rank_v(d)) + w_b / (k + rank_b(d))

    为什么用加权 RRF：
        - 两路分数不可比（L2 距离 vs BM25 分数）
        - RRF 只用"排名"，不用"分数"，天然可融合
        - 加权可以控制两路话语权，语义为主（w_v=1.0，w_b=0.3）

    去重规则：
        按 poi_id 去重。同一个 POI 在两路都出现，RRF 分数累加。

    Args:
        vector_results: 向量检索结果（已按 score 升序）
        bm25_results: BM25 检索结果（已按 score 降序）
        k: RRF 的平滑参数，默认 60
        w_vector: 向量路权重
        w_bm25: BM25 路权重

    Returns:
        融合后的 SearchResult 列表，按 RRF 分数降序
    """
    rrf_scores: dict[str, float] = {}
    result_map: dict[str, SearchResult] = {}

    # 向量路：rank 从 1 开始
    for i, r in enumerate(vector_results, start=1):
        poi_id = r.id
        rrf_scores[poi_id] = rrf_scores.get(poi_id, 0.0) + w_vector / (k + i)
        result_map[poi_id] = r

    # BM25 路：rank 从 1 开始
    for i, r in enumerate(bm25_results, start=1):
        poi_id = r.id
        rrf_scores[poi_id] = rrf_scores.get(poi_id, 0.0) + w_bm25 / (k + i)
        if poi_id not in result_map:
            result_map[poi_id] = r

    # 按 RRF 分数降序排
    sorted_ids = sorted(rrf_scores.keys(), key=lambda pid: -rrf_scores[pid])

    fused = []
    for rank, pid in enumerate(sorted_ids, start=1):
        r = result_map[pid]
        fused.append(SearchResult(document=r.document, score=rrf_scores[pid], rank=rank))

    return fused


# ============================================================
# Retriever 类
# ============================================================
class Retriever:
    """
    检索器。支持三种模式：vector / bm25 / hybrid。

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
                f"请先跑 build：python -m src.build.kb_builder --city <city>"
            )

        # ---- 加载向量库 ----
        logger.info(f"加载 Chroma：{persist_dir}")
        embeddings = DashScopeEmbeddings(model="text-embedding-v2")
        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=str(persist_dir),
        )

        # ---- 加载 BM25 ----
        # collection_name 形如 "wuhan_v1"，去掉 "_v1" 得到 city_key
        city_key = collection_name.replace("_v1", "")
        bm25_path = DATA_DIR / "knowledge_base" / f"{city_key}_bm25.pkl"
        bm25_meta_path = DATA_DIR / "knowledge_base" / f"{city_key}_bm25_meta.pkl"

        self.bm25 = None
        self.bm25_meta = None

        if bm25_path.exists() and bm25_meta_path.exists():
            logger.info(f"加载 BM25 索引：{bm25_path}")
            with open(bm25_path, "rb") as f:
                self.bm25 = pickle.load(f)
            with open(bm25_meta_path, "rb") as f:
                self.bm25_meta = pickle.load(f)
        else:
            logger.warning(f"BM25 索引不存在，hybrid/bm25 模式不可用：{bm25_path}")

        self._initialized = True

    # --------------------------------------------------------
    # 向量检索
    # --------------------------------------------------------
    def _search_vector(self, query: str, where: Optional[dict], k: int) -> list[SearchResult]:
        """向量检索，返回 SearchResult 列表（按 L2 距离升序）"""
        raw = self.vectorstore.similarity_search_with_score(
            query,
            k=k,
            filter=where,
        )
        return [
            SearchResult(document=doc, score=float(score), rank=i + 1)
            for i, (doc, score) in enumerate(raw)
        ]

    # --------------------------------------------------------
    # BM25 检索
    # --------------------------------------------------------
    def _search_bm25(
        self,
        query: str,
        city: Optional[str],
        k: int,
    ) -> list[SearchResult]:
        """
        BM25 检索。

        两个关键处理：

        1. 分数过滤：只保留 BM25 分数 > 0 的文档。
           分数 = 0 表示这条文档和 query 没有任何词重叠，不该被召回。
           不加这个过滤，会用 0 分文档填满 top_k，
           这些 0 分文档在 RRF 融合时会稀释真正相关文档的排名。

        2. city 过滤：BM25 不支持 Chroma 的 filter 语法，在 Python 层做。
           当前每个城市一个独立索引文件，理论上不用再筛 city，
           但保险起见还是筛一次。
        """
        if self.bm25 is None:
            logger.warning("BM25 索引未加载，返回空结果")
            return []

        tokens = tokenize(query)
        scores = self.bm25.get_scores(tokens)

        # 按分数降序排，且只保留分数 > 0 的
        sorted_indices = [
            i for i in sorted(range(len(scores)), key=lambda i: -scores[i])
            if scores[i] > 0
        ]

        results = []
        for idx in sorted_indices:
            item = self.bm25_meta[idx]
            metadata = item["metadata"]

            # ---- city 过滤 ----
            if city and metadata.get("city") != city:
                continue

            # ---- 通过过滤，加入结果 ----
            doc = Document(page_content="", metadata=metadata)
            results.append(
                SearchResult(document=doc, score=float(scores[idx]), rank=len(results) + 1)
            )

            # 够了就停
            if len(results) >= k:
                break

        return results

    # --------------------------------------------------------
    # 构造 Chroma 过滤条件
    # --------------------------------------------------------
    def _build_where(self, city: Optional[str]) -> Optional[dict]:
        """
        构造 Chroma 的 where 过滤条件。

        只有 city 是硬过滤，其余全走语义。
        """
        if not city:
            return None
        return {"city": city}

    # --------------------------------------------------------
    # 对外接口
    # --------------------------------------------------------
    def search(
        self,
        query: str,
        city: Optional[str] = None,
        top_k: int = 10,
        mode: str = "hybrid",
    ) -> list:
        """
        检索接口。

        Args:
            query: 自然语言 query
            city: 城市过滤（唯一硬过滤，也用于剥 query 里的城市名）
            top_k: 返回条数
            mode: 检索模式
                - "vector"：只跑向量
                - "bm25"：只跑 BM25
                - "hybrid"：加权 RRF 融合（默认）

        Returns:
            SearchResult 列表
        """
        # ---- 剥 query 里的城市名 ----
        clean_query = strip_city_from_query(query, city)

        # ---- 多召回 ----
        raw_k_vector = top_k * RECALL_MULTIPLIER_VECTOR
        raw_k_bm25 = top_k * RECALL_MULTIPLIER_BM25

        # ---- 构造过滤条件 ----
        where = self._build_where(city)

        # ---- 跑检索 ----
        if mode == "vector":
            results = self._search_vector(clean_query, where, raw_k_vector)

        elif mode == "bm25":
            results = self._search_bm25(clean_query, city, raw_k_bm25)

        elif mode == "hybrid":
            v_results = self._search_vector(clean_query, where, raw_k_vector)
            b_results = self._search_bm25(clean_query, city, raw_k_bm25)
            results = rrf_fuse(v_results, b_results)

        else:
            raise ValueError(f"未知 mode：{mode}，可选 vector / bm25 / hybrid")

        # ---- 截断到 top_k（同名收敛已在 build 阶段完成）----
        results = results[:top_k]

        # ---- 重排 rank ----
        for i, r in enumerate(results, start=1):
            r.rank = i

        return results


# ============================================================
# 命令行测试入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="检索测试")
    parser.add_argument("--city", required=True, help="城市 key，如 wuhan")
    parser.add_argument("--query", required=True, help="查询语句")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--mode", default="hybrid", choices=["vector", "bm25", "hybrid"])
    args = parser.parse_args()

    collection_name = f"{args.city}_v1"
    retriever = Retriever(collection_name)

    city_name_map = {"wuhan": "武汉", "chongqing": "重庆", "nanjing": "南京"}
    city_name = city_name_map.get(args.city, args.city)

    results = retriever.search(
        args.query,
        city=city_name,
        top_k=args.top_k,
        mode=args.mode,
    )

    print("\n" + "=" * 60)
    print(f"Query: {args.query}")
    print(f"City:  {city_name}")
    print(f"Mode:  {args.mode}")
    print("=" * 60)

    for r in results:
        print(f"[{r.rank:2d}] score={r.score:.4f}  {r.name}  "
              f"tags={r.tags}  behaviors={r.behaviors}  "
              f"rating={r.rating:.1f}  ({r.lat:.4f},{r.lng:.4f})")


if __name__ == "__main__":
    main()