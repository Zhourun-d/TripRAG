"""
知识库构建模块。

职责：
1. 读打完标签的 POI
2. 按 schema.yaml 的 text_template，把字段拼成 text
3. 调 embedding 模型，把 text 转成向量
4. 存进 Chroma（每城市一个 collection）
5. 同时输出一份 jsonl 存档
6. 建 BM25 索引（供混合检索用）

输入：data/processed/{city}_tagged.jsonl
输出：
  - data/knowledge_base/{city}_v1.jsonl  （人类可读的存档）
  - data/knowledge_base/{city}_bm25.pkl  （BM25 索引）
  - data/knowledge_base/{city}_bm25_meta.pkl  （BM25 元数据）
  - chroma_db/{city}_v1/                 （向量数据库）

关键处理：
- is_attraction=false 的 POI 直接剔除，不进 Chroma、不进存档
- location 字符串 "lng,lat" 拆成 lat/lng 两个 float 进 metadata
- type 派生成 clean_type（统一分隔符、去泛词、去重），进 text
- tags / behaviors / description 规范化后拼进 text
- photos 存 JSON 字符串：Chroma metadata 不支持嵌套 list
"""

import argparse
import json
import shutil
import pickle
import jieba

from pathlib import Path
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_chroma import Chroma
from rank_bm25 import BM25Okapi

from src.config import (
    PROJECT_ROOT,
    tagged_path,
    kb_path,
    CHROMA_DIR,
    list_cities,
    get_text_template,
)
from src.logger import get_logger

# 加载 .env（确保 DASHSCOPE_API_KEY 可用）
load_dotenv(PROJECT_ROOT / ".env")

logger = get_logger("build")


# ============================================================
# 配置
# ============================================================
EMBEDDING_MODEL = "text-embedding-v2"   # 阿里云的 embedding 模型

# 无效坐标的标记值
INVALID_COORD = -999.0

# type 清洗时去掉的泛词
# 这些词对语义贡献小，且占 token
GENERIC_TYPE_WORDS = {
    "风景名胜",
    "风景名胜相关",
    "公园广场",
    "餐饮相关",
    "购物服务",
    "购物相关场所",
    "科教文化服务",
    "其他",
    "地名地址信息",
    "交通地名",
    "自然地名",
}


# ============================================================
# 1. 读 tagged jsonl
# ============================================================
def load_tagged(path: Path) -> list[dict]:
    """
    读 jsonl 文件，返回 POI 列表。

    jsonl 格式：每行一个 JSON 对象。
    """
    logger.info(f"读取：{path}")
    pois = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                pois.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(f"  跳过一行解析失败的数据：{e}")
    logger.info(f"  读入 {len(pois)} 条 POI")
    return pois


# ============================================================
# 2. 工具函数
# ============================================================
def _split_tags(raw) -> list:
    """
    把逗号分隔的标签字符串，拆成数组。

    例：
        "历史人文,自然风光"  → ["历史人文", "自然风光"]
        "家庭, 朋友"         → ["家庭", "朋友"]   （去掉空格）
        ""                   → []
        None                 → []

    为什么要拆成数组：
    - Chroma 对字符串只支持精确匹配（等于）
    - 对数组支持 $contains 过滤（包含）
    - 前端要让用户按标签过滤，必须用数组
    """
    if not raw:
        return []

    # 兼容中文逗号、英文逗号
    raw = str(raw).replace("，", ",")

    # 拆分 + 去空格 + 去空项
    items = [x.strip() for x in raw.split(",")]
    return [x for x in items if x]


def _clean_type(raw_type: str) -> str:
    """
    清洗 type：统一分隔符、去泛词、去重、逗号拼接。

    高德的 type 有两级分隔符：
      - ; 分隔同一分类树内的多个类型
      - | 分隔不同分类树

    例：
        "风景名胜;风景名胜;国家级景点"  → "国家级景点"
        "科教文化服务;博物馆;博物馆"    → "博物馆"
        "购物服务;购物相关场所"         → ""  （全被筛光）
        "购物服务;特色商业街;步行街|地名地址信息;自然地名;河流|风景名胜;风景名胜相关;旅游景点"
            → "特色商业街,步行街,河流,旅游景点"
        ""                              → ""

    为空时，build_text 里会用"景点"兜底。
    """
    if not raw_type:
        return ""

    # 统一分隔符：| 也当 ; 处理
    normalized = str(raw_type).replace("|", ";")

    # 按 ; 拆开
    parts = [p.strip() for p in normalized.split(";") if p.strip()]

    # 去泛词
    filtered = [p for p in parts if p not in GENERIC_TYPE_WORDS]

    # 去重（保持顺序）
    seen = set()
    deduped = []
    for p in filtered:
        if p not in seen:
            seen.add(p)
            deduped.append(p)

    return ",".join(deduped)


def _parse_location(location: str) -> tuple[float, float]:
    """
    解析高德的 location 字符串，返回 (lat, lng)。

    高德格式："经度,纬度"，例如 "114.364514,30.536243"
    注意顺序：经度在前，纬度在后。返回时对调成 (lat, lng)。

    解析失败返回 (INVALID_COORD, INVALID_COORD)。
    空间排序时会跳过这些点。
    """
    if not location:
        return INVALID_COORD, INVALID_COORD

    try:
        lng_str, lat_str = location.split(",")
        lng = float(lng_str)
        lat = float(lat_str)
        return lat, lng
    except (ValueError, AttributeError):
        return INVALID_COORD, INVALID_COORD


# ============================================================
# 3. 拼 text（把字段拼成一段自然语言）
# ============================================================
def build_text(poi: dict, template: str) -> str:
    """
    按 text_template 把 POI 字段拼成 text。

    模板里的 {xxx} 会被替换成 POI 里对应字段的值。
    如果字段不存在或为空，用空字符串兜底。

    为什么要拼 text：
    - embedding 模型吃的是自然语言，不是 JSON
    - 拼好的 text 会进向量库，检索时用它算相似度
    """
    # tags / behaviors 规范化后再进 text
    tags_normalized = ",".join(_split_tags(poi.get("tags", "")))
    behaviors_normalized = ",".join(_split_tags(poi.get("behaviors", "")))
    clean_type = _clean_type(poi.get("type", ""))

    values = {
        "name": poi.get("name", ""),
        "city": poi.get("city", ""),
        "district": poi.get("district", ""),
        "clean_type": clean_type or "景点",   # 空时兜底
        "tags": tags_normalized,
        "behaviors": behaviors_normalized,
        "description": poi.get("description", "") or "",
    }

    try:
        text = template.format(**values)
    except KeyError as e:
        logger.warning(f"  模板字段缺失：{e}，用原始 name 兜底")
        text = f"【景点名称】{values['name']}"

    return text.strip()


# ============================================================
# 4. 构造 Document 列表
# ============================================================
def build_documents(pois: list[dict], template: str) -> tuple[list[Document], list[dict]]:
    """
    把 POI 列表转成 LangChain 的 Document 列表。

    is_attraction=false 的 POI 会被跳过——它们不是旅游地点，
    不该进向量库，也不该出现在存档里。

    Document 是 LangChain 的标准结构：
    - page_content：正文（拼好的 text），给 embedding 用
    - metadata：附加信息（结构化字段），给过滤和展示用

    metadata 字段说明：
    - tags：数组，支持 Chroma 的 $contains 过滤。
    - behaviors：数组，同上。
    - photos：JSON 字符串，Chroma 不支持嵌套 list
    - lat/lng：float，供空间排序直接计算

    Returns:
        (documents, kept_pois)
        documents: Document 列表
        kept_pois: 和 documents 一一对应的 POI 列表（已过滤 is_attraction=false）
    """
    documents = []
    kept_pois = []
    skipped = 0

    for poi in pois:
        # is_attraction=false 的是"场所"，不是旅游地点，跳过
        if not poi.get("is_attraction", True):
            skipped += 1
            continue

        text = build_text(poi, template)

        # 解析坐标
        lat, lng = _parse_location(poi.get("location", ""))

        # tags / behaviors 拆成数组，供 $contains 过滤
        tags_list = _split_tags(poi.get("tags", ""))
        behaviors_list = _split_tags(poi.get("behaviors", ""))

        # photos 序列化成 JSON 字符串
        photos_raw = poi.get("photos", [])
        photos_str = json.dumps(photos_raw, ensure_ascii=False) if photos_raw else "[]"

        metadata = {
            # ---- 基础标识 ----
            "id": poi.get("id", ""),
            "name": poi.get("name", ""),

            # ---- 过滤字段 ----
            "city": poi.get("city", ""),
            "district": poi.get("district", ""),

            # ---- 展示字段 ----
            "type": poi.get("type", ""),
            "rating": poi.get("rating", -1.0),
            "opentime": poi.get("opentime", ""),
            "poi_level": poi.get("poi_level", ""),
            "address": poi.get("address", ""),
            "photos": photos_str,
            "description": poi.get("description", ""),

            # ---- 计算字段 ----
            "lat": lat,
            "lng": lng,

            # ---- 系统字段 ----
            "source": poi.get("source", ""),
        }

        # tags / behaviors 非空时才写入
        if tags_list:
            metadata["tags"] = tags_list
        if behaviors_list:
            metadata["behaviors"] = behaviors_list

        documents.append(Document(page_content=text, metadata=metadata))
        kept_pois.append(poi)

    if skipped:
        logger.info(f"  跳过 is_attraction=false 的 POI：{skipped} 条")
    logger.info(f"  实际进库：{len(documents)} 条")

    return documents, kept_pois


# ============================================================
# 5. 写入 Chroma
# ============================================================
def write_to_chroma(documents: list[Document], collection_name: str, persist_dir: Path):
    """
    把 Document 列表写入 Chroma。

    步骤：
    1. 如果目录已存在，先删掉（全量重建）
    2. 初始化 embedding 模型
    3. Chroma.from_documents 会自动：
       - 对每条 Document 的 page_content 调 embedding
       - 把 text + vector + metadata 存进 Chroma
    """
    if persist_dir.exists():
        logger.info(f"  删除旧目录：{persist_dir}")
        shutil.rmtree(persist_dir)

    logger.info(f"  初始化 embedding 模型：{EMBEDDING_MODEL}")
    embeddings = DashScopeEmbeddings(model=EMBEDDING_MODEL)

    logger.info(f"  写入 Chroma（{len(documents)} 条）...")
    logger.info("  注意：embedding 调用较慢，请耐心等待")

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=str(persist_dir),
    )

    logger.info("  写入完成")
    return vectorstore


# ============================================================
# 6. 输出 jsonl 存档
# ============================================================
def save_jsonl(pois: list[dict], documents: list[Document], out_path: Path):
    """
    把 POI + 拼好的 text 存成 jsonl。

    为什么要存 text：
    - 便于人类查看"到底往向量库里塞了什么"
    - 如果以后换 embedding 模型，不用重新拼 text，直接用这个文件重建

    注意：pois 已经是过滤后的（不含 is_attraction=false），
    和 documents 一一对应。
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for poi, doc in zip(pois, documents):
            item = {
                **poi,                    # 原 POI 的所有字段
                "text": doc.page_content,  # 加上拼好的 text
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


# ============================================================
# 7. 构建 BM25 索引
# ============================================================
def tokenize(text: str) -> list[str]:
    """
    中文分词。

    用 jieba 切词，转小写（兼容英文）。
    为什么要分词：BM25 是词袋模型，必须把句子切成词才能算词频。

    例：
        "武汉大学 历史人文 武昌区"
        → ["武汉大学", "历史人文", "武昌区"]
    """
    return [w.lower() for w in jieba.cut(text) if w.strip()]


def build_bm25_index(
    documents: list[Document],
    index_path: Path,
    meta_path: Path,
) -> None:
    """
    构建 BM25 索引并保存。

    存两个文件：
    - {city}_bm25.pkl      : BM25Okapi 对象（含词频统计）
    - {city}_bm25_meta.pkl : 元数据（id / name / metadata 映射）

    为什么要存 meta：
        BM25 返回的是"第几条文档"，需要靠 meta 还原成具体 POI。
    """
    logger.info(f"  分词中（{len(documents)} 条）...")

    # 对每条 doc 的 page_content 分词
    tokenized_corpus = [tokenize(doc.page_content) for doc in documents]

    logger.info("  构建 BM25 索引...")
    bm25 = BM25Okapi(tokenized_corpus)

    # 元数据：保持和 tokenized_corpus 的顺序一致
    meta = [
        {
            "id": doc.metadata.get("id", ""),
            "name": doc.metadata.get("name", ""),
            "metadata": doc.metadata,   # 完整 metadata，还原时直接用
        }
        for doc in documents
    ]

    # 保存
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with open(index_path, "wb") as f:
        pickle.dump(bm25, f)
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)

    logger.info(f"  BM25 索引已保存：{index_path}")
    logger.info(f"  BM25 元数据已保存：{meta_path}")


# ============================================================
# 主流程
# ============================================================
def build_city(city_key: str, force: bool = False) -> None:
    logger.info("=" * 60)
    logger.info(f"开始构建知识库：{city_key}")
    logger.info("=" * 60)

    in_path = tagged_path(city_key)
    out_jsonl = kb_path(city_key)
    chroma_dir = CHROMA_DIR / f"{city_key}_v1"
    collection_name = f"{city_key}_v1"

    # ---- 检查输入 ----
    if not in_path.exists():
        logger.error(f"输入文件不存在：{in_path}")
        logger.error(f"请先跑 tag 阶段：python -m src.tag.poi_tagger --city {city_key}")
        return

    # ---- 检查输出 ----
    if out_jsonl.exists() and not force:
        logger.info(f"输出文件已存在：{out_jsonl}")
        logger.info("如需重建，加 --force 参数")
        return

    # ---- 读数据 ----
    pois = load_tagged(in_path)

    # ---- 读 text 模板 ----
    template = get_text_template()
    if not template:
        logger.error("text_template 为空，检查 schema.yaml")
        return

    # ---- 拼 text + 过滤非景点 ----
    logger.info("拼装 text（同时过滤 is_attraction=false）...")
    documents, kept_pois = build_documents(pois, template)

    if not documents:
        logger.error("过滤后没有任何 POI，检查 tag 结果")
        return

    # ---- 打印一条样例，方便肉眼检查 ----
    logger.info("")
    logger.info("=" * 60)
    logger.info("样例 text（第一条）：")
    logger.info("=" * 60)
    logger.info(documents[0].page_content)
    logger.info("=" * 60)
    logger.info("")
    logger.info("样例 metadata（第一条）：")
    logger.info("=" * 60)
    for k, v in documents[0].metadata.items():
        logger.info(f"  {k}: {v}")
    logger.info("=" * 60)
    logger.info("")

    # ---- 写入 Chroma ----
    write_to_chroma(documents, collection_name, chroma_dir)

    # ---- 保存 jsonl 存档 ----
    logger.info(f"保存 jsonl 存档：{out_jsonl}")
    save_jsonl(kept_pois, documents, out_jsonl)

    # ---- 构建 BM25 索引 ----
    logger.info("")
    logger.info("构建 BM25 索引...")
    bm25_path = out_jsonl.parent / f"{city_key}_bm25.pkl"
    bm25_meta_path = out_jsonl.parent / f"{city_key}_bm25_meta.pkl"
    build_bm25_index(documents, bm25_path, bm25_meta_path)

    logger.info("")
    logger.info("=" * 60)
    logger.info("构建完成")
    logger.info(f"  collection：{collection_name}")
    logger.info(f"  Chroma 目录：{chroma_dir}")
    logger.info(f"  jsonl 存档：{out_jsonl}")
    logger.info(f"  BM25 索引：{bm25_path}")
    logger.info(f"  BM25 元数据：{bm25_meta_path}")
    logger.info("=" * 60)


# ============================================================
# 命令行入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="知识库构建")
    parser.add_argument(
        "--city",
        required=True,
        help=f"城市 key，可选：{list_cities()}",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重建，覆盖已有输出",
    )
    args = parser.parse_args()

    build_city(args.city, force=args.force)


if __name__ == "__main__":
    main()