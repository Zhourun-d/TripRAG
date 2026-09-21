"""
知识库构建模块。

职责：
1. 读打完标签的 POI
2. 按 schema.yaml 的 text_template，把字段拼成 text
3. 调 embedding 模型，把 text 转成向量
4. 存进 Chroma（每城市一个 collection）
5. 同时输出一份 jsonl 存档（人类可读）

输入：data/processed/{city}_tagged.jsonl
输出：
  - data/knowledge_base/{city}_v1.jsonl  （人类可读的存档）
  - chroma_db/{city}_v1/                 （向量数据库）
"""

import argparse
import json
import shutil
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_chroma import Chroma

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
# 2. 拼 text（把字段拼成一段自然语言）
# ============================================================
def build_text(poi: dict, template: str) -> str:
    """
    按 text_template 把 POI 字段拼成 text。

    模板里的 {xxx} 会被替换成 POI 里对应字段的值。
    如果字段不存在或为空，替换成空字符串。

    为什么要拼 text：
    - embedding 模型吃的是自然语言，不是 JSON
    - 拼好的 text 会进向量库，检索时用它算相似度
    """
    # 把 POI 里需要的字段都准备好
    values = {
        "name": poi.get("name", ""),
        "city": poi.get("city", ""),
        "district": poi.get("district", ""),
        "category": poi.get("category", ""),
        "address": poi.get("address", ""),
        "rating": poi.get("rating", ""),
        "opentime": poi.get("opentime", ""),
        "tags": poi.get("tags", ""),
        "crowd": poi.get("crowd", ""),
        "highlight": poi.get("highlight", ""),
    }

    # 用 format 替换模板里的占位符
    # 注意：如果模板里有 {xxx} 但 values 里没有，会报 KeyError
    # 所以用 defaultdict 或者 try-except 容错
    try:
        text = template.format(**values)
    except KeyError as e:
        logger.warning(f"  模板字段缺失：{e}，用原始 name 兜底")
        text = f"【景点名称】{values['name']}"

    return text.strip()


# ============================================================
# 3. 构造 Document 列表
# ============================================================
def build_documents(pois: list[dict], template: str) -> list[Document]:
    """
    把 POI 列表转成 LangChain 的 Document 列表。

    Document 是 LangChain 的标准结构：
    - page_content：正文（就是拼好的 text）
    - metadata：附加信息（用于过滤和展示）
    """
    documents = []
    for poi in pois:
        text = build_text(poi, template)

        # 只放"用于过滤和展示"的字段进 metadata
        # 不放 text 里的字段（因为 text 已经在 page_content 里了）
        metadata = {
            "id": poi.get("id", ""),
            "name": poi.get("name", ""),
            "city": poi.get("city", ""),
            "district": poi.get("district", ""),
            "category": poi.get("category", ""),
            "rating": poi.get("rating", ""),
            "tags": poi.get("tags", ""),
            "crowd": poi.get("crowd", ""),
            "location": poi.get("location", ""),
            "poi_level": poi.get("poi_level", ""),
            "source": poi.get("source", ""),
        }

        documents.append(Document(page_content=text, metadata=metadata))

    return documents


# ============================================================
# 4. 写入 Chroma
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
    # 全量重建：先删旧目录
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

    logger.info(f"  写入完成")
    return vectorstore


# ============================================================
# 5. 输出 jsonl 存档
# ============================================================
def save_jsonl(pois: list[dict], documents: list[Document], out_path: Path):
    """
    把 POI + 拼好的 text 存成 jsonl。

    为什么要存 text：
    - 便于人类查看"到底往向量库里塞了什么"
    - 如果以后换 embedding 模型，不用重新拼 text，直接用这个文件重建
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

    # ---- 拼 text ----
    logger.info("拼装 text...")
    documents = build_documents(pois, template)

    # ---- 打印一条样例，方便肉眼检查 ----
    logger.info("")
    logger.info("=" * 60)
    logger.info("样例 text（第一条）：")
    logger.info("=" * 60)
    logger.info(documents[0].page_content)
    logger.info("=" * 60)
    logger.info("")

    # ---- 写入 Chroma ----
    write_to_chroma(documents, collection_name, chroma_dir)

    # ---- 保存 jsonl 存档 ----
    logger.info(f"保存 jsonl 存档：{out_jsonl}")
    save_jsonl(pois, documents, out_jsonl)

    logger.info("")
    logger.info("=" * 60)
    logger.info(f"构建完成")
    logger.info(f"  collection：{collection_name}")
    logger.info(f"  Chroma 目录：{chroma_dir}")
    logger.info(f"  jsonl 存档：{out_jsonl}")
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