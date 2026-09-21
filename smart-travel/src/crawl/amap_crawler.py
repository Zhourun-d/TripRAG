"""
高德 POI 抓取模块。

职责：
1. 从 cities.yaml 读取某个城市的抓取词
2. 调用高德 API 抓取 POI
3. 给每条 POI 打上 city 和 interest_group 标记
4. 保存原始结果到 data/raw/{city}_pois.json

设计原则：
- 只抓不清洗：清洗是下一步的事，crawl 只负责"抓下来"
- 断点续跑：已抓过的关键词跳过，避免重复消耗 API 配额
- 失败不中断：单个关键词失败，记日志继续，最后统一报告
"""

import argparse
import json
import os
import time
from datetime import datetime
from typing import Optional

import requests

from src.config import get_city_config, raw_path, list_cities
from src.logger import get_logger

logger = get_logger("crawl")


# ============================================================
# 高德 API 配置
# ============================================================
AMAP_TEXT_URL = "https://restapi.amap.com/v5/place/text"

# 全局重试配置
MAX_RETRIES = 3
RETRY_DELAY = 2  # 秒，指数退避的基础值


# ============================================================
# API Key
# ============================================================
def get_amap_key() -> str:
    """从环境变量读高德 Key"""
    key = os.getenv("AMAP_API_KEY")
    if not key:
        raise ValueError(
            "没读到 AMAP_API_KEY。请在项目根目录建 .env 文件，"
            "写入 AMAP_API_KEY=你的key"
        )
    return key


# ============================================================
# 单次 API 调用
# ============================================================
def fetch_page(
    key: str,
    city: str,
    keyword: str,
    page_size: int,
    page_num: int,
) -> list[dict]:
    """
    调用高德 API 抓一页 POI。

    Args:
        key: 高德 API Key
        city: 城市名，如"武汉"
        keyword: 搜索关键词，如"黄鹤楼"
        page_size: 每页条数
        page_num: 页码，从 1 开始

    Returns:
        POI 列表，失败返回空列表

    为什么单独抽一个函数：
    - 重试逻辑只写一次
    - 上层循环只关心"抓到了几条"，不关心 HTTP 细节
    """
    params = {
        "key": key,
        "keywords": keyword,
        "region": city,
        "page_size": page_size,
        "page_num": page_num,
        "show_fields": "business,photos",
    }

    # 指数退避重试
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(AMAP_TEXT_URL, params=params, timeout=10)
            data = resp.json()
        except Exception as e:
            logger.warning(
                f"  [{keyword}] 第{page_num}页 请求异常 "
                f"(尝试 {attempt}/{MAX_RETRIES}): {e}"
            )
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * attempt)
                continue
            return []

        # 高德返回 status=1 表示成功
        if data.get("status") != "1":
            logger.warning(
                f"  [{keyword}] 第{page_num}页 API 返回异常: "
                f"status={data.get('status')}, info={data.get('info')}"
            )
            return []

        pois = data.get("pois", [])
        return pois

    return []


# ============================================================
# 抓一个关键词（可能多页）
# ============================================================
def fetch_keyword(
    key: str,
    city: str,
    keyword: str,
    page_size: int,
    max_pages: int,
    interest_group: Optional[str] = None,
) -> list[dict]:
    """
    抓一个关键词的所有页。

    Args:
        key: 高德 API Key
        city: 城市名
        keyword: 搜索词
        page_size: 每页条数
        max_pages: 最多抓几页
        interest_group: 来源兴趣分组，用于给 POI 打标记

    Returns:
        该关键词抓到的所有 POI，每条已附加 city 和 interest_group
    """
    all_pois = []

    for page_num in range(1, max_pages + 1):
        pois = fetch_page(key, city, keyword, page_size, page_num)

        if not pois:
            break

        logger.info(f"  [{keyword}] 第{page_num}页 {len(pois)} 条")

        # 给每条 POI 补充来源信息
        for poi in pois:
            poi["_city"] = city
            poi["_interest_group"] = interest_group
            poi["_source_keyword"] = keyword

        all_pois.extend(pois)

        # 如果这页不满，说明后面没数据了
        if len(pois) < page_size:
            break

        # 礼貌性 sleep，避免触发限流
        time.sleep(0.3)

    return all_pois


# ============================================================
# 抓一个城市
# ============================================================
def crawl_city(city_key: str, force: bool = False) -> None:
    """
    抓取一个城市的所有 POI。

    Args:
        city_key: 城市 key，如 "wuhan"
        force: 如果 True，即使输出文件已存在也重新抓
    """
    logger.info("=" * 60)
    logger.info(f"开始抓取城市：{city_key}")
    logger.info("=" * 60)

    # ---- 检查输出文件 ----
    out_path = raw_path(city_key)
    if out_path.exists() and not force:
        logger.info(f"输出文件已存在：{out_path}")
        logger.info("如需重新抓取，加 --force 参数")
        return

    # ---- 读配置 ----
    cfg = get_city_config(city_key)
    city_name = cfg["name"]
    region = cfg["region"]
    crawl_cfg = cfg["crawl"]
    core_spots = cfg.get("core_spots", {})
    category_words = cfg.get("category_words", [])

    logger.info(f"城市：{city_name}（region={region}）")
    logger.info(f"核心景点分组：{list(core_spots.keys())}")
    logger.info(f"类别词：{category_words}")

    # ---- 读 Key ----
    key = get_amap_key()

    all_pois = []
    start_time = time.time()

    # ---- 第一轮：核心景点（每个兴趣组下每个词，page_size 小，抓 1 页）----
    logger.info("")
    logger.info("【第一轮】核心景点抓取")
    logger.info("-" * 60)

    for interest, spots in core_spots.items():
        logger.info(f"[{interest}] {len(spots)} 个核心词")
        for spot in spots:
            pois = fetch_keyword(
                key=key,
                city=region,
                keyword=spot,
                page_size=crawl_cfg["page_size_core"],
                max_pages=crawl_cfg["max_pages_core"],
                interest_group=interest,
            )
            all_pois.extend(pois)

    # ---- 第二轮：类别补充（page_size 大，抓 2 页）----
    logger.info("")
    logger.info("【第二轮】类别补充抓取")
    logger.info("-" * 60)

    for kw in category_words:
        pois = fetch_keyword(
            key=key,
            city=region,
            keyword=kw,
            page_size=crawl_cfg["page_size_category"],
            max_pages=crawl_cfg["max_pages_category"],
            interest_group=None,  # 类别词不归属某个兴趣
        )
        all_pois.extend(pois)

    # ---- 补充抓取时间 ----
    crawled_at = datetime.now().isoformat()
    for poi in all_pois:
        poi["_crawled_at"] = crawled_at

    # ---- 保存 ----
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_pois, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - start_time
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"抓取完成：{len(all_pois)} 条原始 POI")
    logger.info(f"耗时：{elapsed:.1f} 秒")
    logger.info(f"输出：{out_path}")
    logger.info("=" * 60)


# ============================================================
# 命令行入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="高德 POI 抓取")
    parser.add_argument(
        "--city",
        required=True,
        help=f"城市 key，可选：{list_cities()}",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重新抓取，覆盖已有文件",
    )
    args = parser.parse_args()

    crawl_city(args.city, force=args.force)


if __name__ == "__main__":
    main()