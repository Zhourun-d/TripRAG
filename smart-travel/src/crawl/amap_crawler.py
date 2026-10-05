"""
高德 POI 抓取模块。

职责：
1. 从 config/cities/{city}.yaml 读取某个城市的抓取词
2. 调高德 API 抓取景点 POI，输出 data/raw/{city}_pois.json
3. 调高德 API 抓取锚点（交通枢纽 + 住宿区），输出 data/raw/{city}_anchors.json

关于锚点：
    锚点是空间排序的"起点"——Day1 从出发地（交通枢纽）出发，
    Day2+ 从住宿区出发。它们本身不是景点，不进向量库。
    锚点也需要经纬度，所以同样从高德抓，但单独存文件。

关于配置格式：
    transport_hubs / stay_districts 是纯字符串列表，如：
        transport_hubs: [武汉站, 汉口站, 武昌站]
        stay_districts: [武昌区, 汉阳区, ...]
    只写名字，坐标由本模块抓取后写入 {city}_anchors.json。
    后续 clean_anchors 会把坐标拆成 lat/lng 存进 anchors_clean.json。

设计原则：
- 只抓不清洗：清洗是下一步的事
- 断点续跑：输出文件已存在则跳过，加 --force 才重抓
- 失败不中断：单个关键词/锚点失败，记日志继续，最后统一报告
"""

import argparse
import json
import os
import time
from datetime import datetime
from typing import Optional

import requests

from src.config import (
    get_city_config,
    raw_path,
    anchors_raw_path,
    list_cities,
)
from src.logger import get_logger

logger = get_logger("crawl")


# ============================================================
# 高德 API 配置
# ============================================================
AMAP_TEXT_URL = "https://restapi.amap.com/v5/place/text"

# 重试配置：失败后指数退避重试
MAX_RETRIES = 3
RETRY_DELAY = 2  # 秒，基础值。第 n 次重试等 RETRY_DELAY * n 秒

# 锚点抓取配置
# 锚点只需要 1 条（最匹配的），抓 5 条备用即可
ANCHOR_PAGE_SIZE = 5
ANCHOR_MAX_PAGES = 1


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
        city: 城市名，作为 region 参数限制搜索范围
        keyword: 搜索关键词
        page_size: 每页条数
        page_num: 页码，从 1 开始

    Returns:
        POI 列表，失败返回空列表
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

        return data.get("pois", [])

    return []


# ============================================================
# 抓一个关键词（景点用，可能多页）
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
        city: 城市名（region 参数）
        keyword: 搜索词
        page_size: 每页条数
        max_pages: 最多抓几页
        interest_group: 来源兴趣分组，用于给 POI 打标记

    Returns:
        该关键词抓到的所有 POI，每条已附加 _city / _interest_group / _source_keyword
    """
    all_pois = []

    for page_num in range(1, max_pages + 1):
        pois = fetch_page(key, city, keyword, page_size, page_num)

        if not pois:
            break

        logger.info(f"  [{keyword}] 第{page_num}页 {len(pois)} 条")

        # 给每条 POI 补充来源信息，供 clean 阶段使用
        for poi in pois:
            poi["_city"] = city
            poi["_interest_group"] = interest_group
            poi["_source_keyword"] = keyword

        all_pois.extend(pois)

        # 这页不满，说明后面没数据了
        if len(pois) < page_size:
            break

        # 礼貌性 sleep，避免触发限流
        time.sleep(0.3)

    return all_pois


# ============================================================
# 抓单个锚点
# ============================================================
def fetch_anchor(key: str, city: str, name: str) -> Optional[dict]:
    """
    抓取单个锚点，返回最匹配的那条 POI。

    匹配策略：
        1. 优先找名字完全匹配的（poi["name"] == name）
        2. 找不到则退回第一条（高德按相关性排序，第一条通常最准）
        3. 一条都没有则返回 None

    Args:
        key: 高德 API Key
        city: 城市名（region 参数）
        name: 锚点名，如"武汉站"、"武昌区"

    Returns:
        匹配的 POI 字典，或 None（抓取失败）
    """
    pois = fetch_page(
        key=key,
        city=city,
        keyword=name,
        page_size=ANCHOR_PAGE_SIZE,
        page_num=1,
    )

    if not pois:
        return None

    # 优先完全匹配
    for poi in pois:
        if poi.get("name") == name:
            return poi

    # 退而求其次：取第一条，并记 warning
    logger.warning(
        f"  锚点 [{name}] 无完全匹配，退回第一条：{pois[0].get('name')}"
    )
    return pois[0]


# ============================================================
# 抓一个城市的所有锚点
# ============================================================
def crawl_anchors(key: str, region: str, cfg: dict) -> list[dict]:
    """
    抓取一个城市的所有锚点（交通枢纽 + 住宿区）。

    锚点数据只保留：name、location、type。
    不需要 tags、rating、photos 等景点字段。

    Args:
        key: 高德 API Key
        region: 高德 region 参数
        cfg: 城市配置（读 transport_hubs / stay_districts，纯字符串列表）

    Returns:
        锚点列表：[{"name": ..., "location": ..., "type": ...}, ...]
    """
    anchors = []
    missed = []  # 记录没抓到的锚点，最后统一报告

    # ---- 交通枢纽 ----
    hubs = cfg.get("transport_hubs", []) or []
    logger.info(f"交通枢纽：{len(hubs)} 个")

    for name in hubs:
        poi = fetch_anchor(key, region, name)

        if poi:
            anchors.append({
                "name": name,
                "location": poi.get("location", ""),
                "type": "transport_hub",
            })
            logger.info(f"  [{name}] 抓到：{poi.get('location', '')}")
        else:
            missed.append(name)
            logger.warning(f"  [{name}] 未抓到")

        time.sleep(0.3)

    # ---- 住宿区 ----
    districts = cfg.get("stay_districts", []) or []
    logger.info(f"住宿区：{len(districts)} 个")

    for name in districts:
        poi = fetch_anchor(key, region, name)

        if poi:
            anchors.append({
                "name": name,
                "location": poi.get("location", ""),
                "type": "stay_district",
            })
            logger.info(f"  [{name}] 抓到：{poi.get('location', '')}")
        else:
            missed.append(name)
            logger.warning(f"  [{name}] 未抓到")

        time.sleep(0.3)

    # ---- 汇总未抓到的锚点 ----
    if missed:
        logger.warning(f"未抓到的锚点（{len(missed)} 个）：{missed}")

    return anchors


# ============================================================
# 抓一个城市
# ============================================================
def crawl_city(city_key: str, force: bool = False) -> None:
    """
    抓取一个城市的景点和锚点。

    输出：
        data/raw/{city}_pois.json     - 景点
        data/raw/{city}_anchors.json  - 锚点

    Args:
        city_key: 城市 key，如 "wuhan"
        force: True 时即使输出文件已存在也重新抓
    """
    logger.info("=" * 60)
    logger.info(f"开始抓取城市：{city_key}")
    logger.info("=" * 60)

    # ---- 检查输出文件 ----
    pois_path = raw_path(city_key)
    anchors_path = anchors_raw_path(city_key)

    if pois_path.exists() and anchors_path.exists() and not force:
        logger.info(f"输出文件已存在：")
        logger.info(f"  景点：{pois_path}")
        logger.info(f"  锚点：{anchors_path}")
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

    start_time = time.time()
    crawled_at = datetime.now().isoformat()

    # ========================================================
    # 第一轮：核心景点
    # ========================================================
    logger.info("")
    logger.info("【第一轮】核心景点抓取")
    logger.info("-" * 60)

    all_pois = []

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

    # ========================================================
    # 第二轮：类别补充
    # ========================================================
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

    # ---- 给景点补抓取时间 ----
    for poi in all_pois:
        poi["_crawled_at"] = crawled_at

    # ---- 保存景点 ----
    pois_path.parent.mkdir(parents=True, exist_ok=True)
    with open(pois_path, "w", encoding="utf-8") as f:
        json.dump(all_pois, f, ensure_ascii=False, indent=2)

    logger.info("")
    logger.info(f"景点抓取完成：{len(all_pois)} 条 → {pois_path}")

    # ========================================================
    # 第三轮：锚点
    # ========================================================
    logger.info("")
    logger.info("【第三轮】锚点抓取（交通枢纽 + 住宿区）")
    logger.info("-" * 60)

    anchors = crawl_anchors(key, region, cfg)

    # ---- 保存锚点 ----
    anchors_path.parent.mkdir(parents=True, exist_ok=True)
    with open(anchors_path, "w", encoding="utf-8") as f:
        json.dump(anchors, f, ensure_ascii=False, indent=2)

    logger.info("")
    logger.info(f"锚点抓取完成：{len(anchors)} 条 → {anchors_path}")

    # ========================================================
    # 汇总
    # ========================================================
    elapsed = time.time() - start_time
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"抓取完成")
    logger.info(f"  景点：{len(all_pois)} 条")
    logger.info(f"  锚点：{len(anchors)} 条")
    logger.info(f"  耗时：{elapsed:.1f} 秒")
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