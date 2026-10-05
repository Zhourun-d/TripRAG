"""
POI 清洗模块。

职责：
1. 清洗景点：去重 → 过滤 → 字段规范化 → 截断
2. 清洗锚点：拆分经纬度，供空间排序用

两类数据的区别：
- 景点：需要过滤（黑名单）、打标签、进向量库
- 锚点：不需要过滤（它们本来就是交通枢纽/住宿区），只需拆坐标

过滤规则全部来自 config/cities/{city}.yaml 的 clean_rules 块，
本文件不含硬编码黑名单。改规则只改 yaml，不动代码。

输入：
  data/raw/{city}_pois.json      - 景点原始数据
  data/raw/{city}_anchors.json   - 锚点原始数据
输出：
  data/processed/{city}_clean.json          - 清洗后景点
  data/processed/{city}_anchors_clean.json  - 清洗后锚点

注意：
  clean 阶段不产生 tags / behaviors / description / is_attraction，
  这四个字段由 tag 阶段填充。
  clean 阶段只负责"把 raw 数据规范化成 schema 里 clean 阶段应有的字段"。
"""

import argparse
import json
import re
from collections import Counter

from src.config import (
    get_city_config,
    get_clean_rules,
    raw_path,
    anchors_raw_path,
    clean_path,
    anchors_clean_path,
    list_cities,
)
from src.logger import get_logger

logger = get_logger("clean")


# ============================================================
# 过滤判断（单一入口）
# ============================================================
def is_valid_poi(poi: dict, rules: dict, core_spots_flat: set) -> tuple[bool, str]:
    """
    判断一条 POI 是否保留，并返回原因。

    规则优先级（从上到下，命中即返回）：
        0. 硬过滤        - 命中即干掉，不可被任何豁免绕过
        1. 核心景点完全匹配 - 豁免软过滤，直接保留
        2. 被核心词抓到    - 豁免软过滤，直接保留
        3. 软过滤        - 名字含豁免词则保留，否则干掉
        4. 品牌店格式     - 干掉
        5. 子设施格式     - 干掉
        6. 城市特有黑名单  - 干掉
        7. 通过

    Args:
        poi: 单条 POI 字典
        rules: clean_rules 配置（从 yaml 读）
        core_spots_flat: 核心景点名的集合（拍平后的，用于完全匹配豁免）

    Returns:
        (keep, reason)
        keep: True 保留，False 干掉
        reason: 过滤原因，用于统计
    """
    name = poi.get("name", "")
    poi_type = poi.get("type", "")

    # ---- 规则 0：硬过滤（最高优先级，不可豁免）----
    # 注意：必须在任何豁免之前，否则"归元禅寺停车场"会因核心词被放过
    if any(kw in name for kw in rules.get("hard_exclude_names", [])):
        return False, "hard_exclude"
    if any(kw in poi_type for kw in rules.get("hard_exclude_types", [])):
        return False, "hard_exclude"

    # ---- 规则 1：完全匹配核心景点 → 豁免软过滤 ----
    # 核心景点是我们明确要的，无论类型是什么都保留
    if name in core_spots_flat:
        return True, "core_spot"

    # ---- 规则 2：被核心词抓到 → 豁免软过滤 ----
    # _interest_group 是 crawl 阶段打的标记，有值说明来自核心词抓取
    if poi.get("_interest_group"):
        return True, "interest_group"

    # ---- 规则 3：软过滤 ----
    # 命中软黑名单时，看名字里有没有豁免关键词
    hit_soft = any(kw in poi_type for kw in rules.get("soft_exclude_types", []))
    if hit_soft:
        exempt = any(kw in name for kw in rules.get("soft_exempt_keywords", []))
        if not exempt:
            return False, "soft_exclude"
        # 有豁免词，放行继续后面的检查

    # ---- 规则 4：品牌店格式 ----
    # 如"黄鹤楼(香港路特许店)"，是商业复制品，不是真景点
    pattern = rules.get("brand_store_pattern")
    if pattern and re.search(pattern, name):
        return False, "brand_store"

    # ---- 规则 5：子设施格式 ----
    # 如"归元禅寺-大雄宝殿"，是主景点内部的子设施
    # 判断方式：名字含 "-"，且后缀命中子设施后缀列表
    if "-" in name:
        suffix = name.split("-")[-1]
        bad_suffixes = rules.get("sub_facility_suffixes", [])
        if any(kw in suffix for kw in bad_suffixes):
            return False, "sub_facility"

    # ---- 规则 6：城市特有黑名单 ----
    local_exclude = rules.get("local_exclude", [])
    if local_exclude:
        if any(kw in name for kw in local_exclude):
            return False, "local_exclude"
        if any(kw in poi_type for kw in local_exclude):
            return False, "local_exclude"

    return True, "passed"


# ============================================================
# 去重
# ============================================================
def dedupe(pois: list) -> list:
    """
    按 id 去重。

    同一个景点可能被抓多次：
    - 核心词"黄鹤楼"抓到 → _interest_group = "历史人文"
    - 类别词"风景区"抓到 → _interest_group = None

    去重规则：优先保留有 _interest_group 的那条（核心词抓的更准）。
    """
    best = {}
    for poi in pois:
        pid = poi["id"]
        if pid not in best:
            best[pid] = poi
            continue

        old = best[pid]
        old_has = old.get("_interest_group") is not None
        new_has = poi.get("_interest_group") is not None

        # 新条目有 interest_group 而旧的没有 → 用新的
        if new_has and not old_has:
            best[pid] = poi

    return list(best.values())


# ============================================================
# 字段规范化
# ============================================================
def parse_rating(raw) -> float:
    """
    解析评分，返回 float。

    - 正常数字 → float
    - "暂无" / 空 / 非数字 → -1.0（明确表示"无评分"）

    为什么用 -1.0 而不是 0.0：
        0.0 会和"真的评 0 分"混淆，-1.0 明确表示"没有评分"。
        排序时 -1.0 排在 0 分之后，符合"无评分靠后"的预期。
    """
    if raw is None:
        return -1.0
    try:
        return float(raw)
    except (ValueError, TypeError):
        return -1.0


def extract_photos(photos: list) -> list:
    """
    从高德的 photos 里提取 url。

    高德返回：photos = [{"title": "", "url": "..."}, ...]
    我们只要：["...", "..."]
    """
    if not photos:
        return []
    urls = []
    for p in photos:
        if isinstance(p, dict) and p.get("url"):
            urls.append(p["url"])
    return urls


def clean_one(poi: dict) -> dict:
    """
    把一条高德原始 POI，转成 schema 规范的结构。

    注意：
        clean 阶段不产生 tag 阶段的字段（tags / behaviors / description /
        is_attraction），这些由 tag 阶段填充。
        clean 阶段只负责把 raw 数据规范化成 schema 里 clean 阶段应有的字段。
    """
    business = poi.get("business", {}) or {}

    return {
        # ---- 基础字段 ----
        "id": poi.get("id", ""),
        "name": poi.get("name", ""),
        "city": poi.get("_city", ""),
        "district": poi.get("adname", ""),
        "address": poi.get("address", ""),
        "location": poi.get("location", ""),
        "type": poi.get("type", ""),

        # ---- 业务字段 ----
        "rating": parse_rating(business.get("rating")),
        "opentime": business.get("opentime_today", "") or "",
        "poi_level": business.get("keytag", "") or "",
        "photos": extract_photos(poi.get("photos", [])),

        # ---- 系统字段 ----
        "source": "amap",
        "crawled_at": poi.get("_crawled_at", ""),
    }


# ============================================================
# 清洗景点
# ============================================================
def clean_city(city_key: str, force: bool = False) -> None:
    """
    清洗一个城市的所有景点 POI。

    流程：
        1. 读原始数据
        2. 去重
        3. 过滤（走 is_valid_poi）
        4. 单条清洗（字段映射）
        5. 按 rating 降序排序，超过上限则截断
        6. 保存

    注意：
        district 为空的 POI 直接丢弃——区县是空间排序的必要信息，
        缺失会导致该 POI 无法正确参与行程规划。
    """
    logger.info("=" * 60)
    logger.info(f"开始清洗景点：{city_key}")
    logger.info("=" * 60)

    in_path = raw_path(city_key)
    out_path = clean_path(city_key)

    # ---- 检查输入 ----
    if not in_path.exists():
        logger.error(f"输入文件不存在：{in_path}")
        logger.error(f"请先跑 crawl：python -m src.crawl.amap_crawler --city {city_key}")
        return

    # ---- 检查输出 ----
    if out_path.exists() and not force:
        logger.info(f"输出文件已存在：{out_path}")
        logger.info("如需重新清洗，加 --force 参数")
        return

    # ---- 读配置 ----
    cfg = get_city_config(city_key)
    rules = get_clean_rules(city_key)
    max_pois = cfg.get("limits", {}).get("max_pois", 400)

    # ---- 把核心词拍平成 set，用于豁免 ----
    core_spots_flat = set()
    for spots in cfg.get("core_spots", {}).values():
        core_spots_flat.update(spots)
    logger.info(f"核心词总数：{len(core_spots_flat)}")

    # ---- 读原始数据 ----
    with open(in_path, "r", encoding="utf-8") as f:
        raw_pois = json.load(f)
    logger.info(f"读入原始 POI：{len(raw_pois)} 条")

    # ---- 去重 ----
    deduped = dedupe(raw_pois)
    logger.info(f"去重后：{len(deduped)} 条（去掉 {len(raw_pois) - len(deduped)} 条重复）")

    # ---- 过滤 ----
    # 每个被干掉的 POI 记一次 reason，最后统计
    valid = []
    reason_counter = Counter()

    for p in deduped:
        keep, reason = is_valid_poi(p, rules, core_spots_flat)
        if keep:
            valid.append(p)
        else:
            reason_counter[reason] += 1

    logger.info(f"过滤后：{len(valid)} 条")
    for reason, cnt in reason_counter.most_common():
        logger.info(f"  - {reason}: {cnt} 条")

    # ---- 单条清洗 ----
    cleaned = []
    dropped_no_district = 0

    for p in valid:
        item = clean_one(p)
        # district 为空直接丢弃
        if not item["district"]:
            dropped_no_district += 1
            continue
        cleaned.append(item)

    if dropped_no_district:
        logger.info(f"丢弃 district 为空的：{dropped_no_district} 条")

    # ---- 按 rating 降序排序，超过上限则截断 ----
    # 注意：这里的排序只影响"超限时保留谁"，不影响检索和生成
    cleaned.sort(key=lambda p: p["rating"], reverse=True)

    if len(cleaned) > max_pois:
        logger.info(f"超过上限 {max_pois} 条，按 rating 降序截断")
        cleaned = cleaned[:max_pois]

    # ---- 统计 type 分布（供参考）----
    logger.info("类型分布（按 type 第一段粗看）：")
    type_counter = Counter()
    for p in cleaned:
        t = p.get("type", "")
        first = t.split(";")[0] if t else "未知"
        type_counter[first] += 1
    for t, cnt in type_counter.most_common(10):
        logger.info(f"  {t}: {cnt} 条")

    # ---- 保存 ----
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)

    logger.info("")
    logger.info("=" * 60)
    logger.info(f"景点清洗完成：{len(cleaned)} 条")
    logger.info(f"输出：{out_path}")
    logger.info("=" * 60)


# ============================================================
# 清洗锚点
# ============================================================
def clean_anchors(city_key: str, force: bool = False) -> None:
    """
    清洗一个城市的锚点数据。

    锚点不需要过滤（它们本来就是交通枢纽/住宿区），
    只需要把 location 字符串拆成 lat/lng 两个 float。

    输入格式：
        {"name": "武汉站", "location": "114.424338,30.606981", "type": "transport_hub"}
    输出格式：
        {"name": "武汉站", "lat": 30.606981, "lng": 114.424338, "type": "transport_hub"}

    注意：高德的 location 是"经度,纬度"，转换时要对调顺序。
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"开始清洗锚点：{city_key}")
    logger.info("=" * 60)

    in_path = anchors_raw_path(city_key)
    out_path = anchors_clean_path(city_key)

    if not in_path.exists():
        logger.error(f"锚点输入文件不存在：{in_path}")
        return

    if out_path.exists() and not force:
        logger.info(f"输出文件已存在：{out_path}")
        logger.info("如需重新清洗，加 --force 参数")
        return

    with open(in_path, "r", encoding="utf-8") as f:
        raw_anchors = json.load(f)
    logger.info(f"读入锚点：{len(raw_anchors)} 条")

    cleaned = []
    failed = []

    for a in raw_anchors:
        name = a.get("name", "")
        location = a.get("location", "")

        # 解析 "lng,lat" 字符串
        try:
            lng_str, lat_str = location.split(",")
            lng = float(lng_str)
            lat = float(lat_str)
        except (ValueError, AttributeError):
            logger.warning(f"  [{name}] 坐标解析失败：{location}")
            failed.append(name)
            continue

        cleaned.append({
            "name": name,
            "lat": lat,
            "lng": lng,
            "type": a.get("type", ""),
        })

    if failed:
        logger.warning(f"坐标解析失败 {len(failed)} 条：{failed}")

    # ---- 保存 ----
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)

    logger.info(f"锚点清洗完成：{len(cleaned)} 条")
    logger.info(f"输出：{out_path}")


# ============================================================
# 主流程：清洗景点 + 锚点
# ============================================================
def clean_all(city_key: str, force: bool = False) -> None:
    """清洗一个城市的景点和锚点"""
    clean_city(city_key, force=force)
    clean_anchors(city_key, force=force)


# ============================================================
# 命令行入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="POI 清洗")
    parser.add_argument(
        "--city",
        required=True,
        help=f"城市 key，可选：{list_cities()}",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重新清洗，覆盖已有文件",
    )
    args = parser.parse_args()

    clean_all(args.city, force=args.force)


if __name__ == "__main__":
    main()