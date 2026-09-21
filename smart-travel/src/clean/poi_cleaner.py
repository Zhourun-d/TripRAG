"""
POI 清洗模块。

职责（大白话）：
1. 去重：同一个景点被抓多次，只留一条
2. 两层过滤：硬过滤（不可豁免）+ 软过滤（可豁免）
3. 整理格式：把高德给的数据，整理成我们要的样子
4. 补齐字段：比如 category（大类），从 type 推断

两层过滤的设计：
- 硬过滤：地铁站、停车场、售票处这种，绝对不可能是景点，命中就干掉
- 软过滤：学校、政府机关这种，可能是景点（如武汉大学），需要豁免机制

输入：data/raw/{city}_pois.json
输出：data/processed/{city}_clean.json
"""

import argparse
import json
import re
from collections import Counter

from src.config import get_city_config, raw_path, clean_path, list_cities
from src.logger import get_logger

logger = get_logger("clean")


# ============================================================
# 第一层：硬过滤（绝对黑名单，不可豁免）
# ============================================================
# 这些词命中，无论什么情况都干掉。
# 理由：确实没有景点叫这些名字。
HARD_EXCLUDE_NAMES = [
    "停车场", "售票处", "游客中心", "服务中心", "安检",
    "地铁站", "公交站", "出入口", "派出所",
    "公共厕所", "卫生间",
]

HARD_EXCLUDE_TYPES = [
    "地铁站", "公交车站", "停车场", "售票处",
    "公共厕所", "住宿服务",
]


# ============================================================
# 第二层：软过滤（可豁免黑名单）
# ============================================================
# 这些词命中，要看名字里有没有"豁免关键词"。
# 有，就保留；没有，就干掉。
#
# 例：
#   "武汉大学" type 含"学校"，但名字含"大学" → 保留
#   "XX小学" type 含"学校"，名字不含豁免词 → 干掉
SOFT_EXCLUDE_TYPES = [
    "学校", "科教文化服务",     # 武大是景点，XX 小学不是
    "政府机关",                 # 江汉关博物馆可能是旧址
    "公司企业",                 # 有些遗址被标注成企业
    "商务住宅",                 # 有些名人故居被标注成住宅
    "金融保险",                 # 老银行、老钱庄可能是景点
]

# 软过滤的豁免关键词：名字含这些词的，即使命中软黑名单也保留
SOFT_EXEMPT_KEYWORDS = [
    "博物馆", "纪念馆", "旧址", "遗址", "故居",
    "公园", "大学", "学院", "寺", "庙", "塔", "楼",
    "景区", "风景区", "文化", "艺术",
]


# ============================================================
# 工具函数
# ============================================================
def is_valid_poi(poi: dict, local_exclude: list, core_spots_flat: set = None) -> bool:
    """
    判断一条 POI 是否值得保留。

    规则优先级（从上到下）：
    1. 完全匹配核心景点 → 保留（最强豁免，跳过一切）
    2. 硬过滤：名字/类型命中硬黑名单 → 干掉
    3. 有 _interest_group（被核心词抓到）→ 保留
    4. 软过滤：类型命中软黑名单，且名字不含豁免关键词 → 干掉
    5. 品牌店格式（如"黄鹤楼(香港路特许店)"）→ 干掉
    6. 子设施格式（如"归元禅寺-大雄宝殿"）→ 干掉
    7. 城市特有黑名单 → 干掉

    返回 True 表示保留，False 表示干掉。
    """
    name = poi.get("name", "")
    poi_type = poi.get("type", "")

    # ---- 规则 0：完全匹配核心景点 → 最强豁免 ----
    if core_spots_flat and name in core_spots_flat:
        return True

    # ---- 规则 1：硬过滤 ----
    if any(kw in name for kw in HARD_EXCLUDE_NAMES):
        return False
    if any(kw in poi_type for kw in HARD_EXCLUDE_TYPES):
        return False

    # ---- 规则 2：被核心词抓到的 → 保留 ----
    # 走到这一步说明没命中硬过滤，可以放心保留
    if poi.get("_interest_group"):
        return True

    # ---- 规则 3：软过滤 ----
    hit_soft = any(kw in poi_type for kw in SOFT_EXCLUDE_TYPES)
    if hit_soft:
        # 看名字里有没有豁免关键词
        if not any(kw in name for kw in SOFT_EXEMPT_KEYWORDS):
            return False
        # 有豁免关键词，放行

    # ---- 规则 4：品牌店格式 ----
    if re.search(r"\(.*(店|分店|特许).*\)", name):
        return False

    # ---- 规则 5：子设施格式 ----
    if "-" in name:
        suffix = name.split("-")[-1]
        bad_suffix = [
            "殿", "堂", "亭", "塔", "楼", "院", "阁", "池",
            "林", "台", "宫", "像", "雕像", "铜像",
        ]
        if any(kw in suffix for kw in bad_suffix):
            return False

    # ---- 规则 6：城市特有黑名单 ----
    if local_exclude:
        if any(kw in name for kw in local_exclude):
            return False
        if any(kw in poi_type for kw in local_exclude):
            return False

    return True


def dedupe(pois: list) -> list:
    """
    按 id 去重。

    同一个景点可能被抓到多次，比如"黄鹤楼"：
    - 被核心词"黄鹤楼"抓到，_interest_group = "历史人文"
    - 被类别词"风景区"抓到，_interest_group = None

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
        if new_has and not old_has:
            best[pid] = poi
    return list(best.values())


def normalize_category(poi_type: str, interest_group: str = None) -> str:
    """
    把高德的 type 归一化成大类。

    优先级：
    1. 如果 crawl 阶段打了 interest_group，直接用
    2. 否则按 type 里的关键词匹配
    3. 都不匹配，返回"其他"
    """
    if interest_group:
        return interest_group

    if "博物馆" in poi_type or "展览" in poi_type:
        return "历史人文"
    if "寺庙" in poi_type or "宗教" in poi_type or "教堂" in poi_type:
        return "历史人文"
    if "风景" in poi_type or "公园" in poi_type or "广场" in poi_type:
        return "自然风光"
    if "餐饮" in poi_type or "美食" in poi_type:
        return "美食探店"
    if "购物" in poi_type or "商业" in poi_type or "步行街" in poi_type:
        return "购物休闲"
    if "娱乐" in poi_type or "夜" in poi_type:
        return "夜生活"

    return "其他"


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


# ============================================================
# 单条 POI 清洗
# ============================================================
def clean_one(poi: dict) -> dict:
    """
    把一条高德原始 POI，转成 schema 规范的结构。
    """
    business = poi.get("business", {}) or {}
    interest_group = poi.get("_interest_group")

    return {
        "id": poi.get("id", ""),
        "name": poi.get("name", ""),
        "city": poi.get("_city", ""),
        "district": poi.get("adname", "未知"),
        "address": poi.get("address", ""),
        "location": poi.get("location", ""),
        "type": poi.get("type", ""),
        "category": normalize_category(poi.get("type", ""), interest_group),
        "rating": business.get("rating", "暂无") or "暂无",
        "opentime": business.get("opentime_today", "暂无") or "暂无",
        "poi_level": business.get("keytag", "") or "",
        "photos": extract_photos(poi.get("photos", [])),
        # 以下三个字段由 tag 阶段填充
        "tags": "",
        "crowd": "",
        "highlight": "",
        # 系统字段
        "source": "amap",
        "crawled_at": poi.get("_crawled_at", ""),
        "interest_group": interest_group or "",
    }


# ============================================================
# 清洗一个城市
# ============================================================
def clean_city(city_key: str, force: bool = False) -> None:
    """
    清洗一个城市的所有 POI。

    流程：
    1. 读原始数据
    2. 去重
    3. 两层过滤
    4. 单条清洗（字段映射）
    5. 按评分排序 + 截断
    6. 保存
    """
    logger.info("=" * 60)
    logger.info(f"开始清洗城市：{city_key}")
    logger.info("=" * 60)

    in_path = raw_path(city_key)
    out_path = clean_path(city_key)

    if not in_path.exists():
        logger.error(f"输入文件不存在：{in_path}")
        logger.error(f"请先跑 crawl 阶段：python -m src.crawl.amap_crawler --city {city_key}")
        return

    if out_path.exists() and not force:
        logger.info(f"输出文件已存在：{out_path}")
        logger.info("如需重新清洗，加 --force 参数")
        return

    # ---- 读配置 ----
    cfg = get_city_config(city_key)
    local_exclude = cfg.get("local_exclude", []) or []
    max_pois = cfg.get("limits", {}).get("max_pois", 9999)

    # ---- 把核心词拍平成 set，用于豁免 ----
    core_spots_flat = set()
    for interest, spots in cfg.get("core_spots", {}).items():
        core_spots_flat.update(spots)
    logger.info(f"核心词总数：{len(core_spots_flat)}")

    # ---- 读原始数据 ----
    with open(in_path, "r", encoding="utf-8") as f:
        raw_pois = json.load(f)
    logger.info(f"读入原始 POI：{len(raw_pois)} 条")

    # ---- 去重 ----
    deduped = dedupe(raw_pois)
    logger.info(f"去重后：{len(deduped)} 条（去掉 {len(raw_pois) - len(deduped)} 条重复）")

    # ---- 两层过滤 ----
    valid = []
    filtered_hard = 0
    filtered_soft = 0
    filtered_other = 0

    for p in deduped:
        name = p.get("name", "")
        poi_type = p.get("type", "")

        # 统计过滤原因（用于日志）
        if any(kw in name for kw in HARD_EXCLUDE_NAMES) or \
           any(kw in poi_type for kw in HARD_EXCLUDE_TYPES):
            filtered_hard += 1
            continue
        if p.get("_interest_group"):
            valid.append(p)
            continue
        if any(kw in poi_type for kw in SOFT_EXCLUDE_TYPES):
            if not any(kw in name for kw in SOFT_EXEMPT_KEYWORDS):
                filtered_soft += 1
                continue
        # 其他规则：走完整 is_valid_poi 判断
        if is_valid_poi(p, local_exclude, core_spots_flat):
            valid.append(p)
        else:
            filtered_other += 1

    logger.info(f"过滤后：{len(valid)} 条")
    logger.info(f"  - 硬过滤干掉：{filtered_hard} 条")
    logger.info(f"  - 软过滤干掉：{filtered_soft} 条")
    logger.info(f"  - 其他规则干掉：{filtered_other} 条")

    # ---- 单条清洗 ----
    cleaned = [clean_one(p) for p in valid]

    # ---- 按评分排序，超过上限就截断 ----
    def rating_key(p):
        r = p.get("rating", "暂无")
        try:
            return float(r)
        except (ValueError, TypeError):
            return 0.0

    cleaned.sort(key=rating_key, reverse=True)

    if len(cleaned) > max_pois:
        logger.info(f"超过上限 {max_pois} 条，按评分截断")
        cleaned = cleaned[:max_pois]

    # ---- 统计类别分布 ----
    cat_counter = Counter(p["category"] for p in cleaned)
    logger.info("类别分布：")
    for cat, cnt in cat_counter.most_common():
        logger.info(f"  {cat}: {cnt} 条")

    # ---- 保存 ----
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)

    logger.info("")
    logger.info("=" * 60)
    logger.info(f"清洗完成：{len(cleaned)} 条")
    logger.info(f"输出：{out_path}")
    logger.info("=" * 60)


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

    clean_city(args.city, force=args.force)


if __name__ == "__main__":
    main()