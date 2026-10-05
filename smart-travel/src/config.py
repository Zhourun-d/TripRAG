"""
配置加载模块。

职责：
1. 读取 config/cities/ 目录下所有城市 yaml，和 config/schema.yaml
2. 提供统一的配置访问接口
3. 统一管理数据路径

设计说明：
- 城市配置不做 defaults 继承，每个城市一个文件，写全
- get_city_config 直接返回该城市的配置块，不做合并
- schema 相关访问保持简单，一次读文件，缓存复用

为什么单独抽一个模块：
- 避免每个脚本都写一遍 yaml.load
- 配置路径统一，改目录只改这里
"""

from pathlib import Path
from functools import lru_cache
from typing import Any

import yaml
from dotenv import load_dotenv


# ============================================================
# 路径常量
# ============================================================
PROJECT_ROOT = Path(__file__).parent.parent

# 加载 .env —— 必须在所有 os.getenv 之前执行
load_dotenv(PROJECT_ROOT / ".env")

CONFIG_DIR = PROJECT_ROOT / "config"
CITIES_DIR = CONFIG_DIR / "cities"
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
LOG_DIR = PROJECT_ROOT / "logs"

# 确保运行时目录存在
for d in [DATA_DIR, CHROMA_DIR, LOG_DIR]:
    d.mkdir(exist_ok=True)


# ============================================================
# yaml 读取（带缓存）
# ============================================================
@lru_cache(maxsize=1)
def load_cities() -> dict:
    """
    读取 config/cities/ 目录下所有城市 yaml。

    每个城市一个文件（wuhan.yaml / chongqing.yaml / ...），
    文件名（不含后缀）就是 city_key。

    以 _ 开头的文件会被跳过（用于放草稿、备份）。

    Returns:
        {city_key: config, ...}
        例：{"wuhan": {...}, "chongqing": {...}}
    """
    if not CITIES_DIR.exists():
        raise FileNotFoundError(f"找不到城市配置目录：{CITIES_DIR}")

    result = {}
    for path in sorted(CITIES_DIR.glob("*.yaml")):
        if path.stem.startswith("_"):
            continue
        city_key = path.stem  # wuhan.yaml → wuhan
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        result[city_key] = cfg

    if not result:
        raise FileNotFoundError(f"城市配置目录为空：{CITIES_DIR}")

    return result


@lru_cache(maxsize=1)
def load_schema() -> dict:
    """读取 schema.yaml"""
    path = CONFIG_DIR / "schema.yaml"
    if not path.exists():
        raise FileNotFoundError(f"找不到 schema 配置：{path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ============================================================
# 城市配置访问
# ============================================================
def get_city_config(city_key: str) -> dict[str, Any]:
    """
    获取某个城市的完整配置。

    不做 defaults 合并——每个城市在 yaml 里写全，直接返回即可。

    Args:
        city_key: 城市 key，如 "wuhan"

    Returns:
        该城市的配置字典

    Raises:
        KeyError: 城市不存在时抛出
    """
    cities = load_cities()
    city_cfg = cities.get(city_key)

    if city_cfg is None:
        raise KeyError(f"未知城市：{city_key}。可选：{list(cities.keys())}")

    return city_cfg


def list_cities() -> list[str]:
    """列出所有已配置的城市 key"""
    return list(load_cities().keys())


def get_transport_hubs(city_key: str) -> list[str]:
    """
    获取交通枢纽名字列表。

    用途：表单的"出发地"选项；空间排序时作为 Day1 起点。

    注意：只返回名字，坐标从 {city}_anchors_clean.json 读
    （见 src/retrieve/anchors.py 的 load_anchors）。

    Returns:
        ["武汉站", "汉口站", "武昌站", "武汉东站", "天河机场"]
    """
    cfg = get_city_config(city_key)
    return cfg.get("transport_hubs", []) or []


def get_stay_districts(city_key: str) -> list[str]:
    """
    获取住宿区名字列表。

    用途：表单的"住宿区域"选项；空间排序时作为 Day2+ 起点。

    注意：只返回名字，坐标从 {city}_anchors_clean.json 读。

    Returns:
        ["武昌区", "汉阳区", "青山区", ...]
    """
    cfg = get_city_config(city_key)
    return cfg.get("stay_districts", []) or []


def get_clean_rules(city_key: str) -> dict:
    """
    获取清洗规则。

    包含硬过滤、软过滤、豁免词、品牌店正则、子设施后缀、城市黑名单。
    全部来自城市 yaml 的 clean_rules 块。
    """
    cfg = get_city_config(city_key)
    return cfg.get("clean_rules", {}) or {}


def get_spatial_config(city_key: str) -> dict:
    """
    获取空间排序参数。

    用于 spatial_sorter.py，把候选 POI 按地理位置分成 N 天。
    全部来自城市 yaml 的 spatial 块。

    Returns:
        {
            "cluster_k": 12,
            "cluster_max_size": 5,
            "cluster_max_dist": 8.0,
            "min_pts_per_day": 6,
            "urban_districts": [...],
        }

    Raises:
        ValueError: urban_districts 缺失或为空时抛出。
                    它是城郊分离的必要依据，缺了会让所有簇被判为郊区，
                    最终 plan 为空。
    """
    cfg = get_city_config(city_key)
    spatial = cfg.get("spatial", {}) or {}
    merged = {**SPATIAL_DEFAULTS, **spatial}

    if not merged.get("urban_districts"):
        raise ValueError(
            f"[{city_key}] spatial.urban_districts 缺失或为空。"
            f"它是城郊分离的必要依据，请在 config/cities/{city_key}.yaml 里补全。"
        )

    return merged


# 空间排序参数的默认值。
# 城市 yaml 里没写全时，用这些值兜底。
# 注意：urban_districts 不在默认值里，必须由城市 yaml 提供（见上）。
SPATIAL_DEFAULTS = {
    "cluster_k": 12,
    "cluster_max_size": 5,
    "cluster_max_dist": 8.0,
    "min_pts_per_day": 6,
}


# ============================================================
# schema 访问
# ============================================================
def get_poi_schema() -> dict:
    """获取 POI 字段规范"""
    return load_schema().get("poi", {})


def get_interest_tags() -> list[str]:
    """获取兴趣大类词表（LLM 打 tags 的候选集）"""
    tags = load_schema().get("interest_tags", [])
    if not tags:
        raise ValueError("schema.yaml 缺少 interest_tags")
    return tags


def get_behavior_tags() -> list[str]:
    """获取体验词表（LLM 打 behaviors 的候选集）"""
    tags = load_schema().get("behavior_tags", [])
    if not tags:
        raise ValueError("schema.yaml 缺少 behavior_tags")
    return tags


def get_crowd_to_tags() -> dict:
    """
    获取同行人 → 体验词映射。

    Returns:
        {"情侣": ["散步", "拍照", ...], "朋友": [...], ...}
    """
    mapping = load_schema().get("crowd_to_tags", {})
    if not mapping:
        raise ValueError("schema.yaml 缺少 crowd_to_tags")
    return mapping


def get_text_template() -> str:
    """获取向量化文本模板"""
    template = load_schema().get("text_template", "")
    if not template:
        raise ValueError("schema.yaml 缺少 text_template")
    return template


# ============================================================
# 路径工具
# ============================================================
# 命名约定：
#   {city}_pois.json          - 抓取的景点原始数据
#   {city}_anchors.json       - 抓取的锚点（交通枢纽 + 住宿区）原始数据
#   {city}_clean.json         - 清洗后的景点
#   {city}_anchors_clean.json - 清洗后的锚点
#   {city}_tagged.json        - 打标签后的景点
#   {city}_v1.jsonl           - 知识库存档
# ============================================================

def raw_path(city_key: str) -> Path:
    """景点原始数据路径"""
    return DATA_DIR / "raw" / f"{city_key}_pois.json"


def anchors_raw_path(city_key: str) -> Path:
    """锚点原始数据路径（交通枢纽 + 住宿区）"""
    return DATA_DIR / "raw" / f"{city_key}_anchors.json"


def clean_path(city_key: str) -> Path:
    """景点清洗后数据路径"""
    return DATA_DIR / "processed" / f"{city_key}_clean.json"


def anchors_clean_path(city_key: str) -> Path:
    """锚点清洗后数据路径"""
    return DATA_DIR / "processed" / f"{city_key}_anchors_clean.json"


def tagged_path(city_key: str) -> Path:
    """景点打标签后数据路径"""
    return DATA_DIR / "processed" / f"{city_key}_tagged.json"


def kb_path(city_key: str) -> Path:
    """知识库 jsonl 存档路径"""
    return DATA_DIR / "knowledge_base" / f"{city_key}_v1.jsonl"


def eval_path(city_key: str) -> Path:
    """评估集路径"""
    return DATA_DIR / "eval" / f"{city_key}_eval.json"


def report_path(city_key: str) -> Path:
    """评估报告路径"""
    return DATA_DIR / "eval" / "reports" / f"{city_key}_report.json"