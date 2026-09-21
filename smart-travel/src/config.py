"""
配置加载模块。

职责：
1. 读取 config/cities.yaml 和 config/schema.yaml
2. 合并 defaults 和 city 配置
3. 提供统一的配置访问接口

为什么单独抽一个模块：
- 避免每个脚本都写一遍 yaml.load
- 配置路径统一，改目录只改这里
- 合并 defaults 的逻辑只写一次
"""
from pathlib import Path
from functools import lru_cache
from typing import Any

import yaml
from dotenv import load_dotenv


# ============================================================
# 路径常量
# ============================================================
# __file__ 是当前文件路径（src/config.py）
# .parent 是 src/
# .parent.parent 是项目根目录

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# 加载 .env —— 必须在所有 os.getenv 之前执行
load_dotenv(PROJECT_ROOT / ".env")

CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
LOG_DIR = PROJECT_ROOT / "logs"

# 确保运行时目录存在
for d in [DATA_DIR, CHROMA_DIR, LOG_DIR]:
    d.mkdir(exist_ok=True)


# ============================================================
# yaml 读取
# ============================================================
# lru_cache 的作用：同一个文件只读一次，后续调用直接返回缓存
# 为什么需要：配置在程序运行期间不会变，反复读文件是浪费
@lru_cache(maxsize=1)
def load_cities() -> dict:
    """读取 cities.yaml"""
    path = CONFIG_DIR / "cities.yaml"
    if not path.exists():
        raise FileNotFoundError(f"找不到城市配置：{path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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
    获取某个城市的完整配置（defaults + city 合并）。

    Args:
        city_key: 城市 key，如 "wuhan"

    Returns:
        合并后的配置字典

    合并规则：
        - defaults 里的字段作为基础
        - city 里的字段覆盖 defaults
        - 嵌套字典（如 crawl）做浅合并
    """
    cities = load_cities()
    defaults = cities.get("defaults", {})
    city_cfg = cities.get("cities", {}).get(city_key)

    if city_cfg is None:
        available = list(cities.get("cities", {}).keys())
        raise KeyError(f"未知城市：{city_key}。可选：{available}")

    # 浅合并：city 覆盖 defaults
    # 对于 crawl、limits 这种嵌套字典，也要合并
    merged = {**defaults, **city_cfg}

    # 嵌套字段单独合并
    for nested_key in ["crawl", "limits"]:
        if nested_key in defaults:
            merged[nested_key] = {
                **defaults.get(nested_key, {}),
                **city_cfg.get(nested_key, {}),
            }

    return merged


def list_cities() -> list[str]:
    """列出所有已配置的城市 key"""
    cities = load_cities()
    return list(cities.get("cities", {}).keys())


# ============================================================
# schema 访问
# ============================================================
def get_poi_schema() -> dict:
    """获取 POI 字段规范"""
    return load_schema().get("poi", {})


def get_interest_tags() -> list[str]:
    """获取固定兴趣标签词表"""
    return load_schema().get("interest_tags", [])


def get_crowd_tags() -> list[str]:
    """获取固定人群标签词表"""
    return load_schema().get("crowd_tags", [])


def get_text_template() -> str:
    """获取向量化文本模板"""
    return load_schema().get("text_template", "")


# ============================================================
# 路径工具
# ============================================================
def raw_path(city_key: str) -> Path:
    """原始抓取数据路径"""
    return DATA_DIR / "raw" / f"{city_key}_pois.json"


def clean_path(city_key: str) -> Path:
    """清洗后数据路径"""
    return DATA_DIR / "processed" / f"{city_key}_clean.json"


def tagged_path(city_key: str) -> Path:
    """打标签后数据路径"""
    return DATA_DIR / "processed" / f"{city_key}_tagged.json"


def kb_path(city_key: str) -> Path:
    """知识库 jsonl 路径"""
    return DATA_DIR / "knowledge_base" / f"{city_key}_v1.jsonl"


def eval_path(city_key: str) -> Path:
    """评估集路径"""
    return DATA_DIR / "eval" / f"{city_key}_eval.json"


def report_path(city_key: str) -> Path:
    """评估报告路径"""
    return DATA_DIR / "eval" / "reports" / f"{city_key}_report.json"