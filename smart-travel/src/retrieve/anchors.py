"""
锚点加载模块。

锚点 = 交通枢纽 + 住宿区，从 {city}_anchors_clean.json 读。

用途：
- 空间排序时作为起点（Day1 从出发地出发，Day2+ 从住宿区出发）
- 前端表单的"出发地"和"住宿区域"下拉选项

为什么单独一个模块：
锚点不是"检索结果"，是"空间排序的输入"。
它不该进 Retriever，Retriever 只管 POI 检索。
"""

import json
from pathlib import Path

from src.config import anchors_clean_path
from src.logger import get_logger

logger = get_logger("anchors")


# ============================================================
# 缓存
# ============================================================
# 锚点数据在运行期间不变，缓存起来避免重复读文件
_cache: dict[str, dict] = {}


# ============================================================
# 加载
# ============================================================
def load_anchors(city_key: str) -> dict:
    """
    加载某个城市的锚点数据。

    Args:
        city_key: 城市 key，如 "wuhan"

    Returns:
        {
            "transport_hubs": [{"name": "武汉站", "lat": 30.6, "lng": 114.4}, ...],
            "stay_districts": [{"name": "武昌区", "lat": 30.5, "lng": 114.3}, ...]
        }

    如果文件不存在，返回空列表（不抛异常，让调用方自己判断）。
    """
    if city_key in _cache:
        return _cache[city_key]

    path = anchors_clean_path(city_key)

    if not path.exists():
        logger.warning(f"锚点文件不存在：{path}")
        result = {"transport_hubs": [], "stay_districts": []}
        _cache[city_key] = result
        return result

    with open(path, "r", encoding="utf-8") as f:
        anchors = json.load(f)

    # 按 type 分组
    hubs = [a for a in anchors if a.get("type") == "transport_hub"]
    districts = [a for a in anchors if a.get("type") == "stay_district"]

    result = {
        "transport_hubs": hubs,
        "stay_districts": districts,
    }

    logger.info(
        f"加载锚点（{city_key}）："
        f"交通枢纽 {len(hubs)} 个，住宿区 {len(districts)} 个"
    )

    _cache[city_key] = result
    return result


def find_anchor(anchors: dict, name: str) -> dict | None:
    """
    按名字查找锚点。

    Args:
        anchors: load_anchors 返回的字典
        name: 锚点名，如"武汉站"、"武昌区"

    Returns:
        匹配的锚点字典，找不到返回 None
    """
    for group in ["transport_hubs", "stay_districts"]:
        for item in anchors.get(group, []):
            if item.get("name") == name:
                return item
    return None
