"""
片内选点模块。

职责：
从一片候选 POI 里，按方案类型（综合/口碑/悠闲）挑 N 个点。

输入：
- day_pois: 一天的候选，list[poi_dict]（来自 spatial_sorter）
- plan_type: "综合" / "口碑" / "悠闲"
- interests: 用户选的兴趣列表，如 ["历史人文", "自然风光"]
- companion: 同行人，如 "情侣"（可选，None 表示未选）
- city_key: 城市 key，用于加载 core_spots
- start_lat/lng: 排序起点（住宿区或上一片终点）
- seed: 随机种子（可选，用于复现）

输出：
- 选中的 poi_dict 列表，已按地理顺序排好

算法：
1. 每个候选算基础分：
   base = w_rating×rating_norm + w_interest×interest_match
        + w_random×random + w_core×is_core
        + w_behavior×behavior_norm
2. behavior 加成（按 companion 映射）：
   behavior_norm = (0.7×max + 0.3×mean) / max_weight
3. 逐个抽 N 个：
   - 每轮根据"离已选点最近距离"打折
   - 按打折后的分数加权随机抽
4. 按地理贪心排序

距离惩罚（分段）：
- 离最近已选点 < 0.5km  → 分数 × 0.1
- < 1.5km              → 分数 × 0.6
- < 2.0km              → 分数 × 0.8
- 否则                  → 不变
- core_spot：惩罚减半（往 1.0 方向拉）

和 spatial_sorter 的分工：
- spatial_sorter：地理单位，负责「分片」（哪些点归哪一天）
- poi_picker：采样单位，负责「片内选点」（从候选里选 N 个）
"""

import argparse
import math
import random

from src.config import (
    get_city_config,
    get_crowd_to_tags,
)
from src.logger import get_logger

logger = get_logger("picker")


# ============================================================
# 配置：三套方案的权重和目标点数
# ============================================================
PLAN_CONFIG = {
    "综合": {
        "w_rating": 0.15,
        "w_interest": 0.1,
        "w_random": 0.15,
        "w_core": 0.3,
        "w_behavior": 0.3,
        "w_level": 0.0,
        "target_count": 5,
    },
    "口碑": {
        "w_rating": 0.25,
        "w_interest": 0.05,
        "w_random": 0.1,
        "w_core": 0.3,
        "w_behavior": 0.15,
        "w_level": 0.15,
        "target_count": 5,
    },
    "悠闲": {
        "w_rating": 0.1,
        "w_interest": 0.1,
        "w_random": 0.25,
        "w_core": 0.25,
        "w_behavior": 0.3,
        "w_level": 0.0,
        "target_count": 4,
    },
}

# 距离惩罚的分段阈值（km）
PENALTY_NEAR = 0.5              # < 0.5km → ×0.1
PENALTY_MID = 1.5               # < 1.5km → ×0.6
PENALTY_FAR = 2.0               # < 2.0km → ×0.8
PENALTY_NEAR_FACTOR = 0.1
PENALTY_MID_FACTOR = 0.6
PENALTY_FAR_FACTOR = 0.8

# 无评分的 rating 归一化值（当「中等偏上」）
NO_RATING_NORM = 0.6

# behavior 加成的权重分配
BEHAVIOR_MAX_WEIGHT = 0.7
BEHAVIOR_MEAN_WEIGHT = 0.3

# 地球半径（km）
EARTH_RADIUS_KM = 6371.0

# 可视化配色（一天一色）
DAY_COLORS = ["red", "blue", "green", "purple", "orange",
              "darkred", "darkblue", "darkgreen", "cadetblue", "darkpurple"]


# ============================================================
# 距离计算
# ============================================================
def haversine(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Haversine 距离，单位 km。"""
    lat1_r, lat2_r = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlng / 2) ** 2)
    return EARTH_RADIUS_KM * 2 * math.asin(math.sqrt(a))


# ============================================================
# core_spots 加载与匹配
# ============================================================
def load_core_spots(city_key: str) -> set:
    """从 cities/{city}.yaml 读 core_spots，拍平成 set。"""
    cfg = get_city_config(city_key)
    core_spots = cfg.get("core_spots", {}) or {}

    flat = set()
    for spots in core_spots.values():
        flat.update(spots)

    logger.info(f"[picker] 加载 core_spots：{len(flat)} 个")
    return flat


def is_core_spot(poi_name: str, core_spots_flat: set) -> bool:
    """包含匹配：core_spots 里的词只要出现在 poi_name 里，就算命中。"""
    if not poi_name:
        return False
    return any(core in poi_name for core in core_spots_flat)


# ============================================================
# 各维度打分
# ============================================================
def _rating_norm(poi: dict) -> float:
    """评分归一化到 0~1。无评分返回 NO_RATING_NORM。"""
    rating = poi.get("rating", -1.0)
    try:
        rating = float(rating)
    except (ValueError, TypeError):
        rating = -1.0

    if rating < 0:
        return NO_RATING_NORM
    return min(rating / 5.0, 1.0)


def _interest_match(poi: dict, interests: list) -> float:
    """兴趣匹配度，0~1。用 tags 命中数 / interests 总数。"""
    if not interests:
        return 0.5

    tags_raw = poi.get("tags", "")
    if not tags_raw:
        return 0.0

    tags = [
        t.strip()
        for t in str(tags_raw).replace("，", ",").split(",")
        if t.strip()
    ]

    hit_count = sum(1 for t in tags if t in interests)
    return min(hit_count / len(interests), 1.0)


def _behavior_norm(poi: dict, companion: str, crowd_to_tags: dict) -> float:
    """
    behavior 加成归一化到 0~1。

    公式：
        matched = [mapping[b] for b in poi.behaviors if b in mapping]
        若为空 → 0
        否则   → (0.7×max + 0.3×mean) / max(mapping.values())
    """
    if not companion:
        return 0.0

    mapping = crowd_to_tags.get(companion, {})
    if not mapping:
        return 0.0

    behaviors_raw = poi.get("behaviors", "")
    if not behaviors_raw:
        return 0.0

    behaviors = [
        b.strip()
        for b in str(behaviors_raw).replace("，", ",").split(",")
        if b.strip()
    ]

    matched = [mapping[b] for b in behaviors if b in mapping]
    if not matched:
        return 0.0

    max_w = max(mapping.values())
    if max_w <= 0:
        return 0.0

    score = BEHAVIOR_MAX_WEIGHT * max(matched) + BEHAVIOR_MEAN_WEIGHT * (sum(matched) / len(matched))
    return min(score / max_w, 1.0)


# 景区等级加成：5A / 4A 满分，其余（含 3A）不计
LEVEL_SCORE = {"5A": 1.0, "4A": 1.0}


def _level_norm(poi: dict) -> float:
    """
    景区等级归一化。

    5A / 4A 得满分 1.0，3A 及其它不计 0.0。
    说明：口碑里 5A 是「必选」（不参与随机），所以 w_level 实际主要给 4A 加成；
    3A 按用户要求丢弃（3A 景点太普遍，没有区分度）。
    """
    return LEVEL_SCORE.get(str(poi.get("level", "") or ""), 0.0)


def _base_score(
    poi: dict,
    interests: list,
    companion: str,
    cfg: dict,
    core_spots_flat: set,
    crowd_to_tags: dict,
    rng: random.Random,
) -> float:
    """基础分 = 6 项加权和。"""
    return (
        cfg["w_rating"] * _rating_norm(poi)
        + cfg["w_interest"] * _interest_match(poi, interests)
        + cfg["w_random"] * rng.random()
        + cfg["w_core"] * (1.0 if is_core_spot(poi.get("name", ""), core_spots_flat) else 0.0)
        + cfg["w_behavior"] * _behavior_norm(poi, companion, crowd_to_tags)
        + cfg.get("w_level", 0.0) * _level_norm(poi)
    )


# ============================================================
# 距离惩罚
# ============================================================
def _penalty_factor(poi: dict, picked: list, is_core: bool = False) -> float:
    """
    距离惩罚系数。

    - picked 为空 → 1.0
    - 离最近已选点 < 0.5km → 0.1
    - < 1.5km → 0.6
    - < 2.0km → 0.8
    - 否则 → 1.0

    is_core=True 时，惩罚减半（往 1.0 方向拉）。
    """
    if not picked:
        return 1.0

    p_lat, p_lng = poi["lat"], poi["lng"]
    min_d = min(
        haversine(p_lat, p_lng, q["lat"], q["lng"])
        for q in picked
    )

    if min_d < PENALTY_NEAR:
        factor = PENALTY_NEAR_FACTOR
    elif min_d < PENALTY_MID:
        factor = PENALTY_MID_FACTOR
    elif min_d < PENALTY_FAR:
        factor = PENALTY_FAR_FACTOR
    else:
        return 1.0

    if is_core:
        factor = factor + (1.0 - factor) * 0.5

    return factor


# ============================================================
# 加权随机抽
# ============================================================
def _weighted_sample(candidates: list, weights: list, rng: random.Random) -> int:
    """按权重随机抽一个，返回下标。全 0 时退化为均匀随机。"""
    total = sum(weights)
    if total <= 0:
        return rng.randrange(len(candidates))

    r = rng.random() * total
    acc = 0.0
    for i, w in enumerate(weights):
        acc += w
        if r <= acc:
            return i
    return len(candidates) - 1


# ============================================================
# 地理贪心排序
# ============================================================
def _sort_by_geo(picked: list, start_lat: float, start_lng: float) -> list:
    """从 start 出发，每次选最近的下一个。"""
    if not picked:
        return []

    remaining = list(picked)
    ordered = []
    cur_lat, cur_lng = start_lat, start_lng

    while remaining:
        nearest_idx = min(
            range(len(remaining)),
            key=lambda i: haversine(
                cur_lat, cur_lng,
                remaining[i]["lat"], remaining[i]["lng"],
            ),
        )
        nearest = remaining.pop(nearest_idx)
        ordered.append(nearest)
        cur_lat, cur_lng = nearest["lat"], nearest["lng"]

    return ordered


# ============================================================
# 主接口
# ============================================================
def pick(
    day_pois: list,
    plan_type: str,
    interests: list,
    companion: str,
    city_key: str,
    start_lat: float,
    start_lng: float,
    seed: int = None,
    day_index: int = 0,
    debug_info: list = None,
) -> list:
    """
    从一天的候选里挑 N 个点。

    debug_info 不为 None 时，会往里面 append 一条打分详情，供外部打印。

    seed 按 (plan_type, day_index) 偏移，避免三套方案/多天共用同一条随机流
    （否则「悠闲」会退化成「综合」砍掉一个点）。
    """
    if plan_type not in PLAN_CONFIG:
        logger.warning(f"未知 plan_type：{plan_type}，退回「综合」")
        plan_type = "综合"

    cfg = PLAN_CONFIG[plan_type]
    target = cfg["target_count"]

    if len(day_pois) <= target:
        logger.info(f"[{plan_type}] 候选 {len(day_pois)} ≤ 目标 {target}，全选")
        return _sort_by_geo(list(day_pois), start_lat, start_lng)

    core_spots_flat = load_core_spots(city_key)
    crowd_to_tags = get_crowd_to_tags()

    # seed 按「方案 + 天」偏移，消除同源随机
    if seed is None:
        rng = random.Random()
    else:
        rng = random.Random(f"{seed}|{plan_type}|{day_index}")

    # ---- 1. 算每个候选的基础分 ----
    candidates = list(day_pois)
    bases = [
        _base_score(p, interests, companion, cfg, core_spots_flat, crowd_to_tags, rng)
        for p in candidates
    ]

    picked = []
    picked_idx = set()
    rounds_log = []

    # ---- 2. 口碑：5A 必选，不参与随机 ----
    if plan_type == "口碑":
        five = [i for i, p in enumerate(candidates) if p.get("level") == "5A"]
        if len(five) > target:
            five = sorted(five, key=lambda i: -_rating_norm(candidates[i]))[:target]
        for i in five:
            picked.append(candidates[i])
            picked_idx.add(i)
        if five:
            logger.info(
                f"[口碑] 锁定 5A（{len(five)} 个）："
                f"{[candidates[i]['name'] for i in five]}"
            )

    # ---- 3. 剩余槽位逐个挑 ----
    for round_i in range(target - len(picked)):
        weights = []
        for i, p in enumerate(candidates):
            if i in picked_idx:
                weights.append(0.0)
                continue
            is_core = is_core_spot(p.get("name", ""), core_spots_flat)
            factor = _penalty_factor(p, picked, is_core=is_core)
            weights.append(bases[i] * factor)

        # 记录本轮 top5 权重
        top5 = sorted(
            [(candidates[i]["name"], weights[i]) for i in range(len(candidates))
             if i not in picked_idx],
            key=lambda x: -x[1]
        )[:5]

        chosen_i = _weighted_sample(candidates, weights, rng)

        if chosen_i in picked_idx:
            available = [i for i in range(len(candidates)) if i not in picked_idx]
            if not available:
                break
            chosen_i = rng.choice(available)

        picked.append(candidates[chosen_i])
        picked_idx.add(chosen_i)

        rounds_log.append({
            "round": round_i + 1,
            "picked": candidates[chosen_i]["name"],
            "top5": top5,
        })

    # ---- 4. 按地理排序 ----
    ordered = _sort_by_geo(picked, start_lat, start_lng)

    logger.info(
        f"[{plan_type}] 候选 {len(day_pois)} → 选中 {len(ordered)}："
        f"{[p['name'] for p in ordered]}"
    )

    # ---- 4. debug 信息 ----
    if debug_info is not None:
        debug_info.append({
            "plan_type": plan_type,
            "candidates": [
                {
                    "name": p["name"],
                    "base": bases[i],
                    "rating_norm": _rating_norm(p),
                    "interest_match": _interest_match(p, interests),
                    "behavior_norm": _behavior_norm(p, companion, crowd_to_tags),
                    "is_core": is_core_spot(p.get("name", ""), core_spots_flat),
                    "picked": p in picked,
                }
                for i, p in enumerate(candidates)
            ],
            "rounds": rounds_log,
        })

    return ordered


# ============================================================
# debug 打印
# ============================================================
def _print_debug(day: str, info: dict):
    """打印一天的选点详情。"""
    print(f"\n{'─' * 90}")
    print(f"{day} 选点详情（{info['plan_type']}）")
    print(f"{'─' * 90}")
    print(f"{'名称':<24} {'base':>7} {'rating':>7} {'interest':>9} "
          f"{'behavior':>9} {'core':>6} {'选中':>5}")
    print(f"{'─' * 90}")

    sorted_cands = sorted(info["candidates"], key=lambda x: -x["base"])
    for c in sorted_cands:
        marker = "√" if c["picked"] else ""
        print(f"{c['name'][:22]:<24} {c['base']:>7.4f} {c['rating_norm']:>7.3f} "
              f"{c['interest_match']:>9.3f} {c['behavior_norm']:>9.3f} "
              f"{str(c['is_core']):>6} {marker:>5}")

    print(f"\n每轮抽取：")
    for r in info["rounds"]:
        top5_str = ", ".join(f"{n}({w:.3f})" for n, w in r["top5"])
        print(f"  Round {r['round']}: 选中 [{r['picked']}]  | top5: {top5_str}")


# ============================================================
# 可视化
# ============================================================
def _visualize(picked_plan: dict, plan: dict, stay_lat: float, stay_lng: float,
               out_path: str):
    """画 folium 地图：选中点大圆 + 数字，候选点小圆，住宿区 house 图标。"""
    try:
        import folium
    except ImportError:
        logger.warning("未安装 folium，跳过可视化")
        return

    all_pts = [p for pois in plan.values() for p in pois]
    if not all_pts:
        logger.warning("没有点，跳过可视化")
        return

    center = [
        sum(p["lat"] for p in all_pts) / len(all_pts),
        sum(p["lng"] for p in all_pts) / len(all_pts),
    ]

    m = folium.Map(location=center, zoom_start=11)

    folium.Marker(
        location=[stay_lat, stay_lng],
        popup="住宿区",
        icon=folium.Icon(color="black", icon="home", prefix="fa"),
    ).add_to(m)

    for day_idx, (day, day_pois) in enumerate(plan.items()):
        color = DAY_COLORS[day_idx % len(DAY_COLORS)]
        picked = picked_plan.get(day, [])
        picked_names = {p["name"] for p in picked}

        # 候选点：小圆，半透明
        for p in day_pois:
            if p["name"] in picked_names:
                continue
            folium.CircleMarker(
                location=[p["lat"], p["lng"]],
                radius=4, color=color, fill=True, fillColor=color,
                fillOpacity=0.3,
                popup=f"{day} 候选 | {p['name']} | rating={p['rating']}",
            ).add_to(m)

        # 选中点连线
        if len(picked) >= 2:
            folium.PolyLine(
                locations=[[p["lat"], p["lng"]] for p in picked],
                color=color, weight=3, opacity=0.7,
            ).add_to(m)

        # 选中点：大圆 + 数字
        for i, p in enumerate(picked, 1):
            folium.CircleMarker(
                location=[p["lat"], p["lng"]],
                radius=10, color=color, fill=True, fillColor=color,
                fillOpacity=0.9,
                popup=f"{day} #{i} | {p['name']} | rating={p['rating']}",
                tooltip=f"{day} #{i}: {p['name']}",
            ).add_to(m)
            folium.Marker(
                location=[p["lat"], p["lng"]],
                icon=folium.DivIcon(
                    html=f'<div style="font-size:11px;font-weight:bold;'
                         f'color:white;text-align:center;line-height:20px;">'
                         f'{i}</div>',
                    icon_size=(20, 20),
                    icon_anchor=(10, 10),
                ),
            ).add_to(m)

    m.save(out_path)
    logger.info(f"地图已保存：{out_path}")


# ============================================================
# 命令行测试入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="片内选点测试")
    parser.add_argument("--city", default="wuhan", help="城市 key")
    parser.add_argument("--city_name", default="武汉", help="城市名")
    parser.add_argument("--query", default=None,
                        help="自定义 query（不传则按 interests+companion 拼）")
    parser.add_argument("--top_k", type=int, default=60)
    parser.add_argument("--days", type=int, default=3)
    parser.add_argument("--stay_district", default="武昌区")
    parser.add_argument("--plan_type", default="综合",
                        choices=["综合", "口碑", "悠闲"])
    parser.add_argument("--interests", nargs="*",
                        default=["历史人文", "自然风光"])
    parser.add_argument("--companion", default=None,
                        help="同行人：情侣 / 亲子 / 朋友 / 独行 / 家庭")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # ---- 1. 检索 ----
    from src.retrieve.retriever import Retriever
    from src.plan.spatial_sorter import spatial_sort, get_stay_coord

    retriever = Retriever(f"{args.city}_v1")

    crowd_to_tags = get_crowd_to_tags()
    if args.query:
        query = args.query
    else:
        behavior_words = list(crowd_to_tags.get(args.companion, {}).keys()) if args.companion else []
        query = " ".join(args.interests + behavior_words)
    print(f"\nQuery: {query}")

    results = retriever.search(
        query=query,
        city=args.city_name,
        top_k=args.top_k,
        mode="hybrid",
    )
    print(f"检索召回：{len(results)} 条")

    # ---- 2. 空间排序 ----
    plan, dropped = spatial_sort(
        results=results,
        days=args.days,
        stay_district=args.stay_district,
        city_key=args.city,
        seed=args.seed,
    )

    # ---- 3. 每片选点 ----
    stay_lat, stay_lng = get_stay_coord(args.stay_district, args.city)

    picked_plan = {}

    for day_idx, (day, day_pois) in enumerate(plan.items()):
        print(f"\n{'=' * 90}")
        print(f"{day} 候选（{len(day_pois)} 个）：")
        for p in day_pois:
            print(f"  [{p.get('_cluster_id', '?')}] {p['name']} ({p['district']}) "
                  f"rating={p['rating']} tags={p.get('tags', '')} behaviors={p.get('behaviors', '')}")

        debug_info = []
        picked = pick(
            day_pois=day_pois,
            plan_type=args.plan_type,
            interests=args.interests,
            companion=args.companion,
            city_key=args.city,
            start_lat=stay_lat,
            start_lng=stay_lng,
            seed=args.seed,
            day_index=day_idx,
            debug_info=debug_info,
        )

        print(f"\n{day} 选中（{len(picked)} 个，{args.plan_type}）：")
        for i, p in enumerate(picked, 1):
            print(f"  {i}. {p['name']} ({p['district']}) "
                  f"rating={p['rating']} tags={p.get('tags', '')} behaviors={p.get('behaviors', '')}")

        picked_plan[day] = picked

        if debug_info:
            _print_debug(day, debug_info[0])

    # ---- 4. 可视化 ----
    out_html = f"picker_{args.city}_{args.plan_type}"
    if args.companion:
        out_html += f"_{args.companion}"
    out_html += ".html"

    _visualize(picked_plan, plan, stay_lat, stay_lng, out_html)


if __name__ == "__main__":
    main()