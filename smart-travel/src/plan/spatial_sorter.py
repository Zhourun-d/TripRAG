"""
空间排序模块。

职责：
把候选 POI 列表，按地理位置分成 N 天，并对每天排序。

输入：
- 候选 POI 列表（SearchResult，带 lat/lng/tags/behaviors）
- 天数 days
- 住宿区坐标（stay_lat, stay_lng）
- 城市空间参数（从 cities/{city}.yaml 的 spatial 块读）

输出：
{
    "Day1": [poi_dict, ...],
    "Day2": [...],
    ...
}
每个 poi_dict 含 name/lat/lng/tags/behaviors/rating 等，
并带 _cluster_id（属于哪个小簇），供片内选点参考。

核心流程（v4 定案）：
    1. 严格去重 + 过滤无效坐标
    2. 切小簇：KMeans 聚 k 簇，每簇超 max_size 递归拆，
       每簇内两点最远超 max_dist 再拆
    3. 单点吸收：城区单点簇吸进离它最近的城区多簇
    4. 城郊分离：城区簇参与聚片，郊区簇搁置
    5. KMeans 聚片：对城区簇的簇心聚 days 片
    6. 纠错：每簇归到最近片，先算归属再统一更新，循环到稳定
    7. 补簇：片点数 < min_pts_per_day 的，从就近且搬得起的片挪
    8. 郊区成片：还有天数额度就给郊区一片
    9. 片排序：Day1 = 离住宿区最近的片，之后贪心找最近片
    10. 输出

关于「城/郊」判定：
    用区名表（cities/{city}.yaml 的 urban_districts），不用距离阈值。
    原因：距离阈值要调，设宽了把远郊算成近郊，设窄了又把城区算远郊。
    主城区是地理事实，写死更可靠，换城市补一份即可。

关于「跨江」：
    不特殊处理。武汉地铁过江方便，跨江点（如归元禅寺）
    和普通点一样参与分片。

关于「簇 ≠ 天」：
    簇是地理单位，片才是天。一天 = 1~N 个簇的组合。
"""

import math
from typing import Optional

import numpy as np
from sklearn.cluster import KMeans

from src.retrieve.retriever import SearchResult
from src.config import get_city_config, get_spatial_config
from src.logger import get_logger

logger = get_logger("spatial")


# ============================================================
# 常量
# ============================================================
# 地球半径（km），用于 Haversine 距离
EARTH_RADIUS_KM = 6371.0

# 纠错循环的最大轮数（防止不收敛时死循环）
MAX_CORRECTION_ROUNDS = 10

# 补簇循环的最大轮数（同上）
MAX_FILL_ROUNDS = 30

# KMeans 随机种子，保证结果可复现
DEFAULT_SEED = 42


# ============================================================
# 距离计算
# ============================================================
def haversine(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """
    Haversine 距离，单位 km。

    比欧氏距离准，尤其在中高纬度。经纬度差 1 度约 111km，
    但经度方向要乘以 cos(纬度)，Haversine 内部处理了这个。
    """
    lat1_r, lat2_r = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)

    a = (math.sin(dlat / 2) ** 2
         + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlng / 2) ** 2)
    c = 2 * math.asin(math.sqrt(a))

    return EARTH_RADIUS_KM * c


# ============================================================
# SearchResult ↔ dict
# ============================================================
def _poi_to_dict(r: SearchResult, cluster_id: int) -> dict:
    """
    把 SearchResult 转成 plan 里用的 dict。

    cluster_id 是小簇编号，供片内选点参考
    （比如「悠闲」方案可能每簇只选 1 个）。

    带上 tags / behaviors，供 poi_picker 打 behavior 加成。
    """
    return {
        "poi_id": r.id,
        "name": r.name,
        "tags": r.tags,
        "behaviors": r.behaviors,
        "lat": r.lat,
        "lng": r.lng,
        "rating": r.rating,
        "district": r.district,
        "_cluster_id": cluster_id,
    }


# ============================================================
# 1. 严格去重
# ============================================================
def strict_dedupe(results: list[SearchResult]) -> list[SearchResult]:
    """
    包含关系去重：子串关系合并，只留最短（最准）的那条。

    规则：
        - 先按名字长度升序排（短的在前）
        - 完全相同 → 留 rating 高的
        - 新的是已保留的子串（新的更短）→ 用新的替换
        - 已保留的是新的子串（已保留的更短）→ 丢弃新的

    例子：
        "黄鹤楼" / "黄鹤楼公园"     → 留 "黄鹤楼"
        "晴川阁" / "晴川阁-禹稷行宫" → 留 "晴川阁"
        "江汉路步行街" / "江汉路步行街" → 留 rating 高的

    注意：
        检索层 retriever.py 的 dedupe_by_containment 用前 3 字，
        那是故意的——检索结果允许同一主景点留 2 条，
        方便展示。但空间排序是最终行程，不该有重复感。
    """
    sorted_results = sorted(results, key=lambda r: len(r.name))
    kept: list[SearchResult] = []

    for r in sorted_results:
        name = r.name
        if not name:
            kept.append(r)
            continue

        conflict = False
        i = 0
        while i < len(kept):
            k = kept[i]
            k_name = k.name

            if not k_name:
                i += 1
                continue

            if name == k_name:
                # 完全相同，留 rating 高的
                if r.rating > k.rating:
                    kept.pop(i)
                    kept.append(r)
                conflict = True
                break

            if name in k_name:
                # r 是 k 的子串（r 更短、更准）→ 替换 k
                kept.pop(i)
                kept.append(r)
                conflict = True
                break

            if k_name in name:
                # k 是 r 的子串（k 更短、更准）→ 丢弃 r
                conflict = True
                break

            i += 1

        if not conflict:
            kept.append(r)

    deduped = kept

    if len(deduped) < len(results):
        logger.info(f"包含关系去重：{len(results)} → {len(deduped)}")

    return deduped


# ============================================================
# 2. 从住宿区读坐标
# ============================================================
def get_stay_coord(stay_district: str, city_key: str) -> tuple[float, float]:
    """
    从 {city}_anchors_clean.json 里读住宿区坐标。

    注意：住宿区名字在 cities/{city}.yaml 的 stay_districts 里，
    坐标在 data/processed/{city}_anchors_clean.json 里。

    找不到时返回 (0.0, 0.0)。
    """
    from src.retrieve.anchors import load_anchors

    anchors = load_anchors(city_key)
    for d in anchors.get("stay_districts", []):
        if d.get("name") == stay_district:
            lat = d.get("lat", 0.0)
            lng = d.get("lng", 0.0)
            if lat == 0.0 and lng == 0.0:
                logger.warning(
                    f"住宿区 [{stay_district}] 坐标还是 0.0，"
                    f"检查 anchors_clean.json 是否回填了真实坐标"
                )
            return lat, lng

    logger.warning(f"住宿区 [{stay_district}] 没找到，用 (0.0, 0.0) 兜底")
    return 0.0, 0.0


# ============================================================
# 3. 切小簇
# ============================================================
def _kmeans_labels(points: list[dict], k: int, seed: int) -> list[int]:
    """对 points 的 (_lat, _lng) 做 KMeans，返回标签列表。"""
    coords = np.array([[p["_lat"], p["_lng"]] for p in points])
    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    return km.fit_predict(coords).tolist()


def _cap_cluster_size(points: list[dict], max_size: int, seed: int) -> list[list[dict]]:
    """
    递归拆：一簇超过 max_size 就 KMeans 拆成两簇，直到每簇 ≤ max_size。
    """
    if len(points) <= max_size:
        return [points]

    k = max(2, math.ceil(len(points) / max_size))
    labels = _kmeans_labels(points, k, seed)

    groups: dict[int, list[dict]] = {}
    for p, lb in zip(points, labels):
        groups.setdefault(lb, []).append(p)

    result = []
    for g in groups.values():
        if len(g) > max_size:
            result.extend(_cap_cluster_size(g, max_size, seed))
        else:
            result.append(g)
    return result


def _split_by_distance(cluster: list[dict], max_dist_km: float) -> list[list[dict]]:
    """
    递归拆：一簇内两两最大距离超过 max_dist_km，
    就把最远的那个点拆出去，剩余部分继续判。
    """
    if len(cluster) <= 1:
        return [cluster]

    max_d = 0.0
    far_i = 0
    for i in range(len(cluster)):
        for j in range(i + 1, len(cluster)):
            d = haversine(
                cluster[i]["_lat"], cluster[i]["_lng"],
                cluster[j]["_lat"], cluster[j]["_lng"],
            )
            if d > max_d:
                max_d = d
                far_i = i

    if max_d <= max_dist_km:
        return [cluster]

    outlier = cluster[far_i]
    rest = [c for idx, c in enumerate(cluster) if idx != far_i]
    return [[outlier]] + _split_by_distance(rest, max_dist_km)


def cluster_pois(pois: list[dict], cfg: dict, seed: int = DEFAULT_SEED) -> list[list[dict]]:
    """
    切小簇：KMeans 初分 + 超限递归拆。

    Args:
        pois: POI dict 列表（每个含 _lat / _lng）
        cfg: spatial 配置
        seed: KMeans 随机种子

    Returns:
        小簇列表，每个簇是 POI dict 列表
    """
    n = len(pois)
    cluster_k = cfg["cluster_k"]
    max_size = cfg["cluster_max_size"]
    max_dist = cfg["cluster_max_dist"]

    # k 动态：候选少时降 k，避免碎成单点
    k = min(cluster_k, max(2, math.ceil(n / max_size)))
    k = min(k, n)

    coords = np.array([[p["_lat"], p["_lng"]] for p in pois])
    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    labels = km.fit_predict(coords).tolist()

    groups: dict[int, list[dict]] = {}
    for p, lb in zip(pois, labels):
        groups.setdefault(lb, []).append(p)

    clusters = []
    for g in groups.values():
        if len(g) > max_size:
            clusters.extend(_cap_cluster_size(g, max_size, seed))
        else:
            clusters.append(g)

    final = []
    for c in clusters:
        final.extend(_split_by_distance(c, max_dist))

    logger.info(f"切小簇：{n} 点 → {len(final)} 簇")
    return final


# ============================================================
# 4. 簇的工具
# ============================================================
def cluster_center(cluster: list[dict]) -> tuple[float, float]:
    """簇重心（经纬度均值）"""
    lat = sum(p["_lat"] for p in cluster) / len(cluster)
    lng = sum(p["_lng"] for p in cluster) / len(cluster)
    return lat, lng


def cluster_size(cluster: list[dict]) -> int:
    """簇里几个点"""
    return len(cluster)


def cluster_names(cluster: list[dict]) -> list[str]:
    """簇里所有点的名字"""
    return [p.get("name", "") for p in cluster]


def seg_center(segment: list[list[dict]]) -> tuple[float, float]:
    """片（= 几个簇）的重心"""
    all_pts = [p for c in segment for p in c]
    if not all_pts:
        return (0.0, 0.0)
    lat = sum(p["_lat"] for p in all_pts) / len(all_pts)
    lng = sum(p["_lng"] for p in all_pts) / len(all_pts)
    return lat, lng


def seg_size(segment: list[list[dict]]) -> int:
    """片里几个点（所有簇的点数之和）"""
    return sum(cluster_size(c) for c in segment)


def cluster_is_urban(cluster: list[dict], urban_districts: set) -> bool:
    """
    判断一个簇是不是「主城区簇」。

    规则：簇里超过一半的点落在主城区，就算主城区簇。
    """
    urban_cnt = sum(1 for p in cluster if p.get("district", "") in urban_districts)
    return urban_cnt > len(cluster) / 2


# ============================================================
# 5. 单点吸收
# ============================================================
def absorb_singletons(
    clusters: list[list[dict]],
    urban_districts: set,
) -> list[list[dict]]:
    """
    单点簇吸进离它最近的「城区多簇」。

    规则：
        - 只处理主城区的簇
        - 主城区的单点簇 → 吸进离它最近的城区多簇（点数 ≥ 2）
        - 郊区簇原样保留，不参与吸收
        - 没有城区多簇可吸时，原样返回，不做吸收
    """
    urban = [c for c in clusters if cluster_is_urban(c, urban_districts)]
    rural = [c for c in clusters if not cluster_is_urban(c, urban_districts)]

    multi = [c for c in urban if len(c) > 1]
    single = [c for c in urban if len(c) == 1]

    if not multi:
        logger.info("单点吸收：无城区多簇可吸，跳过")
        return clusters

    for sc in single:
        sc_lat, sc_lng = cluster_center(sc)
        nearest = min(
            multi,
            key=lambda mc: haversine(sc_lat, sc_lng, *cluster_center(mc)),
        )
        nearest.extend(sc)

    logger.info(f"单点吸收：{len(single)} 个城区单点簇 → 吸进 {len(multi)} 个城区多簇")
    return multi + rural


# ============================================================
# 6. 聚片 + 纠错 + 补簇
# ============================================================
def kmeans_into_days(
    clusters: list[list[dict]],
    days: int,
    cfg: dict,
    seed: int = DEFAULT_SEED,
) -> list[list[list[dict]]]:
    """
    把城区簇聚成 days 片。

    步骤：
        1. 对簇心 KMeans 聚 days 片
        2. 纠错：每簇归到最近片，先算归属再统一更新，循环到稳定
        3. 补簇：片点数 < min_pts_per_day 的，从就近且搬得起的片挪簇
    """
    min_pts = cfg["min_pts_per_day"]

    if len(clusters) <= days:
        segments = [[c] for c in clusters]
        while len(segments) < days:
            segments.append([])
        return segments[:days]

    # ---- 1. KMeans 聚 days 片 ----
    centers = np.array([cluster_center(c) for c in clusters])
    km = KMeans(n_clusters=days, random_state=seed, n_init=10)
    labels = km.fit_predict(centers).tolist()

    groups: dict[int, list[list[dict]]] = {}
    for c, lb in zip(clusters, labels):
        groups.setdefault(lb, []).append(c)

    segments = [groups[lb] for lb in sorted(groups.keys())]
    while len(segments) < days:
        segments.append([])

    # ---- 2. 纠错：每簇归到最近片，先算归属再统一更新 ----
    for _ in range(MAX_CORRECTION_ROUNDS):
        seg_centers = [
            seg_center(seg) if seg else None
            for seg in segments
        ]

        moves = []
        for seg_i, seg in enumerate(segments):
            for c in seg:
                c_lat, c_lng = cluster_center(c)
                valid = [i for i, sc in enumerate(seg_centers) if sc is not None]
                if not valid:
                    continue
                nearest_i = min(
                    valid,
                    key=lambda i: haversine(c_lat, c_lng, *seg_centers[i]),
                )
                if nearest_i != seg_i:
                    moves.append((c, nearest_i))

        if not moves:
            break

        for c, target in moves:
            for seg in segments:
                if c in seg:
                    seg.remove(c)
                    break
            segments[target].append(c)

    # ---- 3. 补簇：片点数 < min_pts 的，从就近且搬得起的片挪 ----
    for _ in range(MAX_FILL_ROUNDS):
        sizes = [seg_size(seg) for seg in segments]
        if not sizes:
            break
        min_i = sizes.index(min(sizes))

        if sizes[min_i] >= min_pts:
            break

        min_center = seg_center(segments[min_i])

        candidates = []
        for i in range(len(segments)):
            if i == min_i or not segments[i]:
                continue
            smallest_cluster_pts = min(cluster_size(c) for c in segments[i])
            if sizes[i] - smallest_cluster_pts >= min_pts:
                candidates.append(i)

        if not candidates:
            break

        donor_i = min(
            candidates,
            key=lambda i: haversine(min_center[0], min_center[1],
                                    *seg_center(segments[i])),
        )

        donor_seg = segments[donor_i]
        dists = [
            (idx, haversine(*cluster_center(c), *min_center))
            for idx, c in enumerate(donor_seg)
        ]
        dists.sort(key=lambda x: x[1])
        pick_idx = dists[0][0]

        moved = donor_seg.pop(pick_idx)
        segments[min_i].append(moved)

    return segments


# ============================================================
# 7. 片排序
# ============================================================
def order_segments(
    segments: list[list[list[dict]]],
    stay_lat: float,
    stay_lng: float,
) -> list[list[list[dict]]]:
    """
    片排序：Day1 = 离住宿区最近的片，之后贪心找离上一片最近的片。
    空片排在最后。
    """
    if not segments:
        return segments

    valid = [seg for seg in segments if seg]
    empty = [seg for seg in segments if not seg]

    if not valid:
        return segments

    seg_centers = [seg_center(seg) for seg in valid]

    remaining = list(range(len(valid)))
    ordered = []

    first = min(
        remaining,
        key=lambda i: haversine(stay_lat, stay_lng, *seg_centers[i]),
    )
    ordered.append(valid[first])
    remaining.remove(first)

    cur_center = seg_centers[first]
    while remaining:
        nearest = min(
            remaining,
            key=lambda i: haversine(cur_center[0], cur_center[1], *seg_centers[i]),
        )
        ordered.append(valid[nearest])
        cur_center = seg_centers[nearest]
        remaining.remove(nearest)

    return ordered + empty


# ============================================================
# 主流程
# ============================================================
def spatial_sort(
    results: list[SearchResult],
    days: int,
    stay_district: str,
    city_key: str,
    seed: int = DEFAULT_SEED,
) -> tuple[dict, list]:
    """
    空间排序主流程。

    Args:
        results: RAG 检索返回的候选（SearchResult 列表）
        days: 天数
        stay_district: 住宿区名，如"武昌区"
        city_key: 城市 key，如"wuhan"
        seed: 随机种子

    Returns:
        (plan, dropped)
        plan: {"Day1": [poi_dict, ...], "Day2": [...], ...}
        dropped: 没被选入任何一天的 poi_dict 列表
    """
    logger.info("=" * 60)
    logger.info(f"空间排序：{len(results)} 个候选，{days} 天，住宿={stay_district}")
    logger.info("=" * 60)

    cfg = get_spatial_config(city_key)
    urban_districts = set(cfg["urban_districts"])

    # ---- 1. 严格去重 + 过滤无效坐标 ----
    deduped = strict_dedupe(results)
    valid = [r for r in deduped if r.has_valid_coord]
    if len(valid) < len(deduped):
        logger.info(f"过滤无效坐标：{len(deduped)} → {len(valid)}")

    if not valid:
        logger.error("没有有效坐标的候选，无法空间排序")
        return {}, []

    # ---- 2. 转 dict，切小簇 ----
    pois = [
        {
            "id": r.id,
            "name": r.name,
            "tags": r.tags,
            "behaviors": r.behaviors,
            "rating": r.rating,
            "district": r.district,
            "_lat": r.lat,
            "_lng": r.lng,
            "_result": r,
        }
        for r in valid
    ]

    clusters = cluster_pois(pois, cfg, seed)

    # ---- 3. 读住宿区坐标 ----
    stay_lat, stay_lng = get_stay_coord(stay_district, city_key)
    logger.info(f"住宿区 [{stay_district}] 坐标：({stay_lat:.4f}, {stay_lng:.4f})")

    # ---- 4. 单点吸收 ----
    clusters = absorb_singletons(clusters, urban_districts)

    # ---- 5. 城郊分离 ----
    urban_clusters = [c for c in clusters if cluster_is_urban(c, urban_districts)]
    rural_clusters = [c for c in clusters if not cluster_is_urban(c, urban_districts)]

    if rural_clusters:
        logger.info(
            f"城郊分离：城区 {len(urban_clusters)} 簇，郊区 {len(rural_clusters)} 簇搁置"
        )

    # ---- 6. 聚片（城区簇）----
    segments = kmeans_into_days(urban_clusters, days, cfg, seed)

    # ---- 7. 郊区成片 ----
    empty_slots = sum(1 for seg in segments if not seg)
    if rural_clusters and empty_slots > 0:
        for i, seg in enumerate(segments):
            if not seg:
                segments[i] = rural_clusters
                logger.info(f"郊区成片：{len(rural_clusters)} 个郊区簇塞进 Day{i+1}")
                break

    # ---- 8. 片排序 ----
    segments = order_segments(segments, stay_lat, stay_lng)

    # ---- 9. 转成输出格式 ----
    plan: dict[str, list] = {}
    used_poi_ids = set()

    for day_idx, seg in enumerate(segments, start=1):
        if not seg:
            continue
        day_pois = []
        for c_idx, cluster in enumerate(seg):
            for p in cluster:
                r: SearchResult = p["_result"]
                day_pois.append(_poi_to_dict(r, c_idx))
                used_poi_ids.add(p["id"])
        plan[f"Day{day_idx}"] = day_pois

    dropped = []
    for p in pois:
        if p["id"] not in used_poi_ids:
            dropped.append(_poi_to_dict(p["_result"], -1))

    # ---- 10. 日志 ----
    logger.info("")
    logger.info("分天结果：")
    for day, pois_in_day in plan.items():
        names = [p["name"] for p in pois_in_day]
        logger.info(f"  {day}（{len(pois_in_day)} 个）：{names}")

    if dropped:
        logger.info(f"\n未选中的点（{len(dropped)} 个）")

    logger.info("=" * 60)

    return plan, dropped


# ============================================================
# 命令行测试入口
# ============================================================
def main():
    import argparse

    parser = argparse.ArgumentParser(description="空间排序测试")
    parser.add_argument("--city", default="wuhan", help="城市 key")
    parser.add_argument("--city_name", default="武汉", help="城市名")
    parser.add_argument("--query", default="历史人文 自然风光 旅游 景点")
    parser.add_argument("--top_k", type=int, default=60)
    parser.add_argument("--days", type=int, default=3)
    parser.add_argument("--stay_district", default="武昌区")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    from src.retrieve.retriever import Retriever
    retriever = Retriever(f"{args.city}_v1")
    results = retriever.search(
        query=args.query,
        city=args.city_name,
        top_k=args.top_k,
        mode="hybrid",
    )
    print(f"检索召回：{len(results)} 条")

    plan, dropped = spatial_sort(
        results=results,
        days=args.days,
        stay_district=args.stay_district,
        city_key=args.city,
        seed=args.seed,
    )

    print()
    for day, pois in plan.items():
        print(f"{day}（{len(pois)} 个）：")
        for p in pois:
            print(f"  [{p['_cluster_id']}] {p['name']} ({p['district']}) "
                  f"({p['lat']:.4f},{p['lng']:.4f})")
        print()

    if dropped:
        print(f"未选中（{len(dropped)} 个）：")
        for p in dropped:
            print(f"  {p['name']}")
        print()


if __name__ == "__main__":
    main()