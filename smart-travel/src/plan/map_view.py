"""
行程地图可视化（folium + 高德瓦片）。

职责：
把「分天的已选景点」画成一张 folium 地图。
- 底图用**高德瓦片**（中文标注，国内可加载；OSM 在国内常被墙，导致底图空白）
- 每天一个色块：凸包 + Chaikin 磨圆 + 膨胀，保证完全包住当天所有点（有包裹感）
- 色块上标注 Day1 / Day2
- 景点按序号标数字
- **不画连线**：LLM 会按「早晚/顺路」重排叙述顺序，连线反而误导

几何计算纯 Python，不引新依赖。
"""

import math

import folium
from folium import DivIcon


# 每天一色（十六进制，Polygon 和 HTML 标记都能直接用）
DAY_COLORS = [
    "#e6194b", "#4363d8", "#3cb44b", "#911eb4", "#f58231",
    "#42d4f4", "#9a6324", "#800000", "#808000", "#000075",
    "#f032e6", "#a9a9a9", "#000000", "#fabed4", "#469990",
]

# 高德路网瓦片（中文标注，无需 key）
AMAP_TILES = (
    "https://webrd0{s}.is.autonavi.com/appmaptile"
    "?lang=zh_cn&size=1&scale=1&style=8&x={x}&y={y}&z={z}"
)
AMAP_ATTR = "&copy; 高德地图"

EARTH_RADIUS_KM = 6371.0
LAT_KM = 111.32


# ============================================================
# 几何工具
# ============================================================
def haversine(lat1, lng1, lat2, lng2) -> float:
    lat1_r, lat2_r = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlng / 2) ** 2)
    return EARTH_RADIUS_KM * 2 * math.asin(math.sqrt(a))


def _centroid(points):
    n = len(points)
    return (sum(p[0] for p in points) / n, sum(p[1] for p in points) / n)


def _convex_hull(points):
    """Andrew monotone chain。点数 ≤2 时原样返回。"""
    pts = sorted(set(points))
    if len(pts) <= 2:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def _chaikin(poly, iterations=3):
    """Chaikin 切角，把多边形磨圆。"""
    pts = list(poly)
    for _ in range(iterations):
        new = []
        n = len(pts)
        for i in range(n):
            p, q = pts[i], pts[(i + 1) % n]
            new.append((0.75 * p[0] + 0.25 * q[0], 0.75 * p[1] + 0.25 * q[1]))
            new.append((0.25 * p[0] + 0.75 * q[0], 0.25 * p[1] + 0.75 * q[1]))
        pts = new
    return pts


def _scale_about(points, center, factor):
    return [
        (center[0] + (p[0] - center[0]) * factor,
         center[1] + (p[1] - center[1]) * factor)
        for p in points
    ]


def _circle(points, n=48):
    clat, clng = _centroid(points)
    r = max(haversine(clat, clng, p[0], p[1]) for p in points)
    r = max(r, 0.4)
    out = []
    for k in range(n):
        ang = 2 * math.pi * k / n
        dlat = (r * math.sin(ang)) / LAT_KM
        dlng = (r * math.cos(ang)) / (LAT_KM * math.cos(math.radians(clat)))
        out.append((clat + dlat, clng + dlng))
    return out


def _to_local(points, lat0):
    k = math.cos(math.radians(lat0))
    return [(p[1] * k * LAT_KM, p[0] * LAT_KM) for p in points]


def _point_in_polygon(pt, poly):
    x, y = pt
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def _dist_point_seg(p, a, b):
    px, py = p
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _min_boundary_km(local_pts, local_poly):
    m = float("inf")
    n = len(local_poly)
    for p in local_pts:
        for i in range(n):
            m = min(m, _dist_point_seg(p, local_poly[i], local_poly[(i + 1) % n]))
    return m


def _fit_area(pts, base_poly, center, margin_km=0.35):
    """把色块逐步放大，直到完全包住 pts 且点离边界 ≥ margin_km（有包裹感）。"""
    lat0 = center[0]
    local_pts = _to_local(pts, lat0)
    last = base_poly
    for factor in (1.15, 1.3, 1.45, 1.6, 1.8, 2.0, 2.3, 2.7, 3.2):
        poly = _scale_about(base_poly, center, factor)
        local_poly = _to_local(poly, lat0)
        if (all(_point_in_polygon(p, local_poly) for p in local_pts)
                and _min_boundary_km(local_pts, local_poly) >= margin_km):
            return poly
        last = poly
    return last


# ============================================================
# 构建地图
# ============================================================
def build_map(days: list, stay_lat: float = 0.0, stay_lng: float = 0.0) -> folium.Map:
    """
    把分天的已选景点画成 folium 地图（高德底图）。

    Args:
        days: [{"day": "Day1", "pois": [{"name"/"display_name","lat","lng"}, ...]}, ...]
    """
    all_latlng = [(p["lat"], p["lng"]) for d in days for p in d.get("pois", [])]

    if all_latlng:
        lat_c = sum(p[0] for p in all_latlng) / len(all_latlng)
        lng_c = sum(p[1] for p in all_latlng) / len(all_latlng)
    else:
        lat_c, lng_c = 30.55, 114.30

    m = folium.Map(location=[lat_c, lng_c], zoom_start=12, tiles=None, control_scale=True)
    folium.TileLayer(
        tiles=AMAP_TILES, attr=AMAP_ATTR,
        subdomains="1234", max_zoom=19, name="高德地图",
    ).add_to(m)

    # 住宿区
    if stay_lat or stay_lng:
        folium.Marker(
            [stay_lat, stay_lng], popup="住宿区", tooltip="住宿区",
            icon=folium.Icon(color="black", icon="home", prefix="fa"),
        ).add_to(m)

    for day_idx, d in enumerate(days):
        color = DAY_COLORS[day_idx % len(DAY_COLORS)]
        pois = d.get("pois", [])
        pts = [(p["lat"], p["lng"]) for p in pois]
        if not pts:
            continue
        center = _centroid(pts)

        # ---- 范围色块 ----
        base = _chaikin(_convex_hull(pts), 3) if len(pts) >= 3 else _circle(pts)
        area = _fit_area(pts, base, center)
        folium.Polygon(
            locations=area, color=color, weight=2, dash_array="6",
            fill=True, fill_opacity=0.12, popup=d.get("day", ""),
        ).add_to(m)

        # ---- Day 标签：放在色块西北角（左上），避免挡住中间的点 ----
        max_lat = max(p[0] for p in area)
        min_lng = min(p[1] for p in area)
        nw = min(
            area,
            key=lambda p: (max_lat - p[0]) + (p[1] - min_lng),
        )
        label_html = (
            f'<div style="font-size:17px;font-weight:700;color:{color};'
            f'text-shadow:0 0 3px #fff,0 0 3px #fff;white-space:nowrap">'
            f'{d.get("day", "")}</div>'
        )
        folium.Marker(
            location=[nw[0], nw[1]],
            icon=DivIcon(html=label_html, icon_size=(64, 24), icon_anchor=(32, 12)),
        ).add_to(m)

        # ---- 编号点 ----
        for i, (p, latlng) in enumerate(zip(pois, pts), start=1):
            html = (
                f'<div style="background:{color};color:#fff;border-radius:50%;'
                f'width:22px;height:22px;text-align:center;line-height:22px;'
                f'font-size:12px;font-weight:bold;border:2px solid #fff;'
                f'box-shadow:0 0 3px rgba(0,0,0,.5)">{i}</div>'
            )
            name = p.get("display_name") or p.get("name", "")
            folium.Marker(
                location=list(latlng),
                tooltip=f"{d.get('day', '')} #{i} {name}",
                icon=DivIcon(html=html, icon_size=(22, 22), icon_anchor=(11, 11)),
            ).add_to(m)

    if all_latlng:
        try:
            m.fit_bounds([[lat, lng] for lat, lng in all_latlng], padding=(40, 40))
        except Exception:
            pass

    return m


def map_html(m: folium.Map) -> str:
    """folium 地图的 HTML，供 Streamlit components.html 嵌入。"""
    return m._repr_html_()
