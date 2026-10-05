"""
智能旅行规划 - Streamlit 前端。

流程：
表单（城市 / 行程风格 / 兴趣 / 同行人 / 天数 / 出发地 / 住宿区）
  → generate_plan（选点定内容，LLM 只写文案）
  → pydeck 地图 + 每天一个卡片（文案 + 景点图片）

运行：
    streamlit run app.py
"""

import html

import streamlit as st
import streamlit.components.v1 as components

from src.config import (
    list_cities,
    get_city_config,
    get_stay_districts,
    get_transport_hubs,
)
from src.plan.itinerary_generator import generate_plan, STYLES, DEFAULT_STYLE
from src.plan.map_view import build_map, map_html
from src.plan.spatial_sorter import get_stay_coord


# ============================================================
# 配置
# ============================================================
CITY_MAP = {get_city_config(k).get("name", k): k for k in list_cities()}

INTEREST_OPTIONS = ["历史人文", "自然风光", "美食探店", "购物休闲", "夜生活"]
COMPANION_OPTIONS = ["独行", "情侣", "家庭", "朋友", "亲子"]

MAX_PHOTOS_PER_DAY = 6


def _pname(p: dict) -> str:
    """展示用名字：优先 display_name。"""
    return p.get("display_name") or p.get("name", "")


# ============================================================
# 图片
# ============================================================
def _render_photos(pois: list, max_n: int = MAX_PHOTOS_PER_DAY):
    """每个景点一张缩略图 + 名字链接；图挂了兜底成文字链接。"""
    cards = []
    shown = 0
    for p in pois:
        if shown >= max_n:
            break
        photos = p.get("photos") or []
        if not photos:
            continue
        url = html.escape(photos[0], quote=True)
        name = html.escape(_pname(p))
        cards.append(
            '<div style="display:inline-block;width:150px;margin:6px;text-align:center;'
            'vertical-align:top">'
            f'<a href="{url}" target="_blank">'
            f'<img src="{url}" alt="{name}" '
            'style="width:150px;height:110px;object-fit:cover;border-radius:10px;'
            'border:1px solid #ddd" '
            "onerror=\"this.style.display='none'\"></a>"
            '<div style="font-size:12px;margin-top:4px">'
            f'<a href="{url}" target="_blank">{name}</a></div></div>'
        )
        shown += 1

    if cards:
        st.markdown("".join(cards), unsafe_allow_html=True)


# ============================================================
# 页面
# ============================================================
st.set_page_config(page_title="智能旅行规划", layout="centered")
st.title("智能旅行规划")
st.caption("选好偏好和风格，AI 为你排一条顺路的行程")


# ---- 城市放在表单外：换城市时立刻联动「出发地 / 住宿区」 ----
city = st.selectbox("城市", list(CITY_MAP.keys()))
city_key = CITY_MAP[city]


# ---- 其余字段放表单里 ----
with st.form("plan_form"):
    st.subheader("旅行偏好")

    style = st.selectbox("行程风格", STYLES, index=STYLES.index(DEFAULT_STYLE))

    interests = st.multiselect(
        "兴趣偏好", INTEREST_OPTIONS, default=["自然风光", "购物休闲"]
    )
    companion = st.selectbox("和谁一起", COMPANION_OPTIONS, index=0)
    days = st.number_input("天数", min_value=1, max_value=10, value=3, step=1)

    hubs = get_transport_hubs(city_key)
    depart_from = st.selectbox("出发地", hubs) if hubs else ""

    districts = get_stay_districts(city_key)
    stay_district = st.selectbox("住宿区域", districts) if districts else ""

    submitted = st.form_submit_button("生成行程", use_container_width=True)


def _do_generate(form: dict) -> dict:
    with st.spinner("AI 正在规划行程，请稍候（约 20 秒）..."):
        return generate_plan(
            city=form["city"],
            interests=form["interests"],
            companion=form["companion"],
            days=form["days"],
            depart_from=form["depart_from"],
            stay_district=form["stay_district"],
            style=form["style"],
            city_key=form["city_key"],
        )


if submitted:
    if not interests:
        st.warning("请至少选择一个兴趣偏好")
        st.stop()

    form = {
        "city": city,
        "city_key": city_key,
        "style": style,
        "interests": interests,
        "companion": companion,
        "days": int(days),
        "depart_from": depart_from,
        "stay_district": stay_district,
    }
    try:
        st.session_state["result"] = _do_generate(form)
        st.session_state["form"] = form
    except Exception as e:
        st.error(f"生成失败：{e}")
        st.stop()


# ---- 展示 ----
result = st.session_state.get("result")
if result:
    form = st.session_state.get("form", {})
    days_data = result.get("days", [])

    if not days_data:
        st.warning("没有生成出行程，换一组条件再试试。")
    else:
        st.success(f"生成完成：{result.get('style', '')} 方案")

        # 地图（pydeck，原生组件）
        slat, slng = get_stay_coord(
            form.get("stay_district", ""), form.get("city_key", "wuhan")
        )
        components.html(map_html(build_map(days_data, slat, slng)), height=620)

        # 每天一个卡片
        for d in days_data:
            with st.container(border=True):
                st.subheader(d["day"])
                st.write(d.get("text") or "（无内容）")
                _render_photos(d.get("pois", []))

        if st.button("换一批（重新随机选点）"):
            try:
                st.session_state["result"] = _do_generate(form)
                st.rerun()
            except Exception as e:
                st.error(f"生成失败：{e}")
