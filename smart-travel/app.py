"""
智能旅行规划 - Streamlit 前端。

职责：
1. 展示表单，收集用户输入
2. 调 itinerary_generator 生成行程
3. 让用户从三套方案里选一套，展示选中的

运行：
    streamlit run app.py
"""

import streamlit as st

from src.config import (
    get_city_config,
    get_stay_districts,
    get_transport_hubs,
)
from src.plan.itinerary_generator import (
    generate,
    calc_recall_count,
    PLAN_NAMES,
)
from src.retrieve.retriever import Retriever


# ============================================================
# 配置
# ============================================================
# 城市名 → city_key 映射
# 目前只有武汉有知识库，所以只放武汉
CITY_MAP = {
    "武汉": "wuhan",
}

# 兴趣标签（和 schema.yaml 的 interest_tags 一致）
INTEREST_OPTIONS = [
    "历史人文",
    "自然风光",
    "美食探店",
    "购物休闲",
    "夜生活",
]

# 同行人（和 schema.yaml 的 crowd_tags 部分一致）
COMPANION_OPTIONS = [
    "独行",
    "情侣",
    "家庭",
    "朋友",
    "亲子",
]


# ============================================================
# 页面配置
# ============================================================
st.set_page_config(
    page_title="智能旅行规划",
    layout="centered",
)

st.title("智能旅行规划")
st.caption("选择你的偏好，AI 为你生成三套行程方案")


# ============================================================
# 表单
# ============================================================
with st.form("plan_form"):
    st.subheader("旅行偏好")

    # 城市（目前只有武汉）
    city = st.selectbox("城市", list(CITY_MAP.keys()))

    # 拿到 city_key 和城市配置
    city_key = CITY_MAP[city]

    # 兴趣（多选）
    interests = st.multiselect(
        "兴趣偏好",
        INTEREST_OPTIONS,
        default=["自然风光", "购物休闲"],
    )

    # 同行人
    companion = st.selectbox("和谁一起", COMPANION_OPTIONS, index=0)

    # 天数
    days = st.number_input("天数", min_value=1, max_value=10, value=3, step=1)

    # 出发地（从配置读）
    hubs = get_transport_hubs(city_key)
    depart_from = st.selectbox("出发地", hubs) if hubs else ""

    # 住宿区（从配置读，用 stay_districts）
    districts = get_stay_districts(city_key)
    stay_district = st.selectbox("住宿区域", districts) if districts else ""

    # 提交按钮
    submitted = st.form_submit_button("生成行程", use_container_width=True)


# ============================================================
# 生成
# ============================================================
if submitted:
    # 校验：兴趣不能为空
    if not interests:
        st.warning("请至少选择一个兴趣偏好")
        st.stop()

    # 生成
    with st.spinner("AI 正在规划行程，请稍候（约 30 秒）..."):
        try:
            plans = generate(
                city=city,
                interests=interests,
                companion=companion,
                days=days,
                depart_from=depart_from,
                stay_district=stay_district,
                city_key=city_key,
            )
            # 存进 session_state，避免页面重跑后丢失
            st.session_state["plans"] = plans
            st.session_state["last_form"] = {
                "city": city,
                "city_key": city_key,
                "interests": interests,
                "days": days,
            }
        except Exception as e:
            st.error(f"生成失败：{e}")
            st.stop()


# ============================================================
# 展示（只要 session_state 里有结果就展示）
# ============================================================
if "plans" in st.session_state:
    plans = st.session_state["plans"]
    form_info = st.session_state.get("last_form", {})

    st.success("生成完成！请从下面三套方案中选择一套：")
    st.divider()

    # ---- 三套方案单选 ----
    selected = st.radio(
        "选择你喜欢的方案",
        PLAN_NAMES,
        horizontal=True,
        key="selected_plan",
    )

    # ---- 展示选中方案 ----
    st.subheader(f"【{selected}】")
    st.write(plans.get(selected, "（无内容）"))
    st.divider()

    # ---- 候选景点（折叠）----
    with st.expander("查看候选景点（AI 参考了哪些地方）"):
        try:
            city_key = form_info.get("city_key", "wuhan")
            city = form_info.get("city", "武汉")
            interests = form_info.get("interests", [])
            days = form_info.get("days", 3)

            retriever = Retriever(f"{city_key}_v1")
            recall_n = calc_recall_count(days)
            results = retriever.search(
                query=f"{city} {' '.join(interests)} 旅游 景点",
                city=city,
                tags=interests if interests else None,
                top_k=recall_n,
            )
            for r in results:
                meta = r.document.metadata
                st.markdown(
                    f"**{meta.get('name', '')}**  "
                    f"`{meta.get('category', '')}`  "
                    f"评分 {meta.get('rating', '')}  \n"
                    f"{meta.get('highlight', '')}"
                )
        except Exception as e:
            st.warning(f"加载候选景点失败：{e}")
