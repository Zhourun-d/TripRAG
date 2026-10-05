"""
行程生成模块。

职责（大白话）：
1. 接收结构化输入（城市、兴趣、同行人、天数、出发地、住宿区）
2. 用兴趣作为 tags 过滤，召回一批候选 POI
3. 把候选 POI 精简成给 LLM 的清单
4. 拼 prompt，调 qwen-plus，一次生成三套行程方案
5. 把结果切成三份返回

设计要点：
- 召回数量随天数变：max(20, days * 8)，封顶 60
- 给 LLM 的 POI 描述：名字 | 类别 | 评分 | 亮点（不给地址和开放时间）
- 输出自然语言，三套方案用 【综合】/【口碑】/【悠闲】 分隔
- 返回字典 {"综合": "...", "口碑": "...", "悠闲": "..."}
- 切分失败有兜底：全塞进"综合"
"""

import argparse
import re

from dotenv import load_dotenv
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.messages import SystemMessage, HumanMessage

from src.config import PROJECT_ROOT
from src.retrieve.retriever import Retriever
from src.logger import get_logger

# 加载 .env（确保 DASHSCOPE_API_KEY 可用）
load_dotenv(PROJECT_ROOT / ".env")

logger = get_logger("plan")


# ============================================================
# 配置
# ============================================================
LLM_MODEL = "qwen-plus"      # 生成行程用 plus，比 turbo 质量好
LLM_TEMPERATURE = 0.7        # 平衡稳定和多样

# 召回数量规则
MIN_RECALL = 20              # 下限
RECALL_PER_DAY = 8           # 每天对应多少条
MAX_RECALL = 60              # 上限

# 三套方案的名字
PLAN_NAMES = ["综合", "口碑", "悠闲"]


# ============================================================
# 1. 计算召回数量
# ============================================================
def calc_recall_count(days: int) -> int:
    """
    根据天数算召回数量。

    规则：max(20, days * 8)，封顶 60

    例：
        1 天 → 20
        3 天 → 24
        5 天 → 40
        7 天 → 56
        10 天 → 60（封顶）
    """
    n = max(MIN_RECALL, days * RECALL_PER_DAY)
    return min(n, MAX_RECALL)


# ============================================================
# 2. 把召回结果精简成给 LLM 的清单
# ============================================================
def build_poi_list(results: list) -> str:
    """
    把 SearchResult 列表，拼成给 LLM 看的候选景点清单。

    格式：
        1. 黄鹤楼 | 历史人文 | 评分4.8 | 武汉地标，江南三大名楼之一
        2. 户部巷 | 美食探店 | 评分4.7 | 武汉传统小吃聚集地
        ...

    只给名字、类别、评分、亮点，不给地址和开放时间（省 token，也避免干扰）。
    """
    lines = []
    for i, r in enumerate(results, start=1):
        meta = r.document.metadata
        name = meta.get("name", "")
        category = meta.get("category", "")
        rating = meta.get("rating", "")
        highlight = meta.get("highlight", "")

        lines.append(f"{i}. {name} | {category} | 评分{rating} | {highlight}")

    return "\n".join(lines)


# ============================================================
# 3. 拼 prompt
# ============================================================
def build_system_prompt() -> str:
    """
    系统 prompt：定义角色、任务、三套方案的区别、输出格式。
    """
    return """你是一个旅行规划师，帮用户安排行程。你的风格是简洁、实用、口语化，像朋友给你列了个路线清单。

【任务】
根据用户需求和候选景点，生成三套不同的行程方案。
评分仅给你参考，生成方案的时候不要提评分。例如：“武汉大学（评分5.0），黄鹤楼公园（评分4.7，登楼必选）", 不要出现这类字段！

【三套方案的区别】
1. 【综合】平衡兴趣、交通和时间，节奏适中，不偏科。
2. 【口碑】评分高、等级高的景点作为参考，不过暂无评分的也可考虑，生成的方案不提评分。
3. 【悠闲】每天景点少，留白多，节奏慢，适合慢慢逛。

【语言风格】
- 简洁自然，像朋友推荐路线一样直白
- 不要文艺抒情，不要用比喻和排比，不要过度渲染
- 不要用"轻轻收尾""按下淡出键"这类文艺表达
- 重点是信息：去哪、顺序、大致时间、为什么推荐

【关于评分】
候选清单里带评分，但评分只作参考，不是重点。
- 评分不提具体数字，只能说评分较高这些模糊概念
- 目标是读起来像推荐路线，不像念数据

【输出格式要求】
- 不要用 Markdown 标题（#）、不要用列表符号（-、*）、不要用 JSON
- 每套方案开头单独一行写方案名：【综合】、【口碑】、【悠闲】
- 如果行程是 2 天及以上，每天以"Day1：""Day2："这样的形式开头，后面接一整段内容
- 如果行程只有 1 天，不写"Day1："前缀，直接写内容
- 每天的内容写成一整段，段与段之间空一行
- 三套方案之间空两行

【重要】
- 只用候选景点里的地方，不要自己编造景点
- 每天的景点数量要合理，不要塞太满，也不要太空
- 出发地和住宿区域只作为参考，帮助安排路线顺序，不要写进"候选景点"
"""


def build_user_prompt(
    city: str,
    interests: list,
    companion: str,
    days: int,
    depart_from: str,
    stay_district: str,
    poi_list: str,
) -> str:
    """
    用户 prompt：用户需求 + 候选景点清单。
    """
    interests_str = "、".join(interests) if interests else "不限"

    return f"""【用户需求】
城市：{city}
兴趣偏好：{interests_str}
同行人：{companion}
天数：{days} 天
出发地：{depart_from}
住宿区域：{stay_district}

【候选景点】（共 {len(poi_list.splitlines())} 个）
{poi_list}

请根据以上信息，生成三套 {days} 天的行程方案。"""


# ============================================================
# 4. 切分三套方案
# ============================================================
def split_plans(raw: str) -> dict:
    """
    把 LLM 的完整输出，按【综合】/【口碑】/【悠闲】切分成三段。

    兜底：如果切不出来，全部塞进"综合"，另外两个为空。

    返回：{"综合": "...", "口碑": "...", "悠闲": "..."}
    """
    result = {name: "" for name in PLAN_NAMES}

    # 用正则找每个方案名的位置
    # 匹配形如【综合】或 [综合] 或 综合：的标记
    positions = {}
    for name in PLAN_NAMES:
        # 找 【综合】 / [综合] / 综合： 这几种写法
        pattern = rf"[【\[]\s*{name}\s*[】\]]|{name}[:：]"
        match = re.search(pattern, raw)
        if match:
            positions[name] = match.start()

    # 如果三个都没找到，兜底
    if not positions:
        logger.warning("  方案切分失败，全部塞进'综合'")
        result["综合"] = raw.strip()
        return result

    # 按位置排序，切出每段
    sorted_names = sorted(positions.keys(), key=lambda n: positions[n])
    for i, name in enumerate(sorted_names):
        start = positions[name]
        if i + 1 < len(sorted_names):
            end = positions[sorted_names[i + 1]]
        else:
            end = len(raw)

        segment = raw[start:end].strip()
        # 去掉开头的方案名标记
        segment = re.sub(rf"^[【\[]?\s*{name}\s*[】\]]?[:：]?\s*", "", segment)
        result[name] = segment

    return result


# ============================================================
# 5. 主流程：生成行程
# ============================================================
def generate(
    city: str,
    interests: list,
    companion: str,
    days: int,
    depart_from: str,
    stay_district: str,
    city_key: str = "wuhan",
) -> dict:
    """
    生成三套行程方案。

    Args:
        city: 城市名，如"武汉"
        interests: 兴趣标签列表，如 ["历史人文", "自然风光"]
        companion: 同行人，如"情侣"
        days: 天数
        depart_from: 出发地，如"武汉站"
        stay_district: 住宿区，如"武昌区"
        city_key: 城市 key，用于加载向量库，如"wuhan"

    Returns:
        {"综合": "...", "口碑": "...", "悠闲": "..."}
    """
    logger.info("=" * 60)
    logger.info(f"生成行程：{city}，{days} 天，兴趣={interests}")
    logger.info("=" * 60)

    # ---- 1. 召回 ----
    recall_n = calc_recall_count(days)
    logger.info(f"召回数量：{recall_n}")

    retriever = Retriever(f"{city_key}_v1")
    results = retriever.search(
        query=f"{city} {' '.join(interests)} 旅游 景点",
        city=city,
        tags=interests if interests else None,
        top_k=recall_n,
    )
    logger.info(f"召回结果：{len(results)} 条")

    if not results:
        logger.warning("召回为空，无法生成行程")
        return {name: "抱歉，没有找到符合条件的景点。" for name in PLAN_NAMES}

    # ---- 2. 拼清单 ----
    poi_list = build_poi_list(results)

    # ---- 3. 拼 prompt ----
    system_prompt = build_system_prompt()
    user_prompt = build_user_prompt(
        city=city,
        interests=interests,
        companion=companion,
        days=days,
        depart_from=depart_from,
        stay_district=stay_district,
        poi_list=poi_list,
    )

    # ---- 4. 调 LLM ----
    logger.info(f"调用 {LLM_MODEL}（temperature={LLM_TEMPERATURE}）...")
    llm = ChatTongyi(model=LLM_MODEL, temperature=LLM_TEMPERATURE)

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]

    try:
        response = llm.invoke(messages)
        raw = response.content.strip()
    except Exception as e:
        logger.error(f"LLM 调用失败：{e}")
        return {name: f"生成失败：{e}" for name in PLAN_NAMES}

    logger.info(f"LLM 返回 {len(raw)} 字")

    # ---- 5. 切分三套方案 ----
    plans = split_plans(raw)

    for name in PLAN_NAMES:
        logger.info(f"  【{name}】{len(plans[name])} 字")

    return plans


# ============================================================
# 命令行测试入口
# ============================================================
def main():
    """手动测试行程生成"""
    parser = argparse.ArgumentParser(description="行程生成测试")
    parser.add_argument("--city_key", default="wuhan", help="城市 key")
    parser.add_argument("--city", default="武汉", help="城市名")
    parser.add_argument("--interests", nargs="*", default=["历史人文", "自然风光"],
                        help="兴趣标签，可多个")
    parser.add_argument("--companion", default="情侣", help="同行人")
    parser.add_argument("--days", type=int, default=3, help="天数")
    parser.add_argument("--depart_from", default="武汉站", help="出发地")
    parser.add_argument("--stay_district", default="武昌区", help="住宿区")
    args = parser.parse_args()

    plans = generate(
        city=args.city,
        interests=args.interests,
        companion=args.companion,
        days=args.days,
        depart_from=args.depart_from,
        stay_district=args.stay_district,
        city_key=args.city_key,
    )

    # ---- 打印结果 ----
    for name in PLAN_NAMES:
        print("\n" + "=" * 60)
        print(f"【{name}】")
        print("=" * 60)
        print(plans[name])


if __name__ == "__main__":
    main()
