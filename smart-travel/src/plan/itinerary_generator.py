"""
行程生成模块（风格前置 + 选点定内容 + LLM 只写文案）。

职责：
1. 接收结构化输入 + 行程风格（综合 / 口碑 / 悠闲）
2. 召回 → spatial_sort 分天 → 每天 poi_picker 选点（点定死）
3. 每天一次独立 LLM 调用，只写文案（不得增删景点，可调顺序/时间）
4. 返回结构化结果，供前端渲染地图和每天卡片

返回结构：
    {
      "style": "综合",
      "seed": 123,
      "days": [
        {"day": "Day1", "text": "……",
         "pois": [{name, aliases, level, photos, lat, lng, district, ...}, ...],
         "warnings": [...]},
        ...
      ]
    }

设计要点：
- 选点定内容：每天去哪些点由 poi_picker 决定，LLM 不参与选点
- 每天一次调用（并行），每天的点很少，prompt 小、更稳
- 越界/漏提只记日志，不重试、不弹提示（LLM 顺带提邻近景点是加分项）
"""

import argparse
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from openai import OpenAI

from src.config import PROJECT_ROOT, kb_path
from src.logger import get_logger
from src.retrieve.retriever import Retriever
from src.plan.spatial_sorter import spatial_sort, get_stay_coord
from src.plan.poi_picker import pick, PLAN_CONFIG

# 加载 .env（确保 DASHSCOPE_API_KEY 可用）
load_dotenv(PROJECT_ROOT / ".env")

logger = get_logger("plan")


# ============================================================
# 配置
# ============================================================
LLM_MODEL = "qwen3.8-max"    # 生成行程（走 OpenAI 兼容接口）
LLM_TEMPERATURE = 0.7        # 平衡稳定和多样
ENABLE_THINKING = False      # 关掉深度思考（写文案不需要，且更慢）
OPENAI_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

MIN_RECALL = 40              # 召回下限
RECALL_PER_DAY = 20          # 每天对应多少条
MAX_RECALL = 120             # 召回上限

STYLES = ["综合", "口碑", "悠闲"]
DEFAULT_STYLE = "综合"
PLAN_NAMES = STYLES          # 兼容旧名字

MAX_WORKERS = 4              # 每天一次调用的并行度


# ============================================================
# 召回数量
# ============================================================
def calc_recall_count(days: int) -> int:
    """
    召回数量：max(40, days * 20)，封顶 120。

    为什么给这么宽：每天候选池必须明显大于 target（5）。
    否则 pick() 会「池 ≤ 目标 → 全选」，三套风格就没区别了，
    而且池太小会把中心地标（如黄鹤楼）漏在召回之外。
    郊区会被丢掉，所以宁可多召回。

    例：
        1 天 → 40
        2 天 → 40
        3 天 → 60
        4 天 → 80
        5 天 → 100
        6 天及以上 → 120（封顶）
    """
    return min(max(MIN_RECALL, days * RECALL_PER_DAY), MAX_RECALL)


# ============================================================
# prompt
# ============================================================
_COMMON_RULES = """你是一个旅行规划师，帮用户写「其中一天」的行程文案。
风格：简洁、实用、口语化，像朋友给你列路线。

【硬约束】
- 只能用下面【本日景点】里给出的地方，不得增加、不得删除、不得替换。
- 只能调整它们的叙述顺序和时间安排。
- 不要编造不存在的景点。
- 以住宿地为中心安排每天的路线；不要写「从XX站出发」这类到站信息。

【输出要求】
- 只写这一天的一段话，不要写"Day1"这种前缀。
- 不要 Markdown 标题（#）、不要列表符号（-、*）、不要 JSON。
- 说清楚：先去哪、再去哪、大概什么时候、为什么这么排。
- 本日景点要全部出现。"""

_STYLE_RULES = {
    "综合": "【方案定位】综合：平衡兴趣、交通和时间，节奏适中，不偏科。",
    "口碑": "【方案定位】口碑：优选评分高、等级高的地方，整体质量优先；但要读起来像推荐路线，不要念评分数字。",
    "悠闲": "【方案定位】悠闲：节奏慢，留白多，安排得宽松些，适合慢慢逛，别塞太满。",
}


def build_system_prompt(style: str) -> str:
    return _COMMON_RULES + "\n\n" + _STYLE_RULES.get(style, _STYLE_RULES[DEFAULT_STYLE])


def _display_name(p: dict) -> str:
    """展示名：优先 display_name（去括号后缀），回退 name。"""
    return p.get("display_name") or p.get("name", "")


def build_day_poi_lines(pois: list) -> str:
    """当天给定景点的清单（给 LLM 看）。"""
    lines = []
    for i, p in enumerate(pois, start=1):
        aliases = p.get("aliases") or []
        alias_str = f"（含 {'/'.join(aliases)}）" if aliases else ""
        lines.append(
            f"{i}. {_display_name(p)}{alias_str} | {p.get('tags', '')} | {p.get('description', '')}"
        )
    return "\n".join(lines)


def build_user_prompt(
    city: str, interests: list, companion: str, days: int,
    stay_district: str, day_name: str, pois: list,
) -> str:
    """
    拼一天的 user prompt。

    只给住宿地，不给车站：用户通常先回酒店放行李再出门，
    「从武汉站出发」对行程没意义。
    """
    interests_str = "、".join(interests) if interests else "不限"

    return f"""【用户需求】
城市：{city}
兴趣偏好：{interests_str}
同行人：{companion}
行程共 {days} 天，这是其中：{day_name}
住宿地：{stay_district}

【本日景点】（共 {len(pois)} 个，必须全部出现，不得增删）
{build_day_poi_lines(pois)}

请写 {day_name} 这一天的行程文案。"""


# ============================================================
# 越界校验
# ============================================================
def _load_all_names(city_key: str) -> set:
    """全库景点名，用于检测 LLM 是否用了清单外的景点。"""
    p = kb_path(city_key)
    names = set()
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    o = json.loads(line)
                    names.add(o.get("display_name") or o.get("name", ""))
                except json.JSONDecodeError:
                    continue
    names.discard("")
    return names


def _out_of_scope(text: str, all_names: set, allowed: set) -> list:
    """文本里出现、但不属于当天清单的已知景点。"""
    return [n for n in all_names if n in text and n not in allowed]


def _mentioned(name: str, text: str) -> bool:
    """
    名字是否在文本里被提到（宽松匹配）。

    LLM 常把全名简写：如把「凌波门东湖观景点」写成「凌波门观景点」，
    所以除了全名，再退化到前 3 字判断。
    """
    if not name:
        return True
    if name in text:
        return True
    return len(name) >= 3 and name[:3] in text


# ============================================================
# LLM 调用
# ============================================================
def _call_llm(system_prompt: str, user_prompt: str) -> str:
    """单次对话调用（走 OpenAI 兼容接口，和打标模块保持一致）。"""
    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url=OPENAI_BASE_URL,
    )
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=LLM_TEMPERATURE,
        extra_body={"enable_thinking": ENABLE_THINKING},
    )
    return (resp.choices[0].message.content or "").strip()


def _write_one_day(
    style, city, interests, companion, days, stay_district,
    day_name, pois, all_names,
) -> tuple[str, list]:
    """
    写一天文案。

    越界/漏提只记日志，不重试、不弹提示：
    LLM 顺带提一句邻近景点（如"登晴川阁望长江、看龟山与黄鹤楼"）是加分项，不算违规。
    """
    allowed = {_display_name(p) for p in pois}
    system_prompt = build_system_prompt(style)
    user_prompt = build_user_prompt(
        city, interests, companion, days, stay_district, day_name, pois
    )

    try:
        text = _call_llm(system_prompt, user_prompt)
    except Exception as e:
        logger.error(f"[{day_name}] LLM 调用失败：{e}")
        return f"（这一天生成失败：{e}）", [f"LLM 调用失败：{e}"]

    bad = _out_of_scope(text, all_names, allowed)
    if bad:
        logger.info(f"[{day_name}] 顺带提到清单外景点（仅记录）：{bad}")

    missing = [n for n in allowed if not _mentioned(n, text)]
    if missing:
        logger.info(f"[{day_name}] 未明显提到（仅记录）：{missing}")

    return text, []


# ============================================================
# 主流程
# ============================================================
def generate_plan(
    city: str,
    interests: list,
    companion: str,
    days: int,
    depart_from: str,
    stay_district: str,
    style: str = DEFAULT_STYLE,
    city_key: str = "wuhan",
    seed: int = None,
) -> dict:
    """
    生成一套行程（按指定风格）。

    Args:
        style: 综合 / 口碑 / 悠闲
        seed: 随机种子。None 则每次随机（前端「换一批」靠它）
        depart_from: 已不参与生成（前端仍传，保留兼容）；行程以住宿地为中心

    Returns:
        {"style": ..., "seed": ..., "days": [{"day","text","pois","warnings"}, ...]}
    """
    if style not in PLAN_CONFIG:
        logger.warning(f"未知风格 {style}，退回 {DEFAULT_STYLE}")
        style = DEFAULT_STYLE

    if seed is None:
        seed = random.randint(1, 10 ** 9)

    logger.info("=" * 60)
    logger.info(f"生成行程：{city}，{days} 天，风格={style}，兴趣={interests}，seed={seed}")
    logger.info("=" * 60)

    # ---- 1. 召回 ----
    recall_n = calc_recall_count(days)
    retriever = Retriever(f"{city_key}_v1")
    # 只用「兴趣词」做 query：
    # 1) 城市已经是硬过滤（Chroma filter），写进 query 也会被剥掉，多余；
    # 2) 追加「旅游 景点」这类泛词会把语义重心带偏
    #    （实测 自然风光 0.80→0.60、夜生活 1.00→0.50）。
    query = " ".join(interests) if interests else f"{city} 景点"
    results = retriever.search(
        query=query,
        city=city,
        top_k=recall_n,
    )
    logger.info(f"召回：{len(results)} 条")
    if not results:
        return {"style": style, "seed": seed, "days": []}

    # ---- 2. 分天 ----
    plan, _ = spatial_sort(results, days, stay_district, city_key, seed=seed)
    stay_lat, stay_lng = get_stay_coord(stay_district, city_key)

    # ---- 3. 每天选点（点定死）----
    day_jobs = []
    for di, (day_name, day_pois) in enumerate(plan.items()):
        picked = pick(
            day_pois=day_pois,
            plan_type=style,
            interests=interests,
            companion=companion,
            city_key=city_key,
            start_lat=stay_lat,
            start_lng=stay_lng,
            seed=seed,
            day_index=di,
        )
        day_jobs.append((di, day_name, picked))
        logger.info(f"  {day_name} 选点：{[p['name'] for p in picked]}")

    if not day_jobs:
        return {"style": style, "seed": seed, "days": []}

    # ---- 4. 每天一次 LLM（并行）----
    all_names = _load_all_names(city_key)

    def _run(job):
        di, day_name, picked = job
        text, warnings = _write_one_day(
            style, city, interests, companion, days, stay_district,
            day_name, picked, all_names,
        )
        return di, day_name, picked, text, warnings

    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(day_jobs))) as ex:
        written = list(ex.map(_run, day_jobs))

    # ---- 5. 组装 ----
    out_days = []
    for di, day_name, picked, text, warnings in sorted(written, key=lambda x: x[0]):
        out_days.append({
            "day": day_name,
            "text": text,
            "pois": picked,
            "warnings": warnings,
        })

    logger.info(f"生成完成：{len(out_days)} 天")
    return {"style": style, "seed": seed, "days": out_days}


# ============================================================
# 命令行测试入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="行程生成测试")
    parser.add_argument("--city_key", default="wuhan", help="城市 key")
    parser.add_argument("--city", default="武汉", help="城市名")
    parser.add_argument("--style", default=DEFAULT_STYLE, choices=STYLES)
    parser.add_argument("--interests", nargs="*", default=["历史人文", "自然风光"])
    parser.add_argument("--companion", default="情侣")
    parser.add_argument("--days", type=int, default=3)
    parser.add_argument("--depart_from", default="武汉站")
    parser.add_argument("--stay_district", default="武昌区")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    res = generate_plan(
        city=args.city, interests=args.interests, companion=args.companion,
        days=args.days, depart_from=args.depart_from, stay_district=args.stay_district,
        style=args.style, city_key=args.city_key, seed=args.seed,
    )

    print(f"\n风格：{res['style']}   seed={res.get('seed')}   天数：{len(res['days'])}")
    for d in res["days"]:
        print("\n" + "=" * 60)
        print(f"{d['day']}   景点：{[p['name'] for p in d['pois']]}")
        if d.get("warnings"):
            print(f"  警告：{d['warnings']}")
        print("-" * 60)
        print(d["text"])


if __name__ == "__main__":
    main()
