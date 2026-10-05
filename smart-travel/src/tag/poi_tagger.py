"""
POI 打标签模块。

职责：
1. 读 clean 后的 POI
2. 每条调一次通义千问，返回 is_attraction / tags / behaviors / description
3. 把结果写回 POI，保存

输入：data/processed/{city}_clean.json
输出：data/processed/{city}_tagged.json

四个标签：
- is_attraction：是不是旅游目的地，布尔值
- tags：兴趣标签，多选，从 interest_tags 选 1~3 个
- behaviors：体验词，多选，从 behavior_tags 选 0~N 个
- description：一句话描述，自然语言

is_attraction 和其余字段的关系：
    is_attraction 是第一层判断——"这个地方值不值得游客去"。
    tags / behaviors / description 是第二层——"值得去的话，是什么样"。
    is_attraction=false 时，其余字段全部清空。

LLM 调用说明：
    使用 OpenAI 兼容接口调 qwen3.5-flash。
    qwen3.5-flash 是分类/抽取任务的推荐模型，不带思考链，比 3.8 系列快。
    统一用 OpenAI 兼容接口，后续换模型不用改调用方式。

设计要点：
1. 一次调用输出所有字段，不拆成多次（省成本、保证一致性）
2. 断点续跑：已处理的 id 跳过，支持中途崩溃后继续
3. 失败不中断：单条失败记日志、填空标签、继续下一条
4. 立即落盘：每条处理完就 append 到文件，防止崩溃丢数据

关于 description：
    接受一定幻觉风险，但 prompt 里约束"只写确信的，不确定的不写"。
    只描述"能做什么、有什么特点"，不写"适合谁"（人群映射由系统做）。
"""

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from src.config import (
    PROJECT_ROOT,
    clean_path,
    tagged_path,
    list_cities,
    get_interest_tags,
    get_behavior_tags,
)
from src.logger import get_logger

# 加载 .env（确保 DASHSCOPE_API_KEY 可用）
load_dotenv(PROJECT_ROOT / ".env")

logger = get_logger("tag")


# ============================================================
# LLM 配置
# ============================================================
# qwen3.5-flash：分类/抽取任务专用，不带思考链，比 3.8 快
LLM_MODEL = "qwen3.5-flash"
LLM_TEMPERATURE = 0.3

# OpenAI 兼容接口的 base_url
# 华北2（北京）地域：https://dashscope.aliyuncs.com/compatible-mode/v1
# 如果你用的是其他地域，改这里
OPENAI_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


# ============================================================
# Prompt 模板
# ============================================================
SYSTEM_PROMPT_TEMPLATE = """你是旅游数据标注专家。用户给你一个地点的基本信息，你要输出四个字段。

【输出字段】
- "is_attraction"：这个地点是不是"值得游客专程去的地方"，true / false。
  判断标准：游客来这座城市玩，会不会把它安排进行程？
    例如：
    - true：景区、公园、博物馆、寺庙、历史建筑、知名商业街、步行街、商圈、知名美食街、自然景观等
    - false：
      1. 单个店铺（卡牌店、奶茶店、服装店、便利店）、批发市场、学校、医院、写字楼、住宅小区
      2. 交通工具：观光巴士、观光游船、旅游巴士、游船、码头、观光车、索道站
      3. 景点内部的子设施：大殿、观景台、停车场、母婴室、后花园、文理学部、南馆、东馆、沙滩、音乐喷泉、雕像
      例如：武汉大学信息学部 -> false（武汉大学就是景点，但没有人会推荐去武汉大学信息学部旅游，因为这是子设施）
           武汉旅游观光巴士 -> false（交通工具）
           
  注意：is_attraction=false 时，其余三个字段全部填空字符串。


- "tags"：兴趣标签，从以下词表中选 1~3(含3)个，用英文逗号分隔：
  {tags_options}
  原则：标签描述了这个地点真实的属性，倾向于多打标签，标签越全面越好，但不要为了凑数硬加。

  一些例子参考：
    - 大学：历史人文 + 自然风光（校园本身是景观）
    - 古楼/名楼：历史人文 + 自然风光（登高望远的视野）
    - 寺庙：历史人文 + 自然风光（如果依山而建、有园林）
    - 美食街/小吃街：美食探店 + 购物休闲（吃 + 逛街）
    - 商业步行街：购物休闲 + 历史人文（如果有历史建筑）
    - 江滩/湖岸：自然风光 + 夜生活（如果晚上有灯光）
  反面例子（不要犯这类错）：
    - 纯公园（解放公园、沙湖公园）→ 只有"自然风光"，不要加"购物休闲"


- "behaviors"：体验词，从以下词表中选 2~5(含5)个，用英文逗号分隔：
  {behaviors_options}
  这些词描述"这个点能做什么、有什么氛围"。
  原则：
    - 只勾这个点真的能做的，不要为了凑数硬加
    - 可以多勾，也倾向于多选，词越全面越好，但每个都要有依据
    - 不是每个点都要勾，不确定就少勾

  一些例子参考：
    - 黄鹤楼：拍照,赏景,散步（登楼看江、拍照打卡）
    - 东湖：散步,赏景,拍照,遛娃（开阔、适合慢慢逛）
    - 户部巷：美食,热闹,逛街（小吃街，人多）
    - 湖北省博物馆：看展,研学（看文物、学历史）
    - 江汉路步行街：逛街,美食,热闹,夜游（商业街，晚上热闹）
    - 东湖听涛景区：散步,赏景,安静（人少、清净）
    - 武汉欢乐谷：热闹，溜娃（适合孩子去的都可以算作溜娃）
  反面例子（不要犯这类错）：
    - 纯商业街 → 不要勾"安静"
    - 纯寺庙 → 不要勾"夜游""热闹"


- "description"：一句话描述，20~40 字。
  写清楚这个地点是什么、在哪、有什么特点。

  只允许基于：名称、type、address、评分、景区等级。
  可以带全国公认的背景（如"武汉地标"），不确定的不写。

  不要写：具体的季节/花/树、价格/时间、活动/设施、年份/事件、
          人群词（适合情侣）、不确定的称号（天下第一）。

  例子：
    - 黄鹤楼 → 武昌蛇山上的古楼阁，临长江，武汉地标，5A 级景区。
    - 东湖 → 武汉知名湖泊，武昌区的开阔水域，适合散步观景。
    - 户部巷 → 武昌区的小吃街，武汉小吃聚集地，热闹有烟火气。


【输出格式】
严格返回 JSON，不要任何多余文字，不要用 markdown 代码块包裹。
示例：
{{"is_attraction": true, "tags": "历史人文,自然风光", "behaviors": "拍照,赏景,散步", "description": "武昌蛇山上的古楼阁，临长江，武汉地标，5A 级景区。"}}
{{"is_attraction": false, "tags": "", "behaviors": "", "description": ""}}
"""


def build_system_prompt() -> str:
    """从 schema.yaml 读词表，填进 System Prompt"""
    tags_options = " / ".join(get_interest_tags())
    behaviors_options = " / ".join(get_behavior_tags())
    return SYSTEM_PROMPT_TEMPLATE.format(
        tags_options=tags_options,
        behaviors_options=behaviors_options,
    )


def build_user_prompt(poi: dict) -> str:
    """把一条 POI 拼成给模型的输入"""
    return f"""名称：{poi.get('name', '')}
类型：{poi.get('type', '')}
地址：{poi.get('address', '')}
评分：{poi.get('rating', '')}
景区等级：{poi.get('poi_level', '') or '无'}"""


# ============================================================
# 初始化 LLM 客户端（模块级单例）
# ============================================================
def get_llm_client() -> OpenAI:
    """
    创建 OpenAI 兼容客户端。

    用 OpenAI SDK 调 DashScope 的兼容接口，
    只需改 base_url 和 api_key 即可。
    """
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError(
            "没读到 DASHSCOPE_API_KEY。请在项目根目录的 .env 文件里写入 "
            "DASHSCOPE_API_KEY=你的key"
        )

    return OpenAI(
        api_key=api_key,
        base_url=OPENAI_BASE_URL,
    )


# ============================================================
# 调 LLM 打标签
# ============================================================
def tag_one(client: OpenAI, system_prompt: str, poi: dict) -> dict:
    """
    给一条 POI 打标签。

    返回：{"is_attraction": bool, "tags": "...", "behaviors": "...", "description": "..."}
    失败时返回 is_attraction=True + 空标签（保守策略，避免误杀）。

    注意：失败时选择 True 而不是 False，是因为"误杀一个真景点"
    比"漏进一个非景点"代价更大。
    """
    user_prompt = build_user_prompt(poi)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=LLM_TEMPERATURE,
            extra_body={"enable_thinking": False},  # 关闭思考模式
        )
        raw = response.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"    [LLM 调用失败] {e}")
        return {"is_attraction": True, "tags": "", "behaviors": "", "description": ""}

    # 清理可能的 markdown 包裹
    raw = raw.replace("```json", "").replace("```", "").strip()

    # 解析 JSON
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning(f"    [JSON 解析失败] 原始返回前 80 字：{raw[:80]}")
        return {"is_attraction": True, "tags": "", "behaviors": "", "description": ""}

    # 解析 is_attraction，默认 True（保守）
    is_attraction = data.get("is_attraction", True)
    if not isinstance(is_attraction, bool):
        is_attraction = str(is_attraction).lower() in ("true", "1", "yes")

    tags = str(data.get("tags", "") or "").strip()
    behaviors = str(data.get("behaviors", "") or "").strip()
    description = str(data.get("description", "") or "").strip()

    # is_attraction=false 时，强制清空其余字段
    if not is_attraction:
        tags = ""
        behaviors = ""
        description = ""

    return {
        "is_attraction": is_attraction,
        "tags": tags,
        "behaviors": behaviors,
        "description": description,
    }


# ============================================================
# 断点续跑：读已处理的 id
# ============================================================
def load_done_ids(path: Path) -> set:
    """读输出文件，返回已处理的 id 集合。用于断点续跑。"""
    done = set()
    if not path.exists():
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                done.add(item.get("id"))
            except json.JSONDecodeError:
                continue
    return done


# ============================================================
# 打标签主流程
# ============================================================
def tag_city(city_key: str, force: bool = False) -> None:
    """给一个城市的所有 POI 打标签。force=True 清空重跑，False 断点续跑。"""
    logger.info("=" * 60)
    logger.info(f"开始打标签：{city_key}")
    logger.info("=" * 60)

    in_path = clean_path(city_key)
    out_path = tagged_path(city_key)

    if not in_path.exists():
        logger.error(f"输入文件不存在：{in_path}")
        logger.error(f"请先跑 clean：python -m src.clean.poi_cleaner --city {city_key}")
        return

    with open(in_path, "r", encoding="utf-8") as f:
        pois = json.load(f)
    logger.info(f"读入 clean POI：{len(pois)} 条")

    if force and out_path.exists():
        logger.info(f"--force 指定，删除旧输出：{out_path}")
        out_path.unlink()

    done_ids = load_done_ids(out_path)
    if done_ids:
        logger.info(f"断点续跑：已处理 {len(done_ids)} 条，跳过")

    # ---- 初始化 LLM ----
    system_prompt = build_system_prompt()
    client = get_llm_client()
    logger.info(f"LLM：{LLM_MODEL}（via OpenAI 兼容接口）")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_f = open(out_path, "a", encoding="utf-8")

    success, fail, skipped = 0, 0, 0
    not_attraction = 0

    try:
        for i, poi in enumerate(pois, start=1):
            pid = poi.get("id", "")
            name = poi.get("name", "未知")

            if pid in done_ids:
                skipped += 1
                continue

            logger.info(f"[{i}/{len(pois)}] {name}")

            tags = tag_one(client, system_prompt, poi)

            poi["is_attraction"] = tags["is_attraction"]
            poi["tags"] = tags["tags"]
            poi["behaviors"] = tags["behaviors"]
            poi["description"] = tags["description"]

            out_f.write(json.dumps(poi, ensure_ascii=False) + "\n")
            out_f.flush()

            if not tags["is_attraction"]:
                not_attraction += 1
                logger.info(f"    非景点 → 剔除")
            elif tags["tags"]:
                success += 1
                logger.info(f"    tags={tags['tags']} | behaviors={tags['behaviors']}")
            else:
                fail += 1

    finally:
        out_f.close()

    logger.info("")
    logger.info("=" * 60)
    logger.info(f"打标签完成")
    logger.info(f"  景点：{success} 条")
    logger.info(f"  非景点：{not_attraction} 条")
    logger.info(f"  解析失败：{fail} 条")
    logger.info(f"  跳过：{skipped} 条（断点续跑）")
    logger.info(f"输出：{out_path}")
    logger.info("=" * 60)


# ============================================================
# 命令行入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="POI 打标签")
    parser.add_argument(
        "--city",
        required=True,
        help=f"城市 key，可选：{list_cities()}",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重跑，清空已有输出",
    )
    args = parser.parse_args()

    tag_city(args.city, force=args.force)


if __name__ == "__main__":
    main()