"""
POI 打标签模块。

职责（大白话）：
1. 读 clean 后的 POI
2. 每条调一次通义千问，返回三个标签
3. 把标签写回 POI，保存

输入：data/processed/{city}_clean.json
输出：data/processed/{city}_tagged.json

三个标签：
- tags：兴趣标签（历史人文/自然风光/美食探店/购物休闲/夜生活）
- crowd：适合人群（独行/情侣/家庭/朋友/...）
- highlight：一句话亮点，30 字以内

设计要点：
1. 断点续跑：已处理的 id 跳过，支持中途崩溃后继续
2. 失败不中断：单条失败记日志、填空标签、继续下一条
3. 立即落盘：每条处理完就 append 到文件，防止崩溃丢数据
"""

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.messages import SystemMessage, HumanMessage

from src.config import (
    PROJECT_ROOT,
    clean_path,
    tagged_path,
    list_cities,
    get_interest_tags,
    get_crowd_tags,
)
from src.logger import get_logger

# 加载 .env（确保 DASHSCOPE_API_KEY 可用）
load_dotenv(PROJECT_ROOT / ".env")

logger = get_logger("tag")


# ============================================================
# LLM 配置
# ============================================================
LLM_MODEL = "qwen-turbo"     # 快、便宜，打标签够用
LLM_TEMPERATURE = 0.3        # 略随机，但整体稳定


# ============================================================
# Prompt 模板
# ============================================================
# 注意：这里是字符串模板，{interest_tags} 和 {crowd_tags} 在运行时填入
SYSTEM_PROMPT_TEMPLATE = """你是一个旅游数据标注专家。用户会给你一个景点的基本信息，你要给它打上标签。

请严格返回 JSON 格式，不要有任何多余文字、不要用 markdown 代码块包裹。

返回字段要求：
- "tags"：从以下词表中选 1~3 个最贴切的，用英文逗号分隔：{interest_tags}
- "crowd"：从以下词表中选 1~3 个最贴切的，用英文逗号分隔：{crowd_tags}
- "highlight"：一句话亮点，不超过 30 字，要有画面感

示例输出：
{{"tags": "历史人文,自然风光", "crowd": "历史爱好者,摄影爱好者", "highlight": "武汉地标，江南三大名楼之一，登顶可俯瞰长江"}}
"""


def build_system_prompt() -> str:
    """从 schema.yaml 读词表，填进 System Prompt"""
    interest_tags = ", ".join(get_interest_tags())
    crowd_tags = ", ".join(get_crowd_tags())
    return SYSTEM_PROMPT_TEMPLATE.format(
        interest_tags=interest_tags,
        crowd_tags=crowd_tags,
    )


def build_user_prompt(poi: dict) -> str:
    """
    把一条 POI 拼成给模型的输入。

    为什么只给这几个字段：
    - name、category、address、type 是判断标签的关键
    - rating、poi_level 是辅助参考
    - id、location、photos 对打标签无用，不给（避免干扰）
    """
    return f"""名称：{poi.get('name', '')}
类别：{poi.get('category', '')}
地址：{poi.get('address', '')}
类型标签：{poi.get('type', '')}
评分：{poi.get('rating', '')}
景区等级：{poi.get('poi_level', '') or '无'}"""


# ============================================================
# 调 LLM 打标签
# ============================================================
def tag_one(llm, system_prompt: str, poi: dict) -> dict:
    """
    给一条 POI 打标签。

    返回：{"tags": "...", "crowd": "...", "highlight": "..."}
    失败时返回空标签，不抛异常。
    """
    user_prompt = build_user_prompt(poi)

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]

    try:
        response = llm.invoke(messages)
        raw = response.content.strip()
    except Exception as e:
        logger.warning(f"    [LLM 调用失败] {e}")
        return {"tags": "", "crowd": "", "highlight": ""}

    # 清理可能的 markdown 包裹（```json ... ```）
    raw = raw.replace("```json", "").replace("```", "").strip()

    # 解析 JSON
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning(f"    [JSON 解析失败] 原始返回前 80 字：{raw[:80]}")
        return {"tags": "", "crowd": "", "highlight": ""}

    # 确保三个字段都存在，且是字符串
    return {
        "tags": str(data.get("tags", "") or ""),
        "crowd": str(data.get("crowd", "") or ""),
        "highlight": str(data.get("highlight", "") or ""),
    }


# ============================================================
# 断点续跑：读已处理的 id
# ============================================================
def load_done_ids(path: Path) -> set:
    """
    读输出文件，返回已处理的 id 集合。

    用于断点续跑：如果输出文件已存在，跳过里面已有的 id。
    """
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
    """
    给一个城市的所有 POI 打标签。

    force=True：清空已有输出，从头开始
    force=False：断点续跑，跳过已处理的
    """
    logger.info("=" * 60)
    logger.info(f"开始打标签：{city_key}")
    logger.info("=" * 60)

    in_path = clean_path(city_key)
    out_path = tagged_path(city_key)

    # ---- 检查输入 ----
    if not in_path.exists():
        logger.error(f"输入文件不存在：{in_path}")
        logger.error(f"请先跑 clean 阶段：python -m src.clean.poi_cleaner --city {city_key}")
        return

    # ---- 读 clean 后的 POI ----
    with open(in_path, "r", encoding="utf-8") as f:
        pois = json.load(f)
    logger.info(f"读入 clean POI：{len(pois)} 条")

    # ---- 处理 force 和断点 ----
    if force and out_path.exists():
        logger.info(f"--force 指定，删除旧输出：{out_path}")
        out_path.unlink()

    done_ids = load_done_ids(out_path)
    if done_ids:
        logger.info(f"断点续跑：已处理 {len(done_ids)} 条，跳过")

    # ---- 初始化 LLM ----
    system_prompt = build_system_prompt()
    llm = ChatTongyi(model=LLM_MODEL, temperature=LLM_TEMPERATURE)
    logger.info(f"LLM：{LLM_MODEL}（temperature={LLM_TEMPERATURE}）")

    # ---- 以追加模式打开输出文件 ----
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_f = open(out_path, "a", encoding="utf-8")

    success, fail, skipped = 0, 0, 0

    try:
        for i, poi in enumerate(pois, start=1):
            pid = poi.get("id", "")
            name = poi.get("name", "未知")

            # 断点续跑：跳过已处理的
            if pid in done_ids:
                skipped += 1
                continue

            logger.info(f"[{i}/{len(pois)}] {name}")

            # 调 LLM 打标签
            tags = tag_one(llm, system_prompt, poi)

            # 把标签写回 POI
            poi["tags"] = tags["tags"]
            poi["crowd"] = tags["crowd"]
            poi["highlight"] = tags["highlight"]

            # 立即落盘（防止崩溃丢数据）
            out_f.write(json.dumps(poi, ensure_ascii=False) + "\n")
            out_f.flush()

            if tags["tags"]:
                success += 1
                logger.info(f"    tags={tags['tags']} | crowd={tags['crowd']}")
                logger.info(f"    highlight={tags['highlight']}")
            else:
                fail += 1

    finally:
        out_f.close()

    # ---- 统计 ----
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"打标签完成")
    logger.info(f"  成功：{success} 条")
    logger.info(f"  失败：{fail} 条")
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