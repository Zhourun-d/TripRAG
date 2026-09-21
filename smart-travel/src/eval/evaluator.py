"""
评估脚本。

职责（大白话）：
1. 读评估集（wuhan_eval.json）
2. 对每条 query 跑检索
3. 算 Hit Rate / MRR / Noise Rate
4. 输出报告（控制台 + JSON）

输入：
  - data/eval/{city}_eval.json
  - chroma_db/{city}_v1/
输出：
  - data/eval/reports/{city}_report.json
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from src.config import (
    PROJECT_ROOT,
    eval_path,
    report_path,
    CHROMA_DIR,
    list_cities,
)
from src.retrieve.retriever import Retriever
from src.eval.metrics import evaluate_case, aggregate
from src.logger import get_logger

logger = get_logger("eval")


# 城市 key -> 城市名映射
CITY_NAME_MAP = {"wuhan": "武汉", "chongqing": "重庆", "nanjing": "南京"}


# ============================================================
# 读评估集
# ============================================================
def load_eval_set(city_key: str) -> dict:
    path = eval_path(city_key)
    if not path.exists():
        raise FileNotFoundError(f"评估集不存在：{path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# 跑评估
# ============================================================
def run_eval(city_key: str, top_k: int = 10) -> dict:
    logger.info("=" * 60)
    logger.info(f"开始评估：{city_key}")
    logger.info("=" * 60)

    # ---- 读评估集 ----
    eval_set = load_eval_set(city_key)
    cases = eval_set["cases"]
    city_name = CITY_NAME_MAP.get(city_key, city_key)

    logger.info(f"评估集：{len(cases)} 条用例")
    logger.info(f"城市名：{city_name}")
    logger.info(f"top_k：{top_k}")

    # ---- 加载检索器 ----
    collection_name = f"{city_key}_v1"
    retriever = Retriever(collection_name)

    # ---- 逐条评估 ----
    case_results = []
    for case in cases:
        query = case["query"]

        # 跑检索
        results = retriever.search(query, city=city_name, top_k=top_k)
        top_names = [r.name for r in results]

        # 算指标
        case_result = evaluate_case(case, top_names)
        case_results.append(case_result)

    # ---- 整体指标 ----
    overall = aggregate(case_results)

    # ---- 按 type 分组 ----
    by_type = defaultdict(list)
    for r in case_results:
        by_type[r.case_type].append(r)
    type_stats = {k: aggregate(v) for k, v in by_type.items()}

    # ---- 按 difficulty 分组 ----
    by_diff = defaultdict(list)
    for r in case_results:
        by_diff[r.difficulty].append(r)
    diff_stats = {k: aggregate(v) for k, v in by_diff.items()}

    # ---- 输出报告 ----
    print_report(overall, type_stats, diff_stats, case_results)

    # ---- 保存 JSON 报告 ----
    report = {
        "city": city_key,
        "top_k": top_k,
        "overall": overall,
        "by_type": type_stats,
        "by_difficulty": diff_stats,
        "cases": [
            {
                "id": r.case_id,
                "query": r.query,
                "type": r.case_type,
                "difficulty": r.difficulty,
                "hit": r.hit,
                "rr": r.rr,
                "noise": r.noise,
                "top_names": r.top_names,
                "expected": r.expected,
                "must_not": r.must_not,
                "missed": r.missed,
                "unexpected": r.unexpected,
            }
            for r in case_results
        ],
    }

    out_path = report_path(city_key)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logger.info(f"报告已保存：{out_path}")

    return report


# ============================================================
# 控制台报告
# ============================================================
def print_report(overall, type_stats, diff_stats, case_results):
    print()
    print("=" * 60)
    print("检索评估报告")
    print("=" * 60)
    print(f"【整体指标】")
    print(f"  Hit Rate@{10}: {overall['hit_rate']:.3f}")
    print(f"  MRR:          {overall['mrr']:.3f}")
    print(f"  Noise Rate:   {overall['noise_rate']:.3f}")
    print()

    print("【按 type 分组】")
    for t, s in type_stats.items():
        print(f"  {t:12s} ({s['count']:2d}条): "
              f"Hit={s['hit_rate']:.2f}  MRR={s['mrr']:.2f}  Noise={s['noise_rate']:.2f}")
    print()

    print("【按 difficulty 分组】")
    for d, s in diff_stats.items():
        print(f"  {d:8s} ({s['count']:2d}条): "
              f"Hit={s['hit_rate']:.2f}  MRR={s['mrr']:.2f}  Noise={s['noise_rate']:.2f}")
    print()

    # ---- 失败案例 ----
    # 按 hit 升序取前 3 条（漏得最多的）
    print("【失败案例：漏得最多 top 3】")
    sorted_by_hit = sorted(case_results, key=lambda x: x.hit)
    for r in sorted_by_hit[:3]:
        print(f"  [{r.case_id}] Hit={r.hit:.2f} Noise={r.noise:.2f}")
        print(f"    query: {r.query}")
        print(f"    期望: {r.expected}")
        print(f"    实际 top5: {r.top_names[:5]}")
        print(f"    漏掉: {r.missed}")
        print()

    # 按 noise 降序取前 3 条（混得最多的）
    print("【失败案例：混入最多 top 3】")
    sorted_by_noise = sorted(case_results, key=lambda x: -x.noise)
    for r in sorted_by_noise[:3]:
        if r.noise == 0:
            continue
        print(f"  [{r.case_id}] Noise={r.noise:.2f} Hit={r.hit:.2f}")
        print(f"    query: {r.query}")
        print(f"    不该出现: {r.must_not}")
        print(f"    混入: {r.unexpected}")
        print()

    print("=" * 60)


# ============================================================
# 命令行入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="检索评估")
    parser.add_argument("--city", required=True, help=f"城市 key：{list_cities()}")
    parser.add_argument("--top_k", type=int, default=10)
    args = parser.parse_args()

    run_eval(args.city, top_k=args.top_k)


if __name__ == "__main__":
    main()