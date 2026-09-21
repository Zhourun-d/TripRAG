"""
评估指标计算。

三个指标：
- Hit Rate@K：期望结果里，有多少出现在 top-K
- MRR：第一个期望结果排在第几，取倒数
- Noise Rate@K：不该出现的结果里，有多少混进了 top-K

匹配方式：包含匹配
  期望结果 "黄鹤楼" 只要被 top_names 里任意一条包含，就算命中。
  比如 top_names 里有 "黄鹤楼公园"，也算命中 "黄鹤楼"。

为什么用包含匹配：
  高德返回的名字常带后缀，如 "户部巷小吃一条街"、
  "吉庆街美食生活区"。如果用精确匹配，评估集会全部判错。
"""

from dataclasses import dataclass, field


# ============================================================
# 数据结构
# ============================================================
@dataclass
class CaseResult:
    """单条评估用例的结果"""
    case_id: str
    query: str
    case_type: str
    difficulty: str

    top_names: list = field(default_factory=list)   # 实际 top-K 的名字
    expected: list = field(default_factory=list)    # 期望的名字
    must_not: list = field(default_factory=list)    # 不该出现的名字

    hit: float = 0.0
    rr: float = 0.0
    noise: float = 0.0
    missed: list = field(default_factory=list)      # 没命中的期望结果
    unexpected: list = field(default_factory=list)  # 混入的不该出现结果


# ============================================================
# 匹配工具
# ============================================================
def _is_hit(expected_name: str, top_names: list) -> bool:
    """
    判断一个期望名字是否被 top_names 命中（包含匹配）。

    例子：
      expected_name = "黄鹤楼"
      top_names = ["黄鹤楼公园", "晴川阁", ...]
      → True（因为 "黄鹤楼公园" 包含 "黄鹤楼"）

      expected_name = "户部巷"
      top_names = ["户部巷小吃一条街", ...]
      → True
    """
    return any(expected_name in name for name in top_names)


# ============================================================
# 三个指标
# ============================================================
def compute_hit(top_names: list, expected: list) -> float:
    """
    Hit Rate：期望结果出现在 top 里的比例。

    例子：
      expected = [黄鹤楼, 晴川阁, 古德寺, 归元禅寺]
      top = [黄鹤楼, 江汉关大楼, 晴川阁, ...]
      命中 2 个 → 2/4 = 0.5
    """
    if not expected:
        return 1.0
    hit_count = sum(1 for exp in expected if _is_hit(exp, top_names))
    return hit_count / len(expected)


def compute_rr(top_names: list, expected: list) -> float:
    """
    Reciprocal Rank：第一个期望结果的排名倒数。

    例子：
      期望 [黄鹤楼, 晴川阁, ...]，top = [户部巷, 黄鹤楼, 晴川阁, ...]
      第一个命中是 "黄鹤楼"，排第 2 位 → RR = 1/2 = 0.5
    """
    if not expected:
        return 0.0
    for i, name in enumerate(top_names, start=1):
        for exp in expected:
            if exp in name:
                return 1.0 / i
    return 0.0


def compute_noise(top_names: list, must_not: list) -> float:
    """
    Noise Rate：不该出现的结果混入 top 的比例。

    例子：
      must_not = [光谷步行街, 楚河汉街]
      top = [黄鹤楼, 光谷步行街, ...]
      混入 1 个 → 1/2 = 0.5
    """
    if not must_not:
        return 0.0
    noise_count = sum(1 for bad in must_not if _is_hit(bad, top_names))
    return noise_count / len(must_not)


# ============================================================
# 单条用例评估
# ============================================================
def evaluate_case(case: dict, top_names: list) -> CaseResult:
    """对一条评估用例，算所有指标"""
    expected = case.get("expected_names", [])
    must_not = case.get("must_not_include", [])

    result = CaseResult(
        case_id=case["id"],
        query=case["query"],
        case_type=case.get("type", ""),
        difficulty=case.get("difficulty", ""),
        top_names=top_names,
        expected=expected,
        must_not=must_not,
    )

    result.hit = compute_hit(top_names, expected)
    result.rr = compute_rr(top_names, expected)
    result.noise = compute_noise(top_names, must_not)

    # 包含匹配：哪些期望结果漏了、哪些不该出现的混入了
    result.missed = [exp for exp in expected if not _is_hit(exp, top_names)]
    result.unexpected = [bad for bad in must_not if _is_hit(bad, top_names)]

    return result


# ============================================================
# 聚合统计
# ============================================================
def aggregate(results: list) -> dict:
    """对多条用例结果，算平均值"""
    if not results:
        return {"count": 0, "hit_rate": 0.0, "mrr": 0.0, "noise_rate": 0.0}

    n = len(results)
    return {
        "count": n,
        "hit_rate": sum(r.hit for r in results) / n,
        "mrr": sum(r.rr for r in results) / n,
        "noise_rate": sum(r.noise for r in results) / n,
    }