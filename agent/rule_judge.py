"""规则型评估器（Rule-based Judge）—— 补 LLM Judge 的盲区。

动机：
  模块 4 v1 评测中发现，LLM-as-Judge 抓得到"建议是否空泛"这类语义问题，
  但抓不到"hard_score 数学算错"这类数值问题。
  本模块用确定性规则做校验，与 LLM Judge 形成双通道。

规则集：
  1. math_hard_score      hard_score 应约等于 gaps 里 hard 条目 (full=100/partial=50/missing=0) 的均值
  2. math_overall_score   overall_score 公式校验（考虑 soft_score 为 None 的情况）
  3. coverage_hard        gaps 覆盖所有 hard_requirements
  4. coverage_soft        gaps 覆盖所有 soft_preferences
  5. evidence_non_empty   match_level=full/partial 时 evidence 非空
  6. suggestion_non_empty 每条 gap 都有 suggestion
  7. top_actions_size     top_actions 数量在 3-5 条

面试价值：
  把"LLM 评测的盲区"作为真实迭代案例，引出"规则 + LLM 双通道"的评测体系设计。
  这对应"大模型评测经验"要求里少有人深入的细节。
"""
from typing import Tuple

from agent.schemas import Resume, JobDescription, MatchReport, RuleCheck, RuleEvalResult


_LEVEL_SCORE = {"full": 100, "partial": 50, "missing": 0}


def _check_math_hard(report: MatchReport) -> RuleCheck:
    hard_gaps = [g for g in report.gaps if g.req_type == "hard"]
    if not hard_gaps:
        return RuleCheck(name="math_hard_score", passed=True, detail="无硬性要求")
    expected = round(sum(_LEVEL_SCORE.get(g.match_level, 0) for g in hard_gaps) / len(hard_gaps))
    passed = abs(report.hard_score - expected) <= 10  # 容忍 ±10 分
    detail = f"expected≈{expected}, got={report.hard_score}"
    return RuleCheck(name="math_hard_score", passed=passed, detail=detail)


def _check_math_overall(report: MatchReport) -> RuleCheck:
    if report.soft_score is None:
        expected = report.hard_score
    else:
        expected = round(report.hard_score * 0.7 + report.soft_score * 0.3)
    passed = abs(report.overall_score - expected) <= 5
    return RuleCheck(
        name="math_overall_score",
        passed=passed,
        detail=f"expected={expected}, got={report.overall_score}",
    )


def _check_coverage(report: MatchReport, jd: JobDescription) -> Tuple[RuleCheck, RuleCheck]:
    hard_reqs = set(jd.hard_requirements)
    soft_reqs = set(jd.soft_preferences)
    covered = {g.requirement for g in report.gaps}

    hard_missed = hard_reqs - covered
    soft_missed = soft_reqs - covered
    return (
        RuleCheck(
            name="coverage_hard",
            passed=len(hard_missed) == 0,
            detail=f"未覆盖 {len(hard_missed)}/{len(hard_reqs)} 条" if hard_missed else "全部覆盖",
        ),
        RuleCheck(
            name="coverage_soft",
            passed=len(soft_missed) == 0,
            detail=f"未覆盖 {len(soft_missed)}/{len(soft_reqs)} 条" if soft_missed else "全部覆盖",
        ),
    )


def _check_evidence(report: MatchReport) -> RuleCheck:
    missing = [g for g in report.gaps if g.match_level in ("full", "partial") and not (g.evidence or "").strip()]
    return RuleCheck(
        name="evidence_non_empty",
        passed=len(missing) == 0,
        detail=f"{len(missing)} 条 full/partial 缺 evidence" if missing else "全部有引用",
    )


def _check_suggestion(report: MatchReport) -> RuleCheck:
    missing = [g for g in report.gaps if not (g.suggestion or "").strip()]
    return RuleCheck(
        name="suggestion_non_empty",
        passed=len(missing) == 0,
        detail=f"{len(missing)} 条 gap 缺 suggestion" if missing else "全部有建议",
    )


def _check_top_actions(report: MatchReport) -> RuleCheck:
    n = len(report.top_actions)
    return RuleCheck(
        name="top_actions_size",
        passed=3 <= n <= 5,
        detail=f"共 {n} 条（目标 3-5 条）",
    )


def rule_judge(report: MatchReport, jd: JobDescription) -> RuleEvalResult:
    checks = [
        _check_math_hard(report),
        _check_math_overall(report),
        *_check_coverage(report, jd),
        _check_evidence(report),
        _check_suggestion(report),
        _check_top_actions(report),
    ]
    pass_rate = sum(1 for c in checks if c.passed) / len(checks)
    score = round(pass_rate * 5)
    return RuleEvalResult(checks=checks, pass_rate=round(pass_rate, 2), score=score)
