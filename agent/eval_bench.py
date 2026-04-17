"""Prompt 评测台：对比两版 matcher prompt 在一组测试用例上的质量。

评测流程：
  for each (resume, jd) in 测试集:
      v1_report = matcher(prompt=v1)
      v2_report = matcher(prompt=v2)
      v1_eval   = judge(v1_report, resume, jd)
      v2_eval   = judge(v2_report, resume, jd)
      winner    = 比较 total
  汇总：胜率、平均分、逐案对比

方法论来源：候选人毕业论文"Schema 约束 + 多轮规则迭代"的 LLM 评估体系。
扩展到产品场景时，用 LLM-as-a-Judge 做低成本人工代理，并在 top cases 上人工复核。
"""
import json
import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

from agent.schemas import (
    Resume, JobDescription, MatchReport,
    EvalResult, CaseResult, BenchReport,
)
from agent.matcher import match

load_dotenv()
_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

JUDGE_PROMPT = """你是一个严格的 LLM 输出质量评委。你会看到：
1. 候选人简历 JSON
2. 目标 JD JSON
3. 某个匹配 Agent 产出的 MatchReport

请从以下三个维度评分（每项 0-5 分），并用一句话说明理由：

1. **honesty（诚实度）**：
   - overall_score 是否合理反映简历与 JD 的真实匹配度？
   - 是否存在"soft_score 虚高拉抬综合分"的问题？
   - 低匹配场景是否诚实给低分，高匹配场景是否给足高分？

2. **specificity（具体度）**：
   - suggestion 和 top_actions 是否到"在哪段经历下加什么内容"的粒度？
   - 是否避免了"增加项目经验""强调相关技能"这类空话？

3. **coverage（覆盖度）**：
   - gaps 是否覆盖 JD 所有 hard_requirements 和 soft_preferences 条目？
   - evidence 在 full/partial 时是否都有原文引用？

最后输出 total = honesty + specificity + coverage。

严格按 JSON Schema 输出，不要任何解释。
"""


def judge(report: MatchReport, resume: Resume, jd: JobDescription) -> EvalResult:
    user_msg = (
        f"简历：\n{json.dumps(resume.model_dump(), ensure_ascii=False)}\n\n"
        f"JD：\n{json.dumps(jd.model_dump(), ensure_ascii=False)}\n\n"
        f"待评 MatchReport：\n{json.dumps(report.model_dump(), ensure_ascii=False)}\n\n"
        f"Schema：\n{json.dumps(EvalResult.model_json_schema(), ensure_ascii=False)}"
    )
    response = _client.chat.completions.create(
        model=_MODEL,
        messages=[
            {"role": "system", "content": JUDGE_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    return EvalResult.model_validate_json(response.choices[0].message.content)


def run_bench(cases: list[tuple[str, Resume, JobDescription]]) -> BenchReport:
    """cases: [(name, resume, jd), ...]"""
    case_results = []
    v1_wins = v2_wins = ties = 0
    v1_totals = v2_totals = 0

    for name, resume, jd in cases:
        print(f"\n▶ 运行用例：{name}")
        v1_report = match(resume, jd, prompt_version="v1_baseline")
        v2_report = match(resume, jd, prompt_version="v2_fixed")
        v1_eval = judge(v1_report, resume, jd)
        v2_eval = judge(v2_report, resume, jd)

        if v1_eval.total > v2_eval.total:
            winner = "v1"
            v1_wins += 1
        elif v2_eval.total > v1_eval.total:
            winner = "v2"
            v2_wins += 1
        else:
            winner = "tie"
            ties += 1

        v1_totals += v1_eval.total
        v2_totals += v2_eval.total

        print(f"  v1 total={v1_eval.total}  v2 total={v2_eval.total}  winner={winner}")

        case_results.append(CaseResult(
            case_name=name,
            v1_report=v1_report.model_dump(),
            v2_report=v2_report.model_dump(),
            v1_eval=v1_eval,
            v2_eval=v2_eval,
            winner=winner,
        ))

    n = len(cases)
    return BenchReport(
        total_cases=n,
        v1_wins=v1_wins,
        v2_wins=v2_wins,
        ties=ties,
        v1_avg_total=round(v1_totals / n, 2),
        v2_avg_total=round(v2_totals / n, 2),
        cases=case_results,
    )


if __name__ == "__main__":
    # 约定：data/outputs/my_resume.json + jd_high.json / jd_medium.json / jd_low.json
    resume_path = Path("data/outputs/my_resume.json")
    jd_files = {
        "high_match": "data/outputs/jd_high.json",
        "medium_match": "data/outputs/jd_medium.json",
        "low_match": "data/outputs/jd_low.json",
    }

    resume = Resume.model_validate_json(resume_path.read_text(encoding="utf-8"))
    cases = [
        (name, resume, JobDescription.model_validate_json(Path(p).read_text(encoding="utf-8")))
        for name, p in jd_files.items()
    ]

    report = run_bench(cases)

    output_path = Path("data/outputs/bench_report.json")
    output_path.write_text(
        json.dumps(report.model_dump(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 60)
    print(f"📊 评测汇总（{report.total_cases} 个用例）")
    print("=" * 60)
    print(f"v1_baseline  平均分：{report.v1_avg_total} / 15")
    print(f"v2_fixed     平均分：{report.v2_avg_total} / 15")
    print(f"胜负：v1={report.v1_wins}  v2={report.v2_wins}  tie={report.ties}")
    print(f"\n✅ 详细报告已保存到：{output_path}")
