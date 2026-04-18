"""RAG 效果评测：对同一批 (resume, jd) 跑 rewrite(use_rag=True) vs rewrite(use_rag=False)，
用 LLM-as-a-Judge 做 pairwise 对比，量化 RAG 对改写质量的增益。

设计要点：
1. **Pairwise judge**：直接让评委对比 A/B 两版改写，避免绝对评分的不稳定性（已知 LLM 绝对分会漂）。
2. **顺序随机化**：每个 case 随机把 rag/norag 放 A 或 B，避免位置偏置（position bias）。
3. **多维打分**：关键词对齐 / 量化表达 / 表述专业度 / 真实性 —— 不是单一"谁更好"。
4. **输出**：汇总胜率 + 每 case 评语，可直接塞进 README 作为"RAG 是否有效"的量化证据。
"""
import json
import os
import random
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from agent.schemas import Resume, JobDescription, MatchReport
from agent.rewriter import rewrite

load_dotenv()
_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


JUDGE_PROMPT = """你是一个严格的简历改写质量评委。给定同一份简历、同一份 JD 和两版改写结果（A / B），
从以下 4 个维度各 0-5 分：

1. **keyword_alignment（关键词对齐）**：改写是否精准植入 JD 术语，而非堆砌无关关键词？
2. **quantification（量化表达）**：原文中的数字是否完整保留？改写后的表述是否维持"STAR + 量化"的专业粒度？
3. **professionalism（表述专业度）**：语言是否像业内资深简历的措辞？避免套话/口水话？
4. **authenticity（真实性）**：是否没有编造简历里没有的项目 / 数据 / 经历？

最后给出 winner ∈ {A, B, tie}，以及一句对比评语 rationale。

严格输出 JSON：
{
  "a_scores": {"keyword_alignment": int, "quantification": int, "professionalism": int, "authenticity": int},
  "b_scores": {"keyword_alignment": int, "quantification": int, "professionalism": int, "authenticity": int},
  "winner": "A" | "B" | "tie",
  "rationale": "..."
}
"""


def _rewrite_to_text(r) -> str:
    parts = []
    if r.rewritten_summary:
        parts.append(f"【改写 summary】{r.rewritten_summary}")
    for rb in r.rewritten_bullets:
        parts.append(f"- [{rb.experience_company}] BEFORE: {rb.before}\n  AFTER: {rb.after}\n  REASON: {rb.change_reason}")
    if r.new_keywords_added:
        parts.append(f"【植入关键词】{' / '.join(r.new_keywords_added)}")
    return "\n".join(parts) if parts else "（无改写输出）"


def pairwise_judge(resume: Resume, jd: JobDescription, text_a: str, text_b: str) -> dict:
    user_msg = (
        f"【简历】\n{json.dumps(resume.model_dump(), ensure_ascii=False)[:2000]}\n\n"
        f"【JD】\n{json.dumps(jd.model_dump(), ensure_ascii=False)[:1500]}\n\n"
        f"【A 版改写】\n{text_a}\n\n"
        f"【B 版改写】\n{text_b}\n"
    )
    resp = _client.chat.completions.create(
        model=_MODEL,
        messages=[
            {"role": "system", "content": JUDGE_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    return json.loads(resp.choices[0].message.content)


def run_rag_eval(cases: list[tuple[str, Resume, JobDescription, MatchReport]]) -> dict:
    results = []
    rag_wins = norag_wins = ties = 0
    rag_total = norag_total = 0

    for name, resume, jd, report in cases:
        print(f"\n▶ 用例：{name}")
        r_rag = rewrite(resume, jd, report, use_rag=True)
        r_norag = rewrite(resume, jd, report, use_rag=False)

        t_rag = _rewrite_to_text(r_rag)
        t_norag = _rewrite_to_text(r_norag)

        # 随机化位置避免 position bias
        rag_is_a = random.random() < 0.5
        if rag_is_a:
            verdict = pairwise_judge(resume, jd, t_rag, t_norag)
            a_label, b_label = "rag", "norag"
        else:
            verdict = pairwise_judge(resume, jd, t_norag, t_rag)
            a_label, b_label = "norag", "rag"

        winner_label = {"A": a_label, "B": b_label, "tie": "tie"}[verdict["winner"]]
        rag_scores = verdict["a_scores"] if rag_is_a else verdict["b_scores"]
        norag_scores = verdict["b_scores"] if rag_is_a else verdict["a_scores"]
        rag_sum = sum(rag_scores.values())
        norag_sum = sum(norag_scores.values())

        if winner_label == "rag":
            rag_wins += 1
        elif winner_label == "norag":
            norag_wins += 1
        else:
            ties += 1

        rag_total += rag_sum
        norag_total += norag_sum

        print(f"  rag   总分={rag_sum}/20  detail={rag_scores}")
        print(f"  norag 总分={norag_sum}/20 detail={norag_scores}")
        print(f"  winner: {winner_label}  |  {verdict['rationale']}")

        results.append({
            "case_name": name,
            "rag_scores": rag_scores,
            "norag_scores": norag_scores,
            "rag_total": rag_sum,
            "norag_total": norag_sum,
            "winner": winner_label,
            "rationale": verdict["rationale"],
            "rag_references_count": len(r_rag.rag_references),
        })

    n = len(cases)
    summary = {
        "total_cases": n,
        "rag_wins": rag_wins,
        "norag_wins": norag_wins,
        "ties": ties,
        "rag_avg_total": round(rag_total / n, 2),
        "norag_avg_total": round(norag_total / n, 2),
        "cases": results,
    }
    return summary


if __name__ == "__main__":
    random.seed(42)

    base = Path("data/outputs")
    resume = Resume.model_validate_json((base / "my_resume.json").read_text(encoding="utf-8"))

    jd_files = [
        ("high_match", "jd_high.json", "match_my_resume__jd_high.json"),
        ("medium_match", "jd_medium.json", "match_my_resume__jd_medium.json"),
        ("low_match", "jd_low.json", "match_my_resume__jd_low.json"),
    ]
    cases = []
    for name, jdf, mrf in jd_files:
        jd = JobDescription.model_validate_json((base / jdf).read_text(encoding="utf-8"))
        mr = MatchReport.model_validate_json((base / mrf).read_text(encoding="utf-8"))
        cases.append((name, resume, jd, mr))

    summary = run_rag_eval(cases)

    out = base / "rag_eval_report.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print(f"📊 RAG 评测汇总（{summary['total_cases']} 个用例）")
    print("=" * 60)
    print(f"rag   平均分：{summary['rag_avg_total']} / 20")
    print(f"norag 平均分：{summary['norag_avg_total']} / 20")
    print(f"胜负：rag={summary['rag_wins']}  norag={summary['norag_wins']}  tie={summary['ties']}")
    print(f"\n✅ 详细报告：{out}")
