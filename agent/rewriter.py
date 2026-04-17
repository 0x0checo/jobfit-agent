"""改写 Agent：基于 MatchReport 的 top_actions 和 keyword_misses，
针对性改写简历 bullet 和 summary，保证：
1. 事实不造假（只重组现有经历的表述，不编造新项目）
2. 植入 JD 关键词（对齐招聘方术语）
3. 显式 before/after + 改写理由，便于人工复核
"""
import json
import os
from dotenv import load_dotenv
from openai import OpenAI

from agent.schemas import Resume, JobDescription, MatchReport, RewriteResult

load_dotenv()
_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

SYSTEM_PROMPT = """你是一个资深简历改写顾问。基于候选人简历、目标 JD、匹配报告，
针对性改写简历 bullet 和 summary，以提高匹配度。

【核心约束 —— 严格遵守】
1. **事实不造假**：不允许编造简历里没有的项目、数据、技术栈。只能对已有内容做"重新表述/侧重点调整/植入 JD 关键词"。
2. **量化数据必须保留**：原 bullet 里的数字（如 "+183%""80-90%"）改写后必须完整保留。
3. **改什么**：优先改 match_level=partial 或 missing 的对应经历；full 的原则上不改。

【改写策略】
- bullet 改写：把候选人做过的事，用 JD 关键词重新包装（如原文"LLM提取策略"→改"大模型 Prompt 工程与 Agent 调优"）。
- summary 改写：把个人定位向 JD 方向倾斜，突出最相关经验。
- new_keywords_added 记录本次植入的 JD 关键词（来自 keyword_misses）。

【输出要求】
- rewritten_bullets: 只输出**需要改**的 bullet，不改的不放。每条必须有 before / after / change_reason。
- change_reason 格式："对应 JD 第 X 条要求「…」，植入关键词「…」"。

只输出 JSON，不要任何解释。
"""


def rewrite(resume: Resume, jd: JobDescription, report: MatchReport) -> RewriteResult:
    user_msg = (
        f"【简历】\n{json.dumps(resume.model_dump(), ensure_ascii=False)}\n\n"
        f"【JD】\n{json.dumps(jd.model_dump(), ensure_ascii=False)}\n\n"
        f"【匹配报告】\n{json.dumps(report.model_dump(), ensure_ascii=False)}\n\n"
        f"【输出 Schema】\n{json.dumps(RewriteResult.model_json_schema(), ensure_ascii=False)}"
    )
    response = _client.chat.completions.create(
        model=_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        response_format={"type": "json_object"},
        temperature=0.3,
    )
    return RewriteResult.model_validate_json(response.choices[0].message.content)


def render_markdown(resume: Resume, result: RewriteResult) -> str:
    """把改写结果渲染成可下载的 Markdown 简历（before/after 对比视图）。"""
    lines = [f"# {resume.name} · 简历（改写版）\n"]

    # Summary
    lines.append("## 个人定位\n")
    if result.rewritten_summary:
        lines.append(f"**改写后**：{result.rewritten_summary}\n")
        lines.append(f"> 原版：{resume.summary or '-'}\n")
        if result.summary_change_reason:
            lines.append(f"> 改写理由：{result.summary_change_reason}\n")
    else:
        lines.append(f"{resume.summary or '-'}\n")

    # 改写的 bullets 索引
    rewrite_map = {}
    for rb in result.rewritten_bullets:
        rewrite_map.setdefault(rb.experience_company, {})[rb.before] = rb

    # 经历
    lines.append("\n## 实践经历\n")
    for exp in resume.experience:
        lines.append(f"### {exp.company} · {exp.role}  ({exp.start_date} - {exp.end_date})\n")
        for b in exp.bullets:
            rb = rewrite_map.get(exp.company, {}).get(b)
            if rb:
                lines.append(f"- ✨ **{rb.after}**")
                lines.append(f"  - *原文*：{rb.before}")
                lines.append(f"  - *改写理由*：{rb.change_reason}")
            else:
                lines.append(f"- {b}")
        lines.append("")

    # 新关键词
    if result.new_keywords_added:
        lines.append("\n## 本次植入关键词\n")
        lines.append(" · ".join([f"`{k}`" for k in result.new_keywords_added]))

    if result.notes:
        lines.append(f"\n## 改写策略\n{result.notes}\n")

    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    resume = Resume.model_validate_json(Path(sys.argv[1]).read_text(encoding="utf-8"))
    jd = JobDescription.model_validate_json(Path(sys.argv[2]).read_text(encoding="utf-8"))
    report = MatchReport.model_validate_json(Path(sys.argv[3]).read_text(encoding="utf-8"))

    result = rewrite(resume, jd, report)
    md = render_markdown(resume, result)

    out = Path("data/outputs/rewrite_preview.md")
    out.write_text(md, encoding="utf-8")
    print(md)
    print(f"\n✅ 已保存到：{out}")
