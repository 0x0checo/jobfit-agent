"""模拟面试官 Agent：基于 Resume + JD + MatchReport，生成针对性面试题并支持多轮追问。

设计要点：
1. **三类题型**对应真实面试三段式：
   - resume_deepdive：针对简历 bullets 追问 how/why/metric
   - gap_probe：针对 MatchReport.gaps 中 partial/missing 的硬性要求探测
   - domain_open：按 JD 关键词出岗位专业开放题
2. **三种人设**（persona）：
   - tech：技术面，偏追问实现细节、技术选型、权衡
   - product：产品面（字节 AIPM 风格），偏 case / 产品 sense / 数据驱动
   - hr：HR 终面，偏动机 / 抗压 / 团队合作 / 职业规划
3. **pipeline 复利**：接 MatchReport 而非裸 JD —— 差距题只能从匹配报告推出，体现串联价值。
4. **追问机制**：用户答完 → LLM 判断答案是否足够深入 → 需要则生成 follow-up（模拟真实面试"再具体说说"）。
"""
import json
import os
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

from agent.schemas import Resume, JobDescription, MatchReport, InterviewSet, FollowUpResult

load_dotenv()
_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


PERSONA_STYLES = {
    "tech": {
        "label": "🧑‍💻 技术面试官",
        "style": "字节技术线 AIPM 面试官。追问实现细节、技术选型原因、权衡取舍。不接受空泛回答，要追问到架构/指标层面。",
        "question_focus": "侧重 Agent 架构、Prompt 工程、评测体系、RAG 设计、LLM 能力边界等技术细节的追问。",
    },
    "product": {
        "label": "🎯 产品面试官",
        "style": "字节产品线面试官。偏 case 思维、产品 sense、数据驱动、用户价值。喜欢让候选人从 0 设计功能或拆解现有产品。",
        "question_focus": "侧重产品决策、用户场景、指标体系、迭代方法、取舍逻辑等产品思维的考察。",
    },
    "hr": {
        "label": "💬 HR 终面",
        "style": "字节 HRBP 终面。偏动机 / 抗压 / 团队合作 / 职业规划 / 价值观。温和但犀利，会挖动机的真实性。",
        "question_focus": "侧重选择动机、失败经历、冲突处理、长期规划、对字节文化的理解。",
    },
}


SYSTEM_PROMPT = """你是字节跳动 AI 产品经理岗位的资深面试官，正在为候选人准备一场针对性模拟面试。

【你的人设】
{persona_style}

{question_focus}

【任务】
基于候选人的简历、目标 JD 和匹配报告，生成 3 类共 9 道面试题：

1. **resume_deepdive（3 题）**：针对简历中的具体 bullet / 项目深挖。
   - 必须引用简历里的真实项目 / 数字
   - 问"为什么这么做 / 量化指标怎么来的 / 如果让你重做会怎么改"
   - linked_requirement 填简历里的原文片段

2. **gap_probe（3 题）**：针对 MatchReport.gaps 中 match_level=partial 或 missing 的硬性要求。
   - 委婉而犀利地探测候选人是否真的会
   - 例："你简历里没看到 X 经验，但遇到过类似 Y 场景吗？怎么处理的？"
   - linked_requirement 填 JD 中对应的 requirement 原文

3. **domain_open（3 题）**：基于 JD 关键词 / 团队业务的开放题。
   - 不依赖简历具体内容，考察岗位专业 sense
   - 例："如果让你负责本岗位业务，怎么设计 XX 的评测体系"
   - linked_requirement 填对应的 JD 关键词

【每题必须包含】
- question：问题本身（口语化，像真人面试官）
- intent：考察意图（你作为面试官为什么问这题）
- answer_hints：3-5 条参考答题要点（STAR 关键锚点）
- difficulty：easy / medium / hard
- linked_requirement：对应简历片段或 JD 要求

【难度分布建议】
每类里 1 题 easy（破冰/基础）、1 题 medium（标准深度）、1 题 hard（压力/陷阱题）。

严格输出 JSON，符合给定 schema。
"""


def generate_questions(
    resume: Resume,
    jd: JobDescription,
    report: MatchReport,
    persona: str = "product",
) -> InterviewSet:
    """生成一套完整的模拟面试题。"""
    if persona not in PERSONA_STYLES:
        persona = "product"
    p = PERSONA_STYLES[persona]

    schema = InterviewSet.model_json_schema()

    system_msg = SYSTEM_PROMPT.format(
        persona_style=p["style"],
        question_focus=p["question_focus"],
    )

    user_msg = (
        f"【候选人简历】\n{json.dumps(resume.model_dump(), ensure_ascii=False)[:3000]}\n\n"
        f"【目标 JD】\n{json.dumps(jd.model_dump(), ensure_ascii=False)[:2000]}\n\n"
        f"【匹配报告】\n{json.dumps(report.model_dump(), ensure_ascii=False)[:2500]}\n\n"
        f"【persona 字段请填】：{persona}\n"
        f"【persona_style_note 字段请填】：{p['style']}\n\n"
        f"【输出 Schema】\n{json.dumps(schema, ensure_ascii=False)}"
    )

    response = _client.chat.completions.create(
        model=_MODEL,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        response_format={"type": "json_object"},
        temperature=0.6,  # 面试题适度多样性
    )
    return InterviewSet.model_validate_json(response.choices[0].message.content)


FOLLOWUP_PROMPT = """你是一名资深面试官。刚才你问了候选人一个问题，候选人给出了回答。
你需要：

1. **判断答案是否足够深入**（STAR 结构完整？有量化？关键技术/产品点讲清了？）
2. **如果不够深入**：生成一句追问，指向他讲得模糊的那一点（例："你说优化了 30%，具体是怎么定位瓶颈的？"）
3. **如果已经足够**：needs_followup=false，followup_question=null
4. **给一句点评 + 列亮点 / 薄弱点**：用于用户课后复盘

人设：{persona_style}

严格输出 JSON，字段：needs_followup / followup_question / quick_feedback / strengths / weaknesses。
"""


def follow_up(
    question: str,
    user_answer: str,
    persona: str = "product",
    prior_turns: Optional[list[dict]] = None,
) -> FollowUpResult:
    """对用户答案给出追问或结束，并反馈亮点/薄弱点。

    prior_turns 可选：之前 Q/A 历史，格式 [{"q": "...", "a": "..."}, ...]，
    用于让 LLM 判断追问是否已经足够（避免无限追问）。
    """
    if persona not in PERSONA_STYLES:
        persona = "product"
    p = PERSONA_STYLES[persona]

    schema = FollowUpResult.model_json_schema()

    history_block = ""
    if prior_turns:
        lines = []
        for i, t in enumerate(prior_turns, 1):
            lines.append(f"第{i}轮 Q: {t['q']}\n第{i}轮 A: {t['a']}")
        history_block = "\n【本题此前的追问历史（请避免重复追问同一点；若已追问 2 轮以上则倾向结束）】\n" + "\n".join(lines) + "\n"

    user_msg = (
        f"【当前问题】\n{question}\n\n"
        f"【候选人回答】\n{user_answer}\n"
        f"{history_block}\n"
        f"【输出 Schema】\n{json.dumps(schema, ensure_ascii=False)}"
    )

    response = _client.chat.completions.create(
        model=_MODEL,
        messages=[
            {"role": "system", "content": FOLLOWUP_PROMPT.format(persona_style=p["style"])},
            {"role": "user", "content": user_msg},
        ],
        response_format={"type": "json_object"},
        temperature=0.3,
    )
    return FollowUpResult.model_validate_json(response.choices[0].message.content)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    base = Path("data/outputs")
    resume = Resume.model_validate_json((base / "my_resume.json").read_text(encoding="utf-8"))
    jd_name = sys.argv[1] if len(sys.argv) > 1 else "jd_high"
    persona = sys.argv[2] if len(sys.argv) > 2 else "product"

    jd = JobDescription.model_validate_json((base / f"{jd_name}.json").read_text(encoding="utf-8"))
    report = MatchReport.model_validate_json((base / f"match_my_resume__{jd_name}.json").read_text(encoding="utf-8"))

    qs = generate_questions(resume, jd, report, persona=persona)
    out = base / f"interview_{jd_name}_{persona}.json"
    out.write_text(qs.model_dump_json(indent=2), encoding="utf-8")
    print(f"✅ 已生成：{out}")
    print(f"\n人设：{qs.persona}  ({qs.persona_style_note})")
    for cat, label in [("resume_deepdive", "简历深挖"), ("gap_probe", "差距探测"), ("domain_open", "岗位专业")]:
        print(f"\n===== {label} =====")
        for q in getattr(qs, cat):
            print(f"[{q.difficulty}] {q.question}")
