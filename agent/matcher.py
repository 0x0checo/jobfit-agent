"""匹配 Agent：简历 + JD → MatchReport。

设计要点（面试讲解）：
1. 分层评分：硬性/软性分开打分，再按 0.7/0.3 加权成 overall，避免"软性拉高总分"的假象。
2. 逐条需求对齐：每条 JD 要求都输出 match_level + evidence + suggestion，
   让建议有据可依，不是笼统的"多写项目经验"。
3. 关键词命中率：独立指标，对应 JD 里的 keywords 字段，
   可以直观反映简历里有没有出现招聘方在意的术语（Agent / RAG / SQL 等）。
4. Top actions：给出 3-5 条"性价比最高"的修改动作，
   这是 PM 视角——不是列出所有问题，而是指出最该先做什么。
"""
import json
import os
from dotenv import load_dotenv
from openai import OpenAI

from agent.schemas import Resume, JobDescription, MatchReport
from agent.prompts import MATCHER_PROMPTS

load_dotenv()

_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def match(resume: Resume, jd: JobDescription, prompt_version: str = "v1_baseline") -> MatchReport:
    user_msg = (
        f"候选人简历（JSON）：\n{json.dumps(resume.model_dump(), ensure_ascii=False, indent=2)}\n\n"
        f"目标 JD（JSON）：\n{json.dumps(jd.model_dump(), ensure_ascii=False, indent=2)}\n\n"
        f"输出 Schema：\n{json.dumps(MatchReport.model_json_schema(), ensure_ascii=False, indent=2)}"
    )
    response = _client.chat.completions.create(
        model=_MODEL,
        messages=[
            {"role": "system", "content": MATCHER_PROMPTS[prompt_version]},
            {"role": "user", "content": user_msg},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
    )
    data = json.loads(response.choices[0].message.content)
    return MatchReport.model_validate(data)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    resume_json_path = Path(sys.argv[1])
    jd_json_path = Path(sys.argv[2])

    resume = Resume.model_validate_json(resume_json_path.read_text(encoding="utf-8"))
    jd = JobDescription.model_validate_json(jd_json_path.read_text(encoding="utf-8"))
    report = match(resume, jd)

    output_dir = Path("data/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"match_{resume_json_path.stem}__{jd_json_path.stem}.json"

    json_str = json.dumps(report.model_dump(), ensure_ascii=False, indent=2)
    output_path.write_text(json_str, encoding="utf-8")

    print(json_str)
    print(f"\n✅ 已保存到：{output_path}")
