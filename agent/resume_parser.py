"""简历结构化解析器：PDF → Resume(JSON)。

方法论（可用于面试讲解）：
1. Schema 约束（Pydantic）—— 保证输出字段齐全、类型正确
2. 指令清晰化 —— system prompt 明确抽取规则
3. Structured Output —— 用 OpenAI response_format 强制返回合法 JSON
"""
import json
import os
from dotenv import load_dotenv
from openai import OpenAI

from agent.schemas import Resume
from utils.pdf_parser import extract_text_from_pdf

load_dotenv()

_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

SYSTEM_PROMPT = """你是一个简历结构化抽取专家。任务：把原始简历文本抽取成严格符合给定 JSON Schema 的结构化数据。

规则：
1. 忠实抽取，不编造内容；原文没有的字段留空字符串或空数组。
2. experience.bullets 每条保留原文关键信息（含量化数据），不要改写或缩略。
3. 日期统一为 "YYYY年M月" 格式；"至今" 保留原文。
4. skills 分类时：自然语言放 languages，编程语言/SQL 放 programming，Claude/GPT/LLM API 类放 ai_tools，Axure/Visio/Office 类放 product_tools。
5. 只输出 JSON，不要任何解释。
"""


def parse_resume(pdf_path: str) -> Resume:
    raw_text = extract_text_from_pdf(pdf_path)

    response = _client.chat.completions.create(
        model=_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"简历原文：\n\n{raw_text}\n\nJSON Schema:\n{json.dumps(Resume.model_json_schema(), ensure_ascii=False, indent=2)}"},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    data = json.loads(response.choices[0].message.content)
    return Resume.model_validate(data)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    pdf_path = Path(sys.argv[1])
    resume = parse_resume(str(pdf_path))

    output_dir = Path("data/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{pdf_path.stem}.json"

    json_str = json.dumps(resume.model_dump(), ensure_ascii=False, indent=2)
    output_path.write_text(json_str, encoding="utf-8")

    print(json_str)
    print(f"\n✅ 已保存到：{output_path}")
