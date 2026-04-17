"""JD 结构化解析器：文本或 URL → JobDescription(JSON)。

设计决策：
- 双入口（text / url）：text 走直通，url 先用 Crawl4AI 抓 Markdown 再进 LLM。
- 硬/软要求分离：对应两个目标 JD 里都有"优先/加分"字段，下游匹配可分权重。
- keywords 抽取：专为后续简历关键词命中率分析准备。
"""
import json
import os
from dotenv import load_dotenv
from openai import OpenAI

from agent.schemas import JobDescription

load_dotenv()

_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

SYSTEM_PROMPT = """你是一个招聘 JD 结构化抽取专家。任务：把原始 JD 文本/Markdown 抽取成严格符合给定 JSON Schema 的结构化数据。

规则：
1. 忠实抽取，不编造；原文没有的字段留空字符串或空数组。
2. responsibilities 抽"岗位职责"/"工作内容"下的条目；hard_requirements 抽"任职要求"里非"优先/最好"的条目；soft_preferences 抽含"优先/加分/最好/熟悉更佳"的条目。
3. keywords：从职责和要求中抽取 5-15 个核心技术/能力关键词（如：大模型、Agent、RAG、Prompt 调优、SQL、数据分析、搜索推荐、用户增长等）。
4. 团队介绍/业务背景放 team_intro。
5. 只输出 JSON，不要任何解释。
"""


def parse_jd(text: str = None, url: str = None) -> JobDescription:
    if url:
        from utils.web_scraper import scrape_url  # 懒加载，避免云端无 crawl4ai 时 import 失败
        raw = scrape_url(url)
    elif text:
        raw = text
    else:
        raise ValueError("必须提供 text 或 url 参数之一")

    response = _client.chat.completions.create(
        model=_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"JD 原文：\n\n{raw}\n\nJSON Schema:\n{json.dumps(JobDescription.model_json_schema(), ensure_ascii=False, indent=2)}"},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    data = json.loads(response.choices[0].message.content)
    return JobDescription.model_validate(data)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    arg = sys.argv[1]
    if arg.startswith("http"):
        jd = parse_jd(url=arg)
        stem = "jd_from_url"
    else:
        text = Path(arg).read_text(encoding="utf-8")
        jd = parse_jd(text=text)
        stem = Path(arg).stem

    output_dir = Path("data/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{stem}.json"

    json_str = json.dumps(jd.model_dump(), ensure_ascii=False, indent=2)
    output_path.write_text(json_str, encoding="utf-8")

    print(json_str)
    print(f"\n✅ 已保存到：{output_path}")
