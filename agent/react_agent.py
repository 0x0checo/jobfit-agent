"""JobFit 求职陪跑 Agent —— ReAct 风格的动态 Agent 层。

设计理念:
1. **把现有 workflow 节点降级为工具**:parse_resume / parse_jd / match / rewrite /
   generate_interview_questions 全部包装成 OpenAI function calling tools。
2. **LLM 自主规划**:用户只需说一句自然语言目标(如"帮我准备字节 AIPM 面试"),
   agent 自己判断调用顺序、次数、参数。
3. **手写 agent loop(不用 LangGraph/LangChain)**:更清晰、可控、可调试。
4. **共享 state**:agent 的执行结果写回 st.session_state,和固定 tab 共用
   resume / jd / match_report 等状态 —— 在 agent 里做的事,在其他 tab 能看到。

与 graph.py 的 workflow 的关系:
- graph.py = 固定路径的 workflow(开发者决定流程)
- react_agent.py = 动态规划的 agent(LLM 决定流程)
- 二者共享底层 tools,体现 "Workflow-dominant + Agent-in-the-loop" 混合架构。
"""
import json
import os
from typing import Any, Iterator, Optional

from dotenv import load_dotenv
from openai import OpenAI

from agent.resume_parser import parse_resume as _parse_resume
from agent.jd_parser import parse_jd as _parse_jd
from agent.matcher import match as _match
from agent.rewriter import rewrite as _rewrite
from agent.interviewer import generate_questions as _gen_questions

load_dotenv()
_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


# ============================================================
# Tools Schema —— 遵循 OpenAI function calling 规范
# ============================================================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "parse_resume",
            "description": "把用户上传的简历 PDF 解析为结构化数据(姓名、教育、经历、技能等)。调用前请确认用户已上传 PDF(context 中有 pdf_path)。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "parse_jd",
            "description": "解析 JD(岗位描述)为结构化数据。用户可能提供 JD 文本或 URL,调用时只传一个。",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "JD 的完整文本(与 url 二选一)"},
                    "url": {"type": "string", "description": "JD 所在网页 URL(与 text 二选一)"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "match_resume_jd",
            "description": "生成简历 × JD 的匹配报告(综合分、硬/软打分、逐条差距、优先改进动作)。必须在 parse_resume 和 parse_jd 都完成后才能调用。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rewrite_resume",
            "description": "基于匹配报告改写简历 bullet,植入 JD 关键词。必须在 match_resume_jd 完成后才能调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "use_rag": {
                        "type": "boolean",
                        "description": "是否启用 75 条 bullet 案例库的 RAG 增强,默认 true",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_interview_questions",
            "description": "生成针对该简历 × JD 的 9 道定制面试题(3 类 × 3 难度)。必须在 match_resume_jd 完成后才能调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "persona": {
                        "type": "string",
                        "enum": ["tech", "product", "hr"],
                        "description": "面试官人设:tech=技术面, product=产品面(字节 AIPM 风格), hr=HR 终面",
                    }
                },
                "required": ["persona"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_progress",
            "description": "查看当前 session 已完成了哪些步骤(简历/JD/报告/改写/面试是否已生成)。规划下一步前推荐先调用,避免重复劳动。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


# ============================================================
# Tool 执行器 —— 读写 context(通常桥接 st.session_state)
# ============================================================


def _stringify(obj: Any, limit: int = 1500) -> str:
    """把 Pydantic 对象压成 JSON 字符串,控制长度避免上下文爆炸。"""
    if hasattr(obj, "model_dump"):
        s = json.dumps(obj.model_dump(), ensure_ascii=False)
    elif isinstance(obj, (dict, list)):
        s = json.dumps(obj, ensure_ascii=False)
    else:
        s = str(obj)
    if len(s) > limit:
        s = s[:limit] + "...(已截断)"
    return s


def execute_tool(name: str, args: dict, ctx: dict) -> str:
    """执行单个工具,返回给 LLM 看的字符串结果。

    ctx 是共享的上下文字典,tool 会读写里面的:
      pdf_path / jd_text / jd_url / resume / jd / match_report / rewrite_result / interview_set
    """
    try:
        if name == "parse_resume":
            pdf = ctx.get("pdf_path")
            if not pdf:
                return "❌ 失败:用户还未上传 PDF 简历。请明确告诉用户需要先在对话框上方上传简历 PDF。"
            ctx["resume"] = _parse_resume(pdf)
            return f"✅ 简历解析完成。姓名={ctx['resume'].name},教育={len(ctx['resume'].education)} 条,经历={len(ctx['resume'].experience)} 段。"

        if name == "parse_jd":
            text = args.get("text") or ctx.get("jd_text")
            url = args.get("url") or ctx.get("jd_url")
            if not text and not url:
                return "❌ 失败:用户未提供 JD 文本或 URL。请明确要求用户粘贴 JD 或给出链接。"
            ctx["jd"] = _parse_jd(text=text, url=url)
            return f"✅ JD 解析完成。岗位={ctx['jd'].title},硬性要求 {len(ctx['jd'].hard_requirements)} 条,关键词 {len(ctx['jd'].keywords)} 个。"

        if name == "match_resume_jd":
            if not ctx.get("resume") or not ctx.get("jd"):
                return "❌ 失败:必须先调用 parse_resume 和 parse_jd。"
            ctx["match_report"] = _match(ctx["resume"], ctx["jd"], prompt_version="v1_baseline")
            r = ctx["match_report"]
            return f"✅ 匹配报告完成。综合分={r.overall_score}/100, 硬性={r.hard_score}, 软性={r.soft_score}。一句话:{r.summary}。差距 {len(r.gaps)} 条,优先动作 {len(r.top_actions)} 条。"

        if name == "rewrite_resume":
            if not ctx.get("match_report"):
                return "❌ 失败:必须先调用 match_resume_jd。"
            use_rag = args.get("use_rag", True)
            ctx["rewrite_result"] = _rewrite(ctx["resume"], ctx["jd"], ctx["match_report"], use_rag=use_rag)
            r = ctx["rewrite_result"]
            rag_note = f",参考案例 {len(r.rag_references)} 条" if r.rag_references else ""
            return f"✅ 简历改写完成。改写 bullet {len(r.rewritten_bullets)} 条,植入关键词 {len(r.new_keywords_added)} 个{rag_note}。"

        if name == "generate_interview_questions":
            if not ctx.get("match_report"):
                return "❌ 失败:必须先调用 match_resume_jd。"
            persona = args.get("persona", "product")
            ctx["interview_set"] = _gen_questions(ctx["resume"], ctx["jd"], ctx["match_report"], persona=persona)
            qs = ctx["interview_set"]
            total = len(qs.resume_deepdive) + len(qs.gap_probe) + len(qs.domain_open)
            return f"✅ 面试题生成完成(人设={persona})。共 {total} 道题:简历深挖 {len(qs.resume_deepdive)} 道 / 差距探测 {len(qs.gap_probe)} 道 / 岗位专业 {len(qs.domain_open)} 道。"

        if name == "check_progress":
            done = []
            if ctx.get("resume"): done.append(f"✓ 简历已解析({ctx['resume'].name})")
            if ctx.get("jd"): done.append(f"✓ JD 已解析({ctx['jd'].title})")
            if ctx.get("match_report"): done.append(f"✓ 匹配报告已生成(综合分 {ctx['match_report'].overall_score})")
            if ctx.get("rewrite_result"): done.append(f"✓ 简历已改写({len(ctx['rewrite_result'].rewritten_bullets)} 条 bullet)")
            if ctx.get("interview_set"): done.append(f"✓ 面试题已生成({ctx['interview_set'].persona} 人设)")
            pending = []
            if not ctx.get("pdf_path") and not ctx.get("resume"):
                pending.append("✗ 用户还未上传简历 PDF")
            if not ctx.get("jd_text") and not ctx.get("jd_url") and not ctx.get("jd"):
                pending.append("✗ 用户还未提供 JD")
            return "【进度】\n" + ("\n".join(done) if done else "(尚未完成任何步骤)") + (("\n【待补】\n" + "\n".join(pending)) if pending else "")

        return f"❌ 未知工具:{name}"

    except Exception as e:
        return f"❌ 工具 {name} 执行出错:{type(e).__name__}: {str(e)[:200]}"


# ============================================================
# Agent Loop —— ReAct 风格(Reason + Act)
# ============================================================

SYSTEM_PROMPT = """你是 JobFit 求职陪跑 Agent,帮用户完成求职准备的全流程(解析简历 / 分析 JD / 匹配诊断 / 简历改写 / 模拟面试)。

【工具依赖关系】
- match_resume_jd 依赖 parse_resume + parse_jd
- rewrite_resume 和 generate_interview_questions 依赖 match_resume_jd
- 不确定当前进度时,先调用 check_progress

【行为准则】
1. **主动规划**:理解用户的最终目标后,一次性规划好所有要做的步骤,不要等用户逐步下指令。
2. **先查进度**:如果不确定哪些已完成,先调用 check_progress,避免重复劳动。
3. **每步前简短说明**:在调用工具前,用 1-2 句话告诉用户你接下来要做什么,口吻像真人助手。
4. **缺什么要什么**:如果必要输入缺失(没上传 PDF / 没给 JD),明确告诉用户需要什么,然后停下等待。
5. **最后总结**:所有工具调用完成后,用一段自然语言总结发生了什么、用户可以去哪个 tab 看详情。
6. **不废话**:不要在最终总结里重复工具返回的字段,用用户语言说"你的匹配分是 X,重点差距是 Y,我已经帮你改写了 Z 条"。

【人设选择建议】
如果用户没指定面试人设,默认选 product(对应字节 AIPM 岗最匹配);除非用户明确说"技术面"或"HR 面"。
"""


def run_agent(
    user_message: str,
    ctx: dict,
    history: Optional[list] = None,
    max_iters: int = 8,
) -> Iterator[tuple]:
    """运行 agent,流式 yield 执行事件。

    yield 的 (event_type, payload) 组合:
      ("thought", str)          —— LLM 的自然语言思考/说明
      ("tool_call", dict)       —— 要调用的工具 {name, args}
      ("tool_result", dict)     —— 工具执行结果 {name, result}
      ("final", str)            —— 最终回答
      ("error", str)            —— 出错

    history: 之前几轮对话的 messages(不含 system),格式 [{"role", "content"}, ...]
    """
    # 注入当前上下文状态,让 agent 知道已有什么可用输入(否则它会傻傻要求用户重传)
    ctx_hints = []
    if ctx.get("pdf_path"): ctx_hints.append(f"用户已上传简历 PDF(路径已就绪,你可以直接调用 parse_resume)")
    if ctx.get("jd_text"): ctx_hints.append(f"用户已提供 JD 文本(已就绪,可直接调用 parse_jd,不要再要求)")
    if ctx.get("jd_url"): ctx_hints.append(f"用户已提供 JD URL(已就绪)")
    if ctx.get("resume"): ctx_hints.append(f"简历已解析({ctx['resume'].name})")
    if ctx.get("jd"): ctx_hints.append(f"JD 已解析({ctx['jd'].title})")
    if ctx.get("match_report"): ctx_hints.append(f"匹配报告已生成(综合分 {ctx['match_report'].overall_score})")
    context_preamble = ""
    if ctx_hints:
        context_preamble = "【当前上下文状态】\n" + "\n".join(f"- {h}" for h in ctx_hints) + "\n\n"

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": context_preamble + user_message})

    for _ in range(max_iters):
        response = _client.chat.completions.create(
            model=_MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.3,
        )
        msg = response.choices[0].message

        # 如果 LLM 在 content 里说了话(思考/说明),先展示
        if msg.content and msg.content.strip():
            if msg.tool_calls:
                yield ("thought", msg.content)
            # 如果没有 tool_calls,content 就是最终答复

        # 没有工具调用 —— agent 认为任务已完成
        if not msg.tool_calls:
            yield ("final", msg.content or "(完成)")
            return

        # 记录 assistant 的 tool_calls 消息
        messages.append(
            {
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in msg.tool_calls
                ],
            }
        )

        # 依次执行每个工具
        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            yield ("tool_call", {"name": name, "args": args})
            result = execute_tool(name, args, ctx)
            yield ("tool_result", {"name": name, "result": result})
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    yield ("error", f"达到最大迭代次数 {max_iters},任务未完成。")


if __name__ == "__main__":
    import sys
    from pathlib import Path

    # CLI 测试:python -m agent.react_agent <pdf_path> <jd_text_file>
    pdf = sys.argv[1] if len(sys.argv) > 1 else None
    jd_file = sys.argv[2] if len(sys.argv) > 2 else None

    ctx = {}
    if pdf:
        ctx["pdf_path"] = pdf
    if jd_file:
        ctx["jd_text"] = Path(jd_file).read_text(encoding="utf-8")

    user_msg = "帮我完整准备一下这个岗位的求职:解析简历、分析 JD、生成匹配报告、改写简历、出一套产品面面试题。"

    print(f"\n🧑 用户:{user_msg}\n")
    for event, payload in run_agent(user_msg, ctx):
        if event == "thought":
            print(f"💭 {payload}")
        elif event == "tool_call":
            print(f"🔧 调用 {payload['name']}({payload['args']})")
        elif event == "tool_result":
            print(f"   ↳ {payload['result'][:200]}")
        elif event == "final":
            print(f"\n🤖 {payload}")
        elif event == "error":
            print(f"\n❌ {payload}")
