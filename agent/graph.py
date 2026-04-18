"""LangGraph 多 Agent 编排：把四个 agent 串成有向状态图。

Graph 结构：
    START
      │
      ├─► parse_resume_node  (PDF → Resume)
      │
      ├─► parse_jd_node      (text/url → JobDescription)
      │
      ├─► match_node         (Resume + JD → MatchReport)
      │
      └─► rewrite_node       (Resume + JD + Report → RewriteResult)
      │
      END

设计要点（面试讲解）：
1. State 是全局共享字典，每个 node 读需要的字段、写自己产出的字段 —— 解耦 agent 间通信。
2. 节点幂等：跳过已有结果（例如用户改了 JD 但简历未变，parse_resume 不重跑），节省 token。
3. 可扩展：后续加 RewriteCriticAgent 做改写质检，或并行 parse_resume / parse_jd 只需加一条边。
4. 为什么选 LangGraph：对应字节"智能体开发平台"——图状结构便于可视化 / 工具化 / 二次开发。
"""
from typing import Optional, TypedDict

from langgraph.graph import StateGraph, END

from agent.resume_parser import parse_resume
from agent.jd_parser import parse_jd
from agent.matcher import match
from agent.rewriter import rewrite
from agent.schemas import Resume, JobDescription, MatchReport, RewriteResult


class PipelineState(TypedDict, total=False):
    # 输入
    resume_pdf_path: Optional[str]
    jd_text: Optional[str]
    jd_url: Optional[str]
    prompt_version: str
    use_rag: bool

    # 中间产物
    resume: Optional[Resume]
    jd: Optional[JobDescription]
    match_report: Optional[MatchReport]
    rewrite_result: Optional[RewriteResult]

    # 执行痕迹
    log: list


def _log(state: PipelineState, msg: str) -> PipelineState:
    state.setdefault("log", []).append(msg)
    return state


# ---------- Nodes ----------

def parse_resume_node(state: PipelineState) -> PipelineState:
    if state.get("resume"):
        _log(state, "↻ parse_resume: 跳过（已存在）")
        return state
    _log(state, "▶ parse_resume: 运行中")
    state["resume"] = parse_resume(state["resume_pdf_path"])
    _log(state, f"✓ parse_resume: {state['resume'].name}")
    return state


def parse_jd_node(state: PipelineState) -> PipelineState:
    if state.get("jd"):
        _log(state, "↻ parse_jd: 跳过（已存在）")
        return state
    _log(state, "▶ parse_jd: 运行中")
    state["jd"] = parse_jd(text=state.get("jd_text"), url=state.get("jd_url"))
    _log(state, f"✓ parse_jd: {state['jd'].title}")
    return state


def match_node(state: PipelineState) -> PipelineState:
    _log(state, "▶ match: 运行中")
    state["match_report"] = match(
        state["resume"],
        state["jd"],
        prompt_version=state.get("prompt_version", "v2_fixed"),
    )
    _log(state, f"✓ match: overall_score={state['match_report'].overall_score}")
    return state


def rewrite_node(state: PipelineState) -> PipelineState:
    _log(state, "▶ rewrite: 运行中")
    state["rewrite_result"] = rewrite(
        state["resume"],
        state["jd"],
        state["match_report"],
        use_rag=state.get("use_rag", True),
    )
    n = len(state["rewrite_result"].rewritten_bullets)
    _log(state, f"✓ rewrite: 改写 {n} 条 bullet")
    return state


# ---------- Graph ----------

def build_graph():
    g = StateGraph(PipelineState)
    g.add_node("parse_resume", parse_resume_node)
    g.add_node("parse_jd", parse_jd_node)
    g.add_node("match", match_node)
    g.add_node("rewrite", rewrite_node)

    g.set_entry_point("parse_resume")
    g.add_edge("parse_resume", "parse_jd")
    g.add_edge("parse_jd", "match")
    g.add_edge("match", "rewrite")
    g.add_edge("rewrite", END)

    return g.compile()


GRAPH = build_graph()


NODE_FRIENDLY = {
    "parse_resume": "正在解析你的简历...",
    "parse_jd": "正在研读目标岗位...",
    "match": "正在逐条对齐你的经历与岗位要求...",
    "rewrite": "正在为你重构简历表述...",
}


def run_pipeline(**inputs) -> PipelineState:
    """一键全流程入口（一次性返回最终 state）。"""
    initial: PipelineState = {"prompt_version": "v1_baseline", "log": [], **inputs}
    return GRAPH.invoke(initial)


def stream_pipeline(**inputs):
    """流式执行：每个节点完成后 yield (node_name, friendly_msg, state)。
    供前端做进度条/状态实时展示使用。"""
    initial: PipelineState = {"prompt_version": "v1_baseline", "log": [], **inputs}
    last_state: PipelineState = initial
    for update in GRAPH.stream(initial, stream_mode="updates"):
        for node_name, node_state in update.items():
            last_state = {**last_state, **node_state}
            yield node_name, NODE_FRIENDLY.get(node_name, node_name), last_state


if __name__ == "__main__":
    import sys
    result = run_pipeline(
        resume_pdf_path=sys.argv[1],
        jd_text=open(sys.argv[2], encoding="utf-8").read(),
    )
    print("\n".join(result["log"]))
    print(f"\n综合分：{result['match_report'].overall_score}")
    print(f"改写 bullet 数：{len(result['rewrite_result'].rewritten_bullets)}")
