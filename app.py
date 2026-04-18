"""JobFit Agent · Streamlit UI。

设计原则：
- 面向真实求职者的产品文案（而非开发/面试视角）
- 一键全流程 + 透明的节点级进度
- 默认隐藏开发者视角的 Prompt 评测台，?dev=1 才显示
"""
import json
import os
from pathlib import Path

import streamlit as st

# ---------- Secrets / env 兼容 ----------
try:
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
        os.environ["OPENAI_MODEL"] = st.secrets.get("OPENAI_MODEL", "gpt-4o-mini")
except (FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
    pass

IS_CLOUD = "STREAMLIT_SERVER_PORT" in os.environ

from agent.resume_parser import parse_resume
from agent.jd_parser import parse_jd
from agent.matcher import match
from agent.rewriter import rewrite, render_markdown
from agent.interviewer import generate_questions, follow_up, PERSONA_STYLES
from agent.graph import stream_pipeline, NODE_FRIENDLY
from agent.pdf_export import markdown_to_pdf
from agent.schemas import Resume, JobDescription, MatchReport
from agent.display_labels import label

# ---------- 页面配置 ----------
st.set_page_config(
    page_title="JobFit · 你的 AI 求职助手",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# 是否显示开发者/评测台 tab（?dev=1）
DEV_MODE = st.query_params.get("dev") == "1"

# ---------- 自定义样式 ----------
st.markdown("""
<style>
/* 主容器留白与字体微调 */
.block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1180px; }

/* Hero 区 */
.hero-title {
    font-size: 2.4rem;
    font-weight: 700;
    color: #111827;
    letter-spacing: -0.5px;
    margin-bottom: 0.3rem;
}
.hero-subtitle {
    font-size: 1.05rem;
    color: #6B7280;
    margin-bottom: 2rem;
    font-weight: 400;
}

/* Tab 样式加强 */
.stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid #E5E7EB; }
.stTabs [data-baseweb="tab"] {
    height: 44px;
    padding: 0 18px;
    background: transparent;
    border-radius: 8px 8px 0 0;
    color: #6B7280;
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background: #EEF2FF;
    color: #6366F1;
}

/* Metric 卡片 */
[data-testid="stMetric"] {
    background: #F9FAFB;
    border: 1px solid #E5E7EB;
    padding: 14px 16px;
    border-radius: 10px;
}
[data-testid="stMetricLabel"] { color: #6B7280; font-size: 0.85rem; }
[data-testid="stMetricValue"] { color: #111827; font-weight: 700; }

/* 按钮 */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%);
    border: none;
    font-weight: 600;
    padding: 0.55rem 1.4rem;
    border-radius: 8px;
    box-shadow: 0 2px 8px rgba(99, 102, 241, 0.25);
}
.stButton > button[kind="primary"]:hover { filter: brightness(1.08); }

/* Expander */
div[data-testid="stExpander"] {
    border: 1px solid #E5E7EB;
    border-radius: 10px;
    background: #FFFFFF;
}

/* 下载按钮 */
.stDownloadButton > button {
    background: #FFFFFF;
    border: 1px solid #D1D5DB;
    color: #374151;
    font-weight: 500;
    border-radius: 8px;
}
.stDownloadButton > button:hover {
    border-color: #6366F1;
    color: #6366F1;
}

/* 隐藏 Streamlit 默认页脚/菜单 */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ---------- Session state ----------
for k in ["resume", "jd", "match_report", "rewrite_result", "interview_set", "interview_chat"]:
    if k not in st.session_state:
        st.session_state[k] = None
# interview_chat 是 dict: {question_idx: [{"role": "interviewer"/"candidate"/"feedback", "content": "...", ...}, ...]}
if st.session_state.interview_chat is None:
    st.session_state.interview_chat = {}

DATA_DIR = Path("data")
UPLOAD_DIR = DATA_DIR / "resumes"
OUTPUT_DIR = DATA_DIR / "outputs"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Hero ----------
st.markdown('<div class="hero-title">🎯 JobFit · AI 求职助手</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">上传简历、贴入心仪岗位，AI 为你逐条对齐要求，并生成针对该岗位的定制化简历。</div>',
    unsafe_allow_html=True,
)

# ---------- Tabs ----------
tab_names = [
    "🚀 一键生成",
    "📄 我的简历",
    "💼 目标岗位",
    "📊 匹配报告",
    "✍️ 简历改写",
    "🎤 模拟面试",
]
if DEV_MODE:
    tab_names.append("🔬 评测台（开发者）")

tabs = st.tabs(tab_names)


# ==================== Tab 0: 一键生成 ====================
with tabs[0]:
    st.markdown("#### 一步完成：解析简历 · 理解岗位 · 生成匹配报告 · 针对性改写")
    st.caption("整个流程约 20-40 秒，全程透明，你会看到每一步的进展。")

    c1, c2 = st.columns(2)
    with c1:
        pl_pdf = st.file_uploader("上传你的简历（PDF）", type=["pdf"], key="pl_pdf")
    with c2:
        pl_jd = st.text_area("粘贴目标岗位 JD", height=200, key="pl_jd",
                             placeholder="把招聘页的岗位描述复制过来即可...")

    go = st.button("🚀 开始生成", type="primary", disabled=not (pl_pdf and pl_jd.strip()))

    if go:
        pdf_path = UPLOAD_DIR / pl_pdf.name
        pdf_path.write_bytes(pl_pdf.getbuffer())

        with st.status("AI 正在工作中...", expanded=True) as status:
            step_box = st.empty()
            final_state = None
            try:
                for node_name, friendly_msg, state in stream_pipeline(
                    resume_pdf_path=str(pdf_path), jd_text=pl_jd
                ):
                    step_box.markdown(f"**✓** {friendly_msg}")
                    final_state = state
                status.update(label="✅ 全部完成！", state="complete", expanded=False)
            except Exception as e:
                status.update(label=f"❌ 出错：{e}", state="error")
                raise

        if final_state:
            st.session_state.resume = final_state["resume"]
            st.session_state.jd = final_state["jd"]
            st.session_state.match_report = final_state["match_report"]
            st.session_state.rewrite_result = final_state["rewrite_result"]

            rep = final_state["match_report"]
            res = final_state["rewrite_result"]

            st.success("已为你生成完整分析，可以切换到上方标签页查看详情。")

            c1, c2, c3 = st.columns(3)
            c1.metric("综合匹配度", f"{rep.overall_score} / 100")
            c2.metric("简历改写条目", len(res.rewritten_bullets))
            c3.metric("新增岗位关键词", len(res.new_keywords_added))

            st.info(f"**一句话总结**：{rep.summary}")


# ==================== Tab 1: 我的简历 ====================
with tabs[1]:
    st.markdown("#### 上传并解析你的简历")
    st.caption("AI 会把 PDF 里的信息抽取为结构化数据，方便后续与岗位逐条对齐。")

    uploaded = st.file_uploader("选择 PDF 文件", type=["pdf"], key="tab_resume_upload")
    if uploaded and st.button("解析简历", type="primary", key="btn_parse_resume"):
        pdf_path = UPLOAD_DIR / uploaded.name
        pdf_path.write_bytes(uploaded.getbuffer())
        with st.spinner("AI 正在解析..."):
            st.session_state.resume = parse_resume(str(pdf_path))
        st.success("解析完成")

    if st.session_state.resume:
        resume = st.session_state.resume
        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown(f"### {resume.name}")
            if resume.email: st.caption(f"📧 {resume.email}")
            if resume.phone: st.caption(f"📱 {resume.phone}")
            if resume.website: st.caption(f"🔗 {resume.website}")
            if resume.summary:
                st.markdown("**关于我**")
                st.info(resume.summary)

        with c2:
            st.markdown("**教育经历**")
            for edu in resume.education:
                st.markdown(f"- **{edu.school}** · {edu.major or ''} {edu.degree or ''}  ({edu.start_date} - {edu.end_date})")

            st.markdown("**实践经历**")
            for exp in resume.experience:
                with st.expander(f"{exp.company} · {exp.role} ({exp.start_date} - {exp.end_date})"):
                    for b in exp.bullets:
                        st.markdown(f"- {b}")

            st.markdown("**技能**")
            sk = resume.skills
            if sk.languages:
                st.markdown(f"**语言**：{' · '.join(sk.languages)}")
            if sk.programming:
                st.markdown(f"**编程**：{' · '.join(sk.programming)}")
            if sk.ai_tools:
                st.markdown(f"**AI 工具**：{' · '.join(sk.ai_tools)}")
            if sk.product_tools:
                st.markdown(f"**产品工具**：{' · '.join(sk.product_tools)}")


# ==================== Tab 2: 目标岗位 ====================
with tabs[2]:
    st.markdown("#### 告诉 AI 你想申请什么岗位")
    st.caption("你可以直接粘贴 JD 文本；本地运行时也支持从招聘页 URL 自动抓取。")

    modes = ["粘贴文本"] if IS_CLOUD else ["粘贴文本", "抓取 URL"]
    if IS_CLOUD:
        st.caption("ℹ️ 线上版本仅支持文本输入，如需 URL 抓取请本地运行。")
    mode = st.radio("输入方式", modes, horizontal=True, label_visibility="collapsed")

    if mode == "粘贴文本":
        jd_text = st.text_area("粘贴 JD 全文", height=280, placeholder="把招聘页的岗位描述复制过来...")
        if st.button("分析岗位", type="primary", key="btn_parse_jd_text"):
            if jd_text.strip():
                with st.spinner("AI 正在解读岗位要求..."):
                    st.session_state.jd = parse_jd(text=jd_text)
                st.success("分析完成")
    else:
        url = st.text_input("岗位页面 URL")
        if st.button("抓取并分析", type="primary", key="btn_parse_jd_url"):
            if url.strip():
                with st.spinner("正在抓取页面并分析（约 5-10 秒）..."):
                    st.session_state.jd = parse_jd(url=url)
                st.success("分析完成")

    if st.session_state.jd:
        jd = st.session_state.jd
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"### {jd.title}")
            if jd.company or jd.location:
                st.caption(f"{jd.company or ''} · {jd.location or ''}")
            if jd.team_intro:
                with st.expander("团队介绍"):
                    st.write(jd.team_intro)
            st.markdown("**核心关键词**")
            st.write(" · ".join([f"`{k}`" for k in jd.keywords]) or "-")
        with c2:
            st.markdown("**岗位职责**")
            for r in jd.responsibilities:
                st.markdown(f"- {r}")
            st.markdown("**任职要求**")
            for r in jd.hard_requirements:
                st.markdown(f"- {r}")
            if jd.soft_preferences:
                st.markdown("**加分项**")
                for r in jd.soft_preferences:
                    st.markdown(f"- {r}")


# ==================== Tab 3: 匹配报告 ====================
with tabs[3]:
    if not st.session_state.resume or not st.session_state.jd:
        st.warning("请先在「我的简历」和「目标岗位」完成解析。")
    else:
        col_a, col_b = st.columns([3, 1])
        with col_a:
            st.markdown("#### 简历与岗位的匹配分析")
            st.caption("AI 逐条对齐岗位要求，指出优势、差距和最值得优先改进的方向。")
        with col_b:
            if DEV_MODE:
                prompt_ver = st.selectbox(
                    "Prompt 版本（开发者）",
                    ["v1_baseline", "v2_fixed"],
                    index=0,
                    label_visibility="collapsed",
                )
            else:
                prompt_ver = "v1_baseline"

        if st.button("生成匹配报告", type="primary", key="btn_match"):
            with st.spinner("AI 正在逐条对齐..."):
                st.session_state.match_report = match(
                    st.session_state.resume, st.session_state.jd, prompt_version=prompt_ver
                )
            st.success("报告生成完成")

        if st.session_state.match_report:
            rep = st.session_state.match_report
            c1, c2, c3 = st.columns(3)
            c1.metric("综合匹配度", f"{rep.overall_score}")
            c2.metric("核心要求", f"{rep.hard_score}")
            c3.metric("加分项", f"{rep.soft_score if rep.soft_score is not None else 'N/A'}")

            st.info(f"**一句话总结**：{rep.summary}")

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("##### ✨ 你的核心优势")
                for s in rep.strengths:
                    st.markdown(f"- {s}")
            with c2:
                st.markdown("##### 🎯 优先改进方向")
                for i, a in enumerate(rep.top_actions, 1):
                    st.markdown(f"**{i}.** {a}")

            st.markdown("##### 🔑 关键词覆盖情况")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**已覆盖**")
                st.write(" · ".join([f"`{k}`" for k in rep.keyword_hits]) or "-")
            with c2:
                st.markdown("**尚未覆盖**")
                st.write(" · ".join([f"`{k}`" for k in rep.keyword_misses]) or "-")

            st.markdown("##### 📋 逐条差距分析")
            for g in rep.gaps:
                color = {"full": "🟢", "partial": "🟡", "missing": "🔴"}.get(g.match_level, "⚪")
                tag = "核心要求" if g.req_type == "hard" else "加分项"
                with st.expander(f"{color} [{tag}] {g.requirement}"):
                    if g.evidence:
                        st.markdown(f"**简历中的相关证据**：{g.evidence}")
                    st.markdown(f"**改进建议**：{g.suggestion}")


# ==================== Tab 4: 简历改写 ====================
with tabs[4]:
    if not st.session_state.match_report:
        st.warning("请先在「匹配报告」生成分析结果。")
    else:
        st.markdown("#### 为你生成这份岗位的定制简历")
        st.caption("AI 会保留所有量化数据，只针对岗位要求优化表述、植入招聘方关注的关键词。")

        use_rag = st.toggle(
            "✨ 启用案例库增强（RAG）",
            value=True,
            help="开启后会从 75 条高质量简历 bullet 案例库中检索 top-3 相似案例，作为改写的风格锚点 —— 帮助 AI 写出更像真实优秀 bullet 的表述。",
            key="rewrite_use_rag",
        )

        if st.button("生成改写建议", type="primary", key="btn_rewrite"):
            with st.spinner("AI 正在为你定制..."):
                st.session_state.rewrite_result = rewrite(
                    st.session_state.resume,
                    st.session_state.jd,
                    st.session_state.match_report,
                    use_rag=use_rag,
                )
            st.success("改写完成")

        if st.session_state.rewrite_result:
            res = st.session_state.rewrite_result

            if res.new_keywords_added:
                st.markdown("**🔑 本次植入关键词**：" + " · ".join([f"`{k}`" for k in res.new_keywords_added]))

            if res.rewritten_summary:
                st.markdown("##### 关于我（个人定位）")
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**原版**")
                    st.info(st.session_state.resume.summary or "-")
                with c2:
                    st.markdown("**改写后**")
                    st.success(res.rewritten_summary)
                if res.summary_change_reason:
                    st.caption(f"💡 {res.summary_change_reason}")

            st.markdown("##### 经历条目改写")
            if not res.rewritten_bullets:
                st.caption("未发现需要改写的条目 —— 简历当前表述已与岗位高度匹配。")
            for rb in res.rewritten_bullets:
                with st.expander(f"📝 {rb.experience_company}"):
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("**原文**")
                        st.info(rb.before)
                    with c2:
                        st.markdown("**改写后**")
                        st.success(rb.after)
                    st.caption(f"💡 {rb.change_reason}")

            if res.notes:
                with st.expander("改写策略说明"):
                    st.write(res.notes)

            if res.rag_references:
                with st.expander(f"🔍 本次参考了案例库中的 {len(res.rag_references)} 条 bullet"):
                    st.caption("以下高质量案例作为改写的风格 / 量化粒度锚点（不会直接抄袭内容）")
                    for i, ref in enumerate(res.rag_references, 1):
                        tags = " · ".join(ref.get("skill_tags", []))
                        st.markdown(
                            f"**[{i}] `{ref['role_tag']}` · {tags}**  "
                            f"<span style='color:#6b7280;font-size:0.85em'>相似度 {ref.get('score', 0):.2f}</span>",
                            unsafe_allow_html=True,
                        )
                        st.markdown(f"> {ref['bullet']}")

            # 导出
            md_text = render_markdown(st.session_state.resume, res)
            st.markdown("##### 📥 导出")
            c1, c2 = st.columns(2)
            with c1:
                st.download_button(
                    "下载 Markdown",
                    data=md_text,
                    file_name=f"{st.session_state.resume.name}_改写版.md",
                    mime="text/markdown",
                    use_container_width=True,
                )
            with c2:
                try:
                    pdf_bytes = markdown_to_pdf(md_text)
                    st.download_button(
                        "下载 PDF",
                        data=pdf_bytes,
                        file_name=f"{st.session_state.resume.name}_改写版.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )
                except Exception as e:
                    st.caption(f"⚠️ PDF 暂不可用：{e}")


# ==================== Tab 5: 模拟面试 ====================
with tabs[5]:
    if not st.session_state.resume or not st.session_state.jd or not st.session_state.match_report:
        st.warning("请先完成简历解析、岗位解析和匹配报告，面试题基于三者生成。")
    else:
        st.markdown("#### 用匹配报告当弹药，让 AI 扮演面试官拷问你")
        st.caption("面试题针对这份简历 × 这个岗位定制 —— 会挖你的项目、戳你的差距、考你的岗位 sense。支持多轮追问。")

        c1, c2 = st.columns([2, 1])
        with c1:
            persona = st.radio(
                "选择面试官人设",
                options=list(PERSONA_STYLES.keys()),
                format_func=lambda x: PERSONA_STYLES[x]["label"],
                horizontal=True,
                key="interview_persona",
            )
            st.caption(f"💡 {PERSONA_STYLES[persona]['style']}")
        with c2:
            st.write("")
            st.write("")
            if st.button("🎬 开始面试（生成题目）", type="primary", key="btn_gen_interview"):
                with st.spinner("面试官正在备题..."):
                    st.session_state.interview_set = generate_questions(
                        st.session_state.resume,
                        st.session_state.jd,
                        st.session_state.match_report,
                        persona=persona,
                    )
                    st.session_state.interview_chat = {}  # 重置答题历史
                st.success("题目已生成，开始作答吧")

        qs = st.session_state.interview_set
        if qs:
            st.divider()
            cat_labels = {
                "resume_deepdive": ("🎯 简历深挖", "面试官盯着你的项目细节追问"),
                "gap_probe": ("⚠️ 差距探测", "面试官试图验证你简历里没写透的能力"),
                "domain_open": ("🧠 岗位专业", "面试官考察你对业务领域的 sense"),
            }
            diff_color = {"easy": "🟢", "medium": "🟡", "hard": "🔴"}

            for cat, (label_txt, label_hint) in cat_labels.items():
                questions = getattr(qs, cat)
                if not questions:
                    continue
                st.markdown(f"### {label_txt}")
                st.caption(label_hint)

                for qi, q in enumerate(questions):
                    qid = f"{cat}_{qi}"
                    with st.expander(f"{diff_color.get(q.difficulty, '⚪')} **Q:** {q.question}"):
                        st.caption(f"**考察意图**：{q.intent}")
                        if q.linked_requirement:
                            st.caption(f"**关联**：{q.linked_requirement}")

                        # 展示此前的对话
                        turns = st.session_state.interview_chat.get(qid, [])
                        for t in turns:
                            if t["role"] == "candidate":
                                st.markdown(f"**🗣️ 你的回答**")
                                st.info(t["content"])
                            elif t["role"] == "interviewer_followup":
                                st.markdown(f"**🎙️ 追问**")
                                st.warning(t["content"])
                            elif t["role"] == "feedback":
                                fb = t["content"]
                                st.markdown("**💡 快速反馈**")
                                st.caption(fb["quick_feedback"])
                                if fb.get("strengths"):
                                    st.markdown("**亮点**：" + " · ".join(fb["strengths"]))
                                if fb.get("weaknesses"):
                                    st.markdown("**可改进**：" + " · ".join(fb["weaknesses"]))

                        # 下一轮问题（首问或追问）
                        current_q = q.question
                        for t in reversed(turns):
                            if t["role"] == "interviewer_followup":
                                current_q = t["content"]
                                break

                        # 是否已结束（feedback.needs_followup == False）
                        ended = any(
                            t["role"] == "feedback" and not t["content"].get("needs_followup", True)
                            for t in turns
                        )

                        if not ended:
                            ans_key = f"ans_{qid}_{len(turns)}"
                            ans = st.text_area(
                                "✍️ 作答" if not any(t["role"] == "candidate" for t in turns) else "✍️ 继续作答",
                                key=ans_key,
                                height=120,
                                placeholder="推荐 STAR 结构：Situation 场景 / Task 任务 / Action 动作 / Result 结果+量化",
                            )
                            submit_key = f"submit_{qid}_{len(turns)}"
                            if st.button("提交答案", key=submit_key):
                                if ans.strip():
                                    prior = []
                                    pending_q = q.question
                                    for t in turns:
                                        if t["role"] == "candidate":
                                            prior.append({"q": pending_q, "a": t["content"]})
                                        elif t["role"] == "interviewer_followup":
                                            pending_q = t["content"]
                                    with st.spinner("面试官在思考..."):
                                        fb = follow_up(
                                            question=current_q,
                                            user_answer=ans,
                                            persona=qs.persona,
                                            prior_turns=prior or None,
                                        )
                                    turns.append({"role": "candidate", "content": ans})
                                    turns.append({"role": "feedback", "content": fb.model_dump()})
                                    if fb.needs_followup and fb.followup_question:
                                        turns.append({"role": "interviewer_followup", "content": fb.followup_question})
                                    st.session_state.interview_chat[qid] = turns
                                    st.rerun()
                        else:
                            st.success("✅ 本题已结束。")

                        # 参考答题要点
                        if q.answer_hints:
                            with st.expander("📖 查看参考答题要点（答完再看避免剧透）"):
                                for h in q.answer_hints:
                                    st.markdown(f"- {h}")


# ==================== Tab 6 (dev only): 评测台 ====================
if DEV_MODE:
    with tabs[6]:
        st.subheader("Prompt 版本对比评测（开发者视角）")
        st.caption("LLM-as-Judge + Rule-based Judge 双通道评估 matcher prompt 的稳定性与正确性。")

        bench_path = OUTPUT_DIR / "bench_report.json"
        if not bench_path.exists():
            st.info("尚未运行评测。请在终端运行：`python -m agent.eval_bench`")
        else:
            data = json.loads(bench_path.read_text(encoding="utf-8"))
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("测试用例", data["total_cases"])
            c2.metric("v2 胜场", data["v2_wins"])
            c3.metric("v1 胜场", data["v1_wins"])
            c4.metric("平局", data["ties"])

            c1, c2 = st.columns(2)
            c1.metric("v1 平均分", f"{data['v1_avg_total']} / 15")
            c2.metric("v2 平均分", f"{data['v2_avg_total']} / 15",
                      delta=round(data['v2_avg_total']-data['v1_avg_total'], 2))

            for c in data["cases"]:
                header = f"{c['case_name']} · 胜者：{c['winner']}"
                if c.get("judge_disagreement"):
                    header += " · ⚠️ 双判分歧"
                with st.expander(header):
                    if c.get("judge_disagreement"):
                        st.error(f"**分歧**：{c['judge_disagreement']}")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**v1 — LLM Judge**")
                        st.json(c["v1_eval"])
                        if c.get("v1_rule_eval"):
                            st.markdown(f"**v1 — Rule Judge** ({c['v1_rule_eval']['pass_rate']*100:.0f}%)")
                            for chk in c["v1_rule_eval"]["checks"]:
                                icon = "✅" if chk["passed"] else "❌"
                                st.markdown(f"{icon} `{chk['name']}` — {chk['detail']}")
                    with col2:
                        st.markdown("**v2 — LLM Judge**")
                        st.json(c["v2_eval"])
                        if c.get("v2_rule_eval"):
                            st.markdown(f"**v2 — Rule Judge** ({c['v2_rule_eval']['pass_rate']*100:.0f}%)")
                            for chk in c["v2_rule_eval"]["checks"]:
                                icon = "✅" if chk["passed"] else "❌"
                                st.markdown(f"{icon} `{chk['name']}` — {chk['detail']}")
