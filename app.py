"""JobFit Agent - Streamlit 前端。

产品设计思路：
- 四个 tab 对应 MVP 四大模块，让面试官能逐步看到 agent 的每一层能力
- 左栏做"全局输入"（简历 + JD），右栏做"结果呈现"
- 中文标签通过 display_labels 映射，数据层保持英文键
"""
import json
import os
from pathlib import Path

import streamlit as st

# Streamlit Cloud secrets → 环境变量（本地用 .env，云端用 st.secrets）
if hasattr(st, "secrets") and "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    os.environ["OPENAI_MODEL"] = st.secrets.get("OPENAI_MODEL", "gpt-4o-mini")

IS_CLOUD = os.environ.get("STREAMLIT_RUNTIME_ENV") == "cloud" or "STREAMLIT_SERVER_PORT" in os.environ

from agent.resume_parser import parse_resume
from agent.jd_parser import parse_jd
from agent.matcher import match
from agent.schemas import Resume, JobDescription, MatchReport
from agent.display_labels import label

st.set_page_config(page_title="JobFit Agent", page_icon="🎯", layout="wide")

# ---------- 状态 ----------
if "resume" not in st.session_state:
    st.session_state.resume = None
if "jd" not in st.session_state:
    st.session_state.jd = None
if "match_report" not in st.session_state:
    st.session_state.match_report = None

DATA_DIR = Path("data")
UPLOAD_DIR = DATA_DIR / "resumes"
OUTPUT_DIR = DATA_DIR / "outputs"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------- 标题 ----------
st.title("🎯 JobFit Agent")
st.caption("AI 驱动的求职匹配助手 · PDF 简历 + JD → 结构化匹配分析与改写建议")

tabs = st.tabs(["1️⃣ 简历解析", "2️⃣ JD 分析", "3️⃣ 匹配报告", "4️⃣ Prompt 评测台"])


# ---------- Tab 1: 简历解析 ----------
with tabs[0]:
    st.subheader("上传你的简历 PDF")
    uploaded = st.file_uploader("选择 PDF 文件", type=["pdf"])

    col1, col2 = st.columns([1, 2])
    with col1:
        if uploaded and st.button("🚀 解析简历", type="primary"):
            pdf_path = UPLOAD_DIR / uploaded.name
            pdf_path.write_bytes(uploaded.getbuffer())
            with st.spinner("正在解析..."):
                st.session_state.resume = parse_resume(str(pdf_path))
            st.success("解析完成 ✅")

    if st.session_state.resume:
        resume = st.session_state.resume
        with col1:
            st.markdown(f"**{label('name')}**：{resume.name}")
            st.markdown(f"**{label('email')}**：{resume.email or '-'}")
            st.markdown(f"**{label('phone')}**：{resume.phone or '-'}")
            st.markdown(f"**{label('website')}**：{resume.website or '-'}")
            st.markdown(f"**{label('summary')}**")
            st.info(resume.summary or "-")

        with col2:
            st.markdown(f"### {label('education')}")
            for edu in resume.education:
                st.markdown(f"- **{edu.school}** · {edu.major or ''} {edu.degree or ''}  ({edu.start_date} - {edu.end_date})")

            st.markdown(f"### {label('experience')}")
            for exp in resume.experience:
                with st.expander(f"{exp.company} · {exp.role} ({exp.start_date} - {exp.end_date})"):
                    for b in exp.bullets:
                        st.markdown(f"- {b}")

            st.markdown(f"### {label('skills')}")
            sk = resume.skills
            if sk.languages:
                st.markdown(f"**{label('languages')}**：{' · '.join(sk.languages)}")
            if sk.programming:
                st.markdown(f"**{label('programming')}**：{' · '.join(sk.programming)}")
            if sk.ai_tools:
                st.markdown(f"**{label('ai_tools')}**：{' · '.join(sk.ai_tools)}")
            if sk.product_tools:
                st.markdown(f"**{label('product_tools')}**：{' · '.join(sk.product_tools)}")


# ---------- Tab 2: JD 分析 ----------
with tabs[1]:
    st.subheader("输入目标岗位 JD")
    modes = ["粘贴文本"] if IS_CLOUD else ["粘贴文本", "抓取 URL"]
    if IS_CLOUD:
        st.caption("🌐 线上环境仅支持文本输入；URL 抓取（Crawl4AI）需本地运行。")
    mode = st.radio("输入方式", modes, horizontal=True)
    if mode == "粘贴文本":
        jd_text = st.text_area("粘贴 JD 全文", height=300)
        if st.button("🔍 分析 JD（文本）", type="primary"):
            if jd_text.strip():
                with st.spinner("正在分析..."):
                    st.session_state.jd = parse_jd(text=jd_text)
                st.success("分析完成 ✅")
    else:
        url = st.text_input("JD 页面 URL（建议用官方招聘页）")
        if st.button("🔍 分析 JD（URL）", type="primary"):
            if url.strip():
                with st.spinner("抓取并分析中（首次约 5-10 秒）..."):
                    st.session_state.jd = parse_jd(url=url)
                st.success("分析完成 ✅")

    if st.session_state.jd:
        jd = st.session_state.jd
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"### {jd.title}")
            st.caption(f"{jd.company or ''} · {jd.location or ''}")
            if jd.team_intro:
                with st.expander(label("team_intro", "jd")):
                    st.write(jd.team_intro)
            st.markdown(f"**{label('keywords', 'jd')}**")
            st.write(" · ".join([f"`{k}`" for k in jd.keywords]))
        with c2:
            st.markdown(f"**{label('responsibilities', 'jd')}**")
            for r in jd.responsibilities:
                st.markdown(f"- {r}")
            st.markdown(f"**{label('hard_requirements', 'jd')}**")
            for r in jd.hard_requirements:
                st.markdown(f"- {r}")
            if jd.soft_preferences:
                st.markdown(f"**{label('soft_preferences', 'jd')}**")
                for r in jd.soft_preferences:
                    st.markdown(f"- {r}")


# ---------- Tab 3: 匹配报告 ----------
with tabs[2]:
    if not st.session_state.resume or not st.session_state.jd:
        st.warning("请先在 Tab 1 和 Tab 2 完成简历和 JD 的分析")
    else:
        prompt_ver = st.selectbox("选择 Prompt 版本", ["v2_fixed", "v1_baseline"], index=0)
        if st.button("🎯 生成匹配报告", type="primary"):
            with st.spinner("匹配分析中..."):
                st.session_state.match_report = match(
                    st.session_state.resume,
                    st.session_state.jd,
                    prompt_version=prompt_ver,
                )
            st.success("报告生成完成 ✅")

        if st.session_state.match_report:
            rep = st.session_state.match_report
            c1, c2, c3 = st.columns(3)
            c1.metric("综合匹配度", f"{rep.overall_score}")
            c2.metric("硬性要求", f"{rep.hard_score}")
            c3.metric("软性偏好", f"{rep.soft_score if rep.soft_score is not None else 'N/A'}")

            st.info(f"**总结**：{rep.summary}")

            st.markdown("### ✨ 核心优势")
            for s in rep.strengths:
                st.markdown(f"- {s}")

            st.markdown("### 🎯 优先改进动作")
            for i, a in enumerate(rep.top_actions, 1):
                st.markdown(f"**{i}.** {a}")

            st.markdown("### 🔑 关键词命中")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**✅ 命中**")
                st.write(" · ".join([f"`{k}`" for k in rep.keyword_hits]) or "-")
            with c2:
                st.markdown("**❌ 未命中**")
                st.write(" · ".join([f"`{k}`" for k in rep.keyword_misses]) or "-")

            st.markdown("### 📋 逐条差距分析")
            for g in rep.gaps:
                color = {"full": "🟢", "partial": "🟡", "missing": "🔴"}.get(g.match_level, "⚪")
                with st.expander(f"{color} [{g.req_type}] {g.requirement}"):
                    if g.evidence:
                        st.markdown(f"**简历证据**：{g.evidence}")
                    st.markdown(f"**改进建议**：{g.suggestion}")


# ---------- Tab 4: Prompt 评测台 ----------
with tabs[3]:
    st.subheader("Prompt 版本对比评测")
    st.caption("LLM-as-a-Judge · 三维度打分（诚实度 / 具体度 / 覆盖度）· 对应 JD 要求的「大模型 prompt 调优 + 效果评测经验」")

    bench_path = OUTPUT_DIR / "bench_report.json"
    if not bench_path.exists():
        st.info("尚未运行评测。请在终端运行：`python -m agent.eval_bench`")
    else:
        data = json.loads(bench_path.read_text(encoding="utf-8"))
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("测试用例数", data["total_cases"])
        c2.metric("v2_fixed 胜场", data["v2_wins"])
        c3.metric("v1_baseline 胜场", data["v1_wins"])
        c4.metric("平局", data["ties"])

        c1, c2 = st.columns(2)
        c1.metric("v1 baseline 平均分", f"{data['v1_avg_total']} / 15")
        c2.metric("v2 fixed 平均分", f"{data['v2_avg_total']} / 15", delta=round(data['v2_avg_total']-data['v1_avg_total'],2))

        st.markdown("### 逐案对比")
        for c in data["cases"]:
            with st.expander(f"{c['case_name']} · 胜者：{c['winner']}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**v1 baseline**")
                    st.json(c["v1_eval"])
                with col2:
                    st.markdown("**v2 fixed**")
                    st.json(c["v2_eval"])

        st.markdown("### 💡 评测发现（项目迭代笔记）")
        st.markdown("""
- v2_fixed 平均分 +1 分，在低匹配场景优势最大（修复 soft_score 虚高）
- v2 暴露新 bug：hard_score 数学计算偶发回归（prompt 改动引入）
- LLM-as-Judge 的盲区：抓不到数值错误，需要补规则型 judge 做双重校验
- 下一步 v3：加数值后处理校验 + rule-based judge
        """)
