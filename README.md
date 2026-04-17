# JobFit Agent 🎯

AI 驱动的求职匹配 Agent：把简历 PDF 与目标 JD 喂进去，输出结构化匹配报告、差距分析与定制化改写建议。

> 面向 AI 产品经理岗位的作品项目，完整覆盖 "Agent 架构 / Prompt 工程 / 评测体系 / 数据驱动迭代" 四大能力。

## ✨ 核心能力

| 模块 | 功能 | 对应 PM 能力 |
|---|---|---|
| **简历解析 Agent** | PDF → 结构化 JSON（Pydantic Schema 约束 + Structured Output） | Schema 设计 / Prompt 工程 |
| **JD 分析 Agent** | 文本或 URL → 硬性要求 / 软性偏好 / 关键词分层抽取 | 需求建模 / 爬虫（Crawl4AI） |
| **匹配 Agent** | 简历 × JD → 加权评分 + 逐条差距分析 + 优先改进动作 | Agent 编排 / 业务逻辑 |
| **Prompt 评测台** | 多版本 Prompt A/B 对比，LLM-as-Judge 三维度打分 | 大模型评测 / 迭代闭环 |

## 🧱 技术栈

- **LLM**：OpenAI GPT-4o-mini（MVP 成本优化；架构层 LLM 抽象便于切换）
- **Agent 实现**：手写 agent loop（不引 LangGraph，MVP 避免过度工程）
- **Schema**：Pydantic v2（强类型 + JSON Schema 自动生成）
- **PDF 解析**：pdfplumber
- **爬虫**：Crawl4AI（SPA 网页的 Markdown 化，LLM 友好）
- **前端**：Streamlit

## 🚀 快速开始

```bash
conda create -n jobfit python=3.11 -y
conda activate jobfit
pip install -r requirements.txt
playwright install chromium   # Crawl4AI 依赖
cp .env.example .env          # 填入 OPENAI_API_KEY
streamlit run app.py
```

## 📂 项目结构

```
AI Agent项目/
├── app.py                   # Streamlit 前端（四 tab）
├── agent/
│   ├── schemas.py           # Pydantic Schema（Resume / JD / MatchReport / EvalResult）
│   ├── resume_parser.py     # 模块 1：PDF 简历 → JSON
│   ├── jd_parser.py         # 模块 2：JD 文本/URL → JSON
│   ├── matcher.py           # 模块 3：匹配 Agent
│   ├── eval_bench.py        # 模块 4：Prompt 评测台
│   ├── prompts.py           # Prompt 版本管理（v1_baseline / v2_fixed）
│   └── display_labels.py    # 中文 i18n 映射
├── utils/
│   ├── pdf_parser.py        # pdfplumber 封装
│   └── web_scraper.py       # Crawl4AI 封装（支持 SPA 等待）
└── data/{resumes,jds,outputs}/
```

## 🔬 Prompt 迭代与评测

### 评测流程

```
测试集 (3 个匹配层级：高/中/低)
  → Prompt v1_baseline / v2_fixed 并行跑
  → LLM Judge 按 honesty / specificity / coverage 三维度打分
  → 汇总胜率与平均分
```

运行：`python -m agent.eval_bench`

### 一个真实的迭代故事

**发现问题**：v1 在低匹配岗位打出 overall=73 分，soft_score=100 拉抬综合分——"JD 没软要求 = 软要求满分"是评估陷阱。

**修复动作**：v2 prompt 规则改为"soft_preferences 为空 → soft_score=null, overall=hard_score"。

**评测结果**：v2 平均分 10.67 vs v1 9.67，低匹配场景 honesty 维度 2 → 3。

**但**：v2 在低匹配案例上出现 hard_score 数学计算回归（3 条非零匹配被算成 0），而 LLM Judge 未能识别——**揭示了 LLM-as-Judge 对数值错误的盲区**。

**下一步 v3**：加数值后处理校验 + rule-based judge 做双重评估。

## 🎯 设计决策备注

- **为什么不用 LangGraph**：MVP 流程线性，手写 agent loop 能展示对执行链路的控制力；多 agent 协作时再引入。
- **为什么分硬/软要求**：差异化加权（0.7/0.3），避免软要求拉抬综合分；下游建议模块按类型定优先级。
- **为什么用 gpt-4o-mini**：抽取 / 分类任务复杂度低，mini 成本为 4o 的 1/10，符合 PM 成本-效果权衡。
- **为什么英文 key + 中文 label**：数据层遵循工程标准，展示层走 i18n 映射，可扩展多语言。
