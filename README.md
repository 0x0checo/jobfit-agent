# JobFit Agent 🎯

AI 驱动的求职匹配 Agent：把简历 PDF 与目标 JD 喂进去，输出结构化匹配报告、差距分析与定制化改写建议。

> 面向 AI 产品经理岗位的作品项目，完整覆盖 "Agent 架构 / Prompt 工程 / 评测体系 / 数据驱动迭代" 四大能力。

## ✨ 核心能力

| 模块 | 功能 | 对应 PM 能力 |
|---|---|---|
| **简历解析 Agent** | PDF → 结构化 JSON（Pydantic Schema 约束 + Structured Output） | Schema 设计 / Prompt 工程 |
| **JD 分析 Agent** | 文本或 URL → 硬性要求 / 软性偏好 / 关键词分层抽取 | 需求建模 / 爬虫（Crawl4AI） |
| **匹配 Agent** | 简历 × JD → 加权评分 + 逐条差距分析 + 优先改进动作 | Agent 编排 / 业务逻辑 |
| **改写 Agent (RAG)** | Resume + JD + Report → 定制化改写，从 75 条优质 bullet 案例库检索 top-3 作 few-shot | RAG 设计 / 检索质量评估 |
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
├── app.py                   # Streamlit 前端（五 tab）
├── agent/
│   ├── schemas.py           # Pydantic Schema（Resume / JD / MatchReport / RewriteResult / EvalResult）
│   ├── resume_parser.py     # 模块 1：PDF 简历 → JSON
│   ├── jd_parser.py         # 模块 2：JD 文本/URL → JSON
│   ├── matcher.py           # 模块 3：匹配 Agent
│   ├── rewriter.py          # 模块 4：改写 Agent（RAG 增强）
│   ├── rag_retriever.py     # 模块 4a：bullet 案例库 numpy 向量检索
│   ├── graph.py             # LangGraph 多 Agent 编排
│   ├── eval_bench.py        # 模块 5a：Matcher Prompt 评测台
│   ├── eval_rag.py          # 模块 5b：RAG 效果 pairwise 评测
│   ├── rule_judge.py        # 确定性规则 Judge（补 LLM Judge 盲点）
│   ├── prompts.py           # Prompt 版本管理（v1_baseline / v2_fixed）
│   └── display_labels.py    # 中文 i18n 映射
├── scripts/
│   └── build_index.py       # 一次性脚本：bullet_corpus.jsonl → bullet_index.npz
├── utils/
│   ├── pdf_parser.py        # pdfplumber 封装（含 dedupe_chars 修复字符叠影）
│   └── web_scraper.py       # Crawl4AI 封装（支持 SPA 等待）
└── data/
    ├── bullet_corpus.jsonl  # 75 条 bullet 案例库
    ├── bullet_index.npz     # 向量索引（75 × 1536 float32）
    └── {resumes,jds,outputs}/
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

### 一个真实的三轮迭代故事（v1 → v2 → 回退 v1）

**轮次 1：发现问题**  
v1 在低匹配岗位打出 overall=73，soft_score=100 拉抬综合分——"JD 没软要求 = 软要求满分"是评估陷阱。

**轮次 2：尝试修复（v2）**  
v2 prompt 规则改为"soft_preferences 为空 → soft_score=null, overall=hard_score"。  
首次 LLM-as-Judge 评测：v2 平均分 10.67 > v1 9.67，低匹配场景 honesty 维度 2→3。

**轮次 3：引入 Rule Judge，发现反转**  
怀疑 LLM Judge 抓不到数值错误，加了 7 条确定性规则（分数公式、覆盖率、evidence 非空率等）。  
再跑评测，结果：
- LLM Judge 分数波动（v1 从 9.67 跳到 11.67 —— 同 prompt 多次跑不稳定）
- Rule Judge 揪出 v2 的 `hard_score` 数学回归
- 综合双通道，**v1 反胜（2-0-1）**

**最终决策**：默认 prompt 回退为 v1_baseline，v2 作为"修一个 bug 引入另一个 bug"的反例保留。

### 评测给我的三个真实教训

1. **LLM-as-Judge 自身有随机性**：同 prompt 同输入两次跑，分数能差 2 分。单次评测不可信，必须多 seed 取平均。
2. **Rule Judge 不可缺**：LLM Judge 擅长语义判断（建议是否空泛），但对数值错误盲视。生产级评测必须 rule + LLM 双通道。
3. **迭代不一定是前进**：v2 修了 soft 虚高却引入 hard 计算回归，"净负迭代"比想象中常见。评测体系存在的意义就是识别这种伪进步。

### v3（下一步方向）

- 双保留：v2 的 soft_score=None 逻辑 + 修复 hard_score 数学计算
- Multi-seed LLM Judge（每版跑 3 次取平均）
- 评测数据集扩到 10+ 用例，覆盖更多匹配层级

## 📚 RAG 案例库（Rewriter 增强）

### 方案设计

| 维度 | 取舍 |
|---|---|
| 语料 | 75 条手工构造的高质量 bullet（PM/AIPM 各 15，算法/运营/数据各 15） |
| Embedding | `text-embedding-3-small`（1536 维，$0.02/1M tokens） |
| 向量库 | numpy 点积（已 L2 归一化）——**有意不用 ChromaDB** |
| 检索时机 | Rewriter 改写前，以 `JD.title + hard_requirements + keywords` 为 query 检 top-3 |
| 注入方式 | 拼进 system prompt 作 few-shot，强约束"仅参考表述风格，不得抄袭/编造" |

**为什么不用 ChromaDB**：75 条规模 numpy 点积 < 1ms，零依赖利于 Streamlit Cloud 部署。规模扩展路径：1K+ 切 FAISS，10 万+ 才上向量数据库——按数据规模选型，避免过度工程。

### RAG 效果评测（pairwise LLM Judge）

对同一批 3 个测试 case 跑 `use_rag=True` vs `use_rag=False`，让 LLM 按 4 维度（关键词对齐 / 量化表达 / 专业度 / 真实性）做 A/B 盲评（位置随机化防 bias）。

运行：`python -m agent.eval_rag`

| case | RAG 总分 | noRAG 总分 | winner |
|---|---|---|---|
| high_match（AIPM 岗） | **19/20** | 18/20 | **RAG** ✓ |
| medium_match（游戏直播岗） | 17/20 | **18/20** | noRAG |
| low_match（低匹配岗） | 18/20 | **19/20** | noRAG |
| **平均** | **18.0/20** | **18.33/20** | RAG 小幅落后 |

### 从结果推出的 PM 结论

不掩盖：**当前 RAG 在小样本评测上整体没赢**。但细看分场景，结论很清晰：

1. **RAG 在强同域场景（AIPM 岗）上有增益** —— 语料库里 15 条 AIPM bullet 提供了有效锚点，关键词对齐 +1 分
2. **RAG 在语料覆盖面外的岗位上是噪声** —— 游戏直播岗检索回 PM/AIPM 类案例，反而把改写方向带偏
3. **RAG 在低匹配场景下效果不显** —— 改写空间本就小，few-shot 边际收益低

### v3 修复方向（明确且可执行）

1. **扩语料覆盖面**：补游戏 / 直播 / 硬件 / 金融等垂类 bullet，目标 200+ 条
2. **加 query-doc 相关度阈值**：若 top-1 相似度 < 0.5，跳过 RAG 注入（避免强塞无关案例）
3. **Prompt guardrail 加强**："若检索案例与岗位语境差异大则完全忽略"
4. **评测规模扩大**：3 case → 10 case，降低单次 LLM 评分噪声
5. **分场景部署策略**：对"同域岗位"默认开 RAG，对"跨域岗位"默认关

> **这份评测比让 RAG 赢更有价值。** —— 展示 PM 不是"堆技术点"，而是用量化方法识别技术方案的适用边界。

## 🎯 设计决策备注

- **为什么不用 LangGraph**：MVP 流程线性，手写 agent loop 能展示对执行链路的控制力；多 agent 协作时再引入。
- **为什么分硬/软要求**：差异化加权（0.7/0.3），避免软要求拉抬综合分；下游建议模块按类型定优先级。
- **为什么用 gpt-4o-mini**：抽取 / 分类任务复杂度低，mini 成本为 4o 的 1/10，符合 PM 成本-效果权衡。
- **为什么英文 key + 中文 label**：数据层遵循工程标准，展示层走 i18n 映射，可扩展多语言。
