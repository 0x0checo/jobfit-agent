"""结构化 Schema 定义 —— 对应毕业论文中的 Schema 约束方法论。

用 Pydantic 做强类型约束，喂给 LLM 做 structured output，
保证 JSON 输出字段完整、类型正确，降低结构性错误率。
"""
from typing import List, Optional
from pydantic import BaseModel, Field


class Education(BaseModel):
    school: str
    degree: Optional[str] = None
    major: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    gpa: Optional[str] = None
    highlights: List[str] = Field(default_factory=list)


class Experience(BaseModel):
    company: str
    role: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    location: Optional[str] = None
    bullets: List[str] = Field(default_factory=list, description="核心工作成果，每条一句")


class Skills(BaseModel):
    languages: List[str] = Field(default_factory=list, description="自然语言")
    programming: List[str] = Field(default_factory=list)
    ai_tools: List[str] = Field(default_factory=list)
    product_tools: List[str] = Field(default_factory=list)
    other: List[str] = Field(default_factory=list)


class GapItem(BaseModel):
    """单条差距分析。"""
    requirement: str = Field(description="JD 中的具体要求原文")
    req_type: str = Field(description="hard 或 soft")
    match_level: str = Field(description="full / partial / missing")
    evidence: Optional[str] = Field(None, description="简历里支撑这条匹配的原文片段，missing 时为空")
    suggestion: str = Field(description="针对性的改进建议")


class MatchReport(BaseModel):
    """简历与 JD 的匹配分析报告。"""
    overall_score: int = Field(description="综合匹配度 0-100")
    hard_score: int = Field(description="硬性要求匹配度 0-100")
    soft_score: Optional[int] = Field(None, description="软性偏好匹配度 0-100；JD 无软性要求时为 None")
    summary: str = Field(description="一句话总结匹配情况")
    strengths: List[str] = Field(default_factory=list, description="候选人相对 JD 的核心优势")
    gaps: List[GapItem] = Field(default_factory=list, description="逐条需求的匹配分析")
    keyword_hits: List[str] = Field(default_factory=list, description="简历命中的 JD 关键词")
    keyword_misses: List[str] = Field(default_factory=list, description="简历未命中的 JD 关键词")
    top_actions: List[str] = Field(default_factory=list, description="最优先的 3-5 条修改动作")


class RewriteBullet(BaseModel):
    """单条 bullet 的 before/after 改写。"""
    experience_company: str = Field(description="归属于哪段经历（公司名）")
    before: str = Field(description="原 bullet 原文")
    after: str = Field(description="改写后 bullet")
    change_reason: str = Field(description="改写理由：对应 JD 哪条要求、用了什么技术/产品表述")


class RewriteResult(BaseModel):
    """改写 Agent 输出。"""
    rewritten_summary: Optional[str] = Field(None, description="改写后的个人定位（summary）")
    summary_change_reason: Optional[str] = None
    rewritten_bullets: List[RewriteBullet] = Field(default_factory=list)
    new_keywords_added: List[str] = Field(default_factory=list, description="新植入的 JD 关键词")
    notes: Optional[str] = Field(None, description="整体改写策略说明")
    # RAG 检索到的参考 bullet（由 Rewriter 填入，LLM 不生成此字段）
    rag_references: List[dict] = Field(default_factory=list, description="RAG 检索到的参考案例")


class EvalScore(BaseModel):
    """单维度评分：0-5 分 + 一句理由。"""
    score: int = Field(ge=0, le=5)
    reason: str


class EvalResult(BaseModel):
    """一次 MatchReport 的四维评分。"""
    honesty: EvalScore = Field(description="分数是否反映真实匹配度，不虚高/不压低")
    specificity: EvalScore = Field(description="建议是否具体到'在哪段加什么'")
    coverage: EvalScore = Field(description="是否覆盖所有 JD 要求条目")
    consistency_note: Optional[str] = Field(None, description="一致性备注（两次运行对比时用）")
    total: int = Field(description="三维总分 0-15")


class RuleCheck(BaseModel):
    """单条规则校验结果。"""
    name: str
    passed: bool
    detail: str = ""


class RuleEvalResult(BaseModel):
    """规则型评估结果：0-5 分映射自通过率，附详细 check 列表。"""
    checks: List[RuleCheck] = Field(default_factory=list)
    pass_rate: float = Field(description="通过率 0-1")
    score: int = Field(description="折算到 0-5 分")


class CaseResult(BaseModel):
    """单个测试用例上两个 prompt 版本的对比结果。"""
    case_name: str
    v1_report: dict
    v2_report: dict
    v1_eval: EvalResult
    v2_eval: EvalResult
    v1_rule_eval: Optional[RuleEvalResult] = None
    v2_rule_eval: Optional[RuleEvalResult] = None
    winner: str = Field(description="v1 / v2 / tie")
    judge_disagreement: Optional[str] = Field(None, description="LLM-judge 和 rule-judge 分歧的用例说明")


class BenchReport(BaseModel):
    """评测台汇总报告。"""
    total_cases: int
    v1_wins: int
    v2_wins: int
    ties: int
    v1_avg_total: float
    v2_avg_total: float
    cases: List[CaseResult]


class InterviewQuestion(BaseModel):
    """单道面试题。"""
    category: str = Field(description="resume_deepdive / gap_probe / domain_open")
    question: str = Field(description="问题本身")
    intent: str = Field(description="考察意图（面试官视角为什么问这题）")
    answer_hints: List[str] = Field(default_factory=list, description="参考答题要点（STAR 关键锚点，给用户复盘用）")
    difficulty: str = Field(description="easy / medium / hard")
    linked_requirement: Optional[str] = Field(None, description="对应 JD 哪条要求 / 简历哪段经历")


class InterviewSet(BaseModel):
    """一次生成的面试题集合。"""
    persona: str = Field(description="面试官人设：tech / product / hr")
    persona_style_note: str = Field(description="该人设的风格描述，一句话")
    resume_deepdive: List[InterviewQuestion] = Field(default_factory=list, description="简历深挖题")
    gap_probe: List[InterviewQuestion] = Field(default_factory=list, description="差距探测题")
    domain_open: List[InterviewQuestion] = Field(default_factory=list, description="岗位专业开放题")


class FollowUpResult(BaseModel):
    """针对用户答案生成的追问 + 快速反馈。"""
    needs_followup: bool = Field(description="是否需要追问（答案是否足够深入）")
    followup_question: Optional[str] = Field(None, description="追问内容；needs_followup=False 时为 None")
    quick_feedback: str = Field(description="对刚才答案的一句话点评（是否有 STAR 结构、量化、关键词命中）")
    strengths: List[str] = Field(default_factory=list, description="答案亮点")
    weaknesses: List[str] = Field(default_factory=list, description="答案薄弱点")


class JobDescription(BaseModel):
    """JD 结构化表示。区分硬/软要求，方便下游匹配模块做差异化加权。"""
    title: str = Field(description="岗位名称")
    company: Optional[str] = None
    location: Optional[str] = None
    team_intro: Optional[str] = Field(None, description="团队/业务介绍")
    responsibilities: List[str] = Field(default_factory=list, description="岗位职责")
    hard_requirements: List[str] = Field(default_factory=list, description="硬性要求：学历/专业/技能/年限等")
    soft_preferences: List[str] = Field(default_factory=list, description="软性加分项：优先/最好具备")
    keywords: List[str] = Field(default_factory=list, description="核心关键词：大模型/Agent/RAG/SQL 等")
    salary: Optional[str] = None


class Resume(BaseModel):
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    website: Optional[str] = None
    summary: Optional[str] = Field(None, description="个人定位/自我介绍")
    education: List[Education] = Field(default_factory=list)
    experience: List[Experience] = Field(default_factory=list)
    skills: Skills = Field(default_factory=Skills)
