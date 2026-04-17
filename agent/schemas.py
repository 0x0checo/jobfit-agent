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


class CaseResult(BaseModel):
    """单个测试用例上两个 prompt 版本的对比结果。"""
    case_name: str
    v1_report: dict
    v2_report: dict
    v1_eval: EvalResult
    v2_eval: EvalResult
    winner: str = Field(description="v1 / v2 / tie")


class BenchReport(BaseModel):
    """评测台汇总报告。"""
    total_cases: int
    v1_wins: int
    v2_wins: int
    ties: int
    v1_avg_total: float
    v2_avg_total: float
    cases: List[CaseResult]


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
