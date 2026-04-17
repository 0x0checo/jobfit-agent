"""字段中文标签映射：数据层用英文 key，展示层用中文 label。

这是一个极简 i18n 设计：
- 代码、日志、存储都用英文键，保证工程标准
- Streamlit 前端渲染时查这个映射，展示给用户的是中文
"""

RESUME_LABELS = {
    "name": "姓名",
    "email": "邮箱",
    "phone": "电话",
    "website": "个人网站",
    "summary": "个人定位",
    "education": "教育经历",
    "experience": "实践经历",
    "skills": "技能",
    "school": "学校",
    "degree": "学位",
    "major": "专业",
    "start_date": "开始时间",
    "end_date": "结束时间",
    "gpa": "GPA",
    "highlights": "亮点",
    "company": "公司",
    "role": "职位",
    "location": "地点",
    "bullets": "工作内容",
    "languages": "语言能力",
    "programming": "编程",
    "ai_tools": "AI 工具",
    "product_tools": "产品工具",
    "other": "其他",
}

JD_LABELS = {
    "title": "岗位名称",
    "company": "公司",
    "location": "工作地点",
    "team_intro": "团队介绍",
    "responsibilities": "岗位职责",
    "hard_requirements": "硬性要求",
    "soft_preferences": "加分项",
    "keywords": "核心关键词",
    "salary": "薪资",
}


def label(key: str, domain: str = "resume") -> str:
    """查字段中文标签，找不到则返回原 key。"""
    mapping = RESUME_LABELS if domain == "resume" else JD_LABELS
    return mapping.get(key, key)
