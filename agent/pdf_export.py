"""Markdown → 带中文支持的 PDF 导出（纯 Python，无系统依赖）。

用 fpdf2 手动渲染：按行扫 Markdown，识别标题/列表/引用/粗体，输出简洁简历风 PDF。
字体策略：扫常见系统路径注册 TTF/TTC，降级时以英文字体兜底。
"""
import os
import re
import urllib.request
from pathlib import Path
from io import BytesIO

from fpdf import FPDF


# fpdf2 对 .ttc 字宽处理有 bug（字符三重叠影），坚持用 .ttf。
# 优先使用本地已有的 .ttf，否则首次运行时下载 Google Fonts Noto Sans SC Regular。
_NOTO_URL = "https://github.com/google/fonts/raw/main/ofl/notosanssc/NotoSansSC%5Bwght%5D.ttf"
_CACHE_DIR = Path.home() / ".cache" / "jobfit"
_CACHE_FONT = _CACHE_DIR / "NotoSansSC-Regular.ttf"

# 本地 .ttf 字体候选（跳过 .ttc 集合文件）
_LOCAL_TTF_CANDIDATES = [
    "/usr/share/fonts/truetype/noto/NotoSansSC-Regular.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansSC-Regular.otf",
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/simsun.ttc",  # Windows 下 ttc 相对安全
]


def _ensure_cjk_font() -> str | None:
    # 1) 本地已有 ttf
    for p in _LOCAL_TTF_CANDIDATES:
        if os.path.exists(p):
            return p
    # 2) 缓存字体已下载
    if _CACHE_FONT.exists() and _CACHE_FONT.stat().st_size > 100_000:
        return str(_CACHE_FONT)
    # 3) 首次下载
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(_NOTO_URL, _CACHE_FONT)
        return str(_CACHE_FONT)
    except Exception:
        return None


def _strip_inline_md(s: str) -> str:
    """去掉行内 **加粗**、`code`、[text](url) 里的 markdown 标记，保留文字。"""
    s = re.sub(r"\*\*(.+?)\*\*", r"\1", s)
    s = re.sub(r"\*(.+?)\*", r"\1", s)
    s = re.sub(r"`(.+?)`", r"\1", s)
    s = re.sub(r"\[(.+?)\]\((.+?)\)", r"\1", s)
    return s


def markdown_to_pdf(markdown_text: str) -> bytes:
    pdf = FPDF(format="A4", unit="mm")
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()
    pdf.set_margins(18, 18, 18)

    font_path = _ensure_cjk_font()
    if font_path:
        pdf.add_font("CJK", "", font_path)
        base_font = "CJK"
    else:
        base_font = "Helvetica"  # 兜底，中文会显示不全

    PRIMARY = (99, 102, 241)     # indigo-500
    TEXT = (17, 24, 39)          # gray-900
    MUTED = (107, 114, 128)      # gray-500
    ACCENT = (31, 41, 55)        # gray-800

    for raw in markdown_text.splitlines():
        line = raw.rstrip()

        # 每行开头重置 x 到左边距，避免上一行的 set_x 残留导致宽度异常
        pdf.set_x(pdf.l_margin)

        # 空行
        if not line.strip():
            pdf.ln(2.5)
            continue

        # H1
        if line.startswith("# "):
            pdf.set_font(base_font, "", 20)
            pdf.set_text_color(*TEXT)
            pdf.multi_cell(0, 9, _strip_inline_md(line[2:]))
            pdf.set_draw_color(*PRIMARY)
            pdf.set_line_width(0.5)
            pdf.line(pdf.l_margin, pdf.get_y() + 1, pdf.w - pdf.r_margin, pdf.get_y() + 1)
            pdf.ln(5)
            continue

        # H2
        if line.startswith("## "):
            pdf.ln(2)
            pdf.set_font(base_font, "", 14)
            pdf.set_text_color(*ACCENT)
            pdf.multi_cell(0, 7, _strip_inline_md(line[3:]))
            pdf.ln(1)
            continue

        # H3
        if line.startswith("### "):
            pdf.set_font(base_font, "", 11.5)
            pdf.set_text_color(*TEXT)
            pdf.multi_cell(0, 6, _strip_inline_md(line[4:]))
            continue

        # 列表
        if line.lstrip().startswith(("- ", "* ")):
            indent_level = (len(line) - len(line.lstrip())) // 2
            bullet = "•" if indent_level == 0 else "◦"
            text = _strip_inline_md(line.lstrip()[2:])
            pdf.set_font(base_font, "", 10.5)
            pdf.set_text_color(*TEXT)
            pdf.set_x(pdf.l_margin + 4 + indent_level * 5)
            pdf.multi_cell(0, 5.5, f"{bullet}  {text}")
            continue

        # 引用
        if line.startswith("> "):
            pdf.set_font(base_font, "", 10)
            pdf.set_text_color(*MUTED)
            pdf.set_x(pdf.l_margin + 3)
            pdf.multi_cell(0, 5, _strip_inline_md(line[2:]))
            continue

        # 正文
        pdf.set_font(base_font, "", 10.5)
        pdf.set_text_color(*TEXT)
        pdf.multi_cell(0, 5.5, _strip_inline_md(line))

    out = pdf.output(dest="S")  # fpdf2 2.7+ 返回 bytearray
    return bytes(out)


if __name__ == "__main__":
    import sys
    from pathlib import Path
    p = Path(sys.argv[1])
    Path(p.with_suffix(".pdf")).write_bytes(markdown_to_pdf(p.read_text(encoding="utf-8")))
    print(f"✅ 导出：{p.with_suffix('.pdf')}")
