"""PDF 文本提取：把简历 PDF 转成纯文本，供下游 LLM 结构化。"""
import pdfplumber


def extract_text_from_pdf(pdf_path: str) -> str:
    text_parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            # dedupe_chars: 去掉完全重叠的字形（有些简历模板会把同一字符叠画
            # 2-3 次做加粗/阴影，pdfplumber 会把它们都抽出来，导致"李李李文文文宇宇宇"）
            try:
                page = page.dedupe_chars(tolerance=1)
            except Exception:
                pass
            page_text = page.extract_text() or ""
            text_parts.append(page_text)
    return "\n".join(text_parts).strip()


if __name__ == "__main__":
    import sys
    print(extract_text_from_pdf(sys.argv[1]))
