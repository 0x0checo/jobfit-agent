"""PDF 文本提取：把简历 PDF 转成纯文本，供下游 LLM 结构化。"""
import pdfplumber


def extract_text_from_pdf(pdf_path: str) -> str:
    text_parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text_parts.append(page_text)
    return "\n".join(text_parts).strip()


if __name__ == "__main__":
    import sys
    print(extract_text_from_pdf(sys.argv[1]))
