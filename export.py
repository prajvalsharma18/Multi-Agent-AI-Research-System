"""Export research reports to Markdown and PDF."""

import re
from datetime import datetime
from pathlib import Path

import markdown
from xhtml2pdf import pisa

OUTPUT_DIR = Path(__file__).parent / "reports"


def _slugify(text: str) -> str:
    slug = re.sub(r"[^\w\s-]", "", text.lower())
    slug = re.sub(r"[\s_-]+", "-", slug).strip("-")
    return slug[:60] or "report"


def save_report(
    topic: str,
    report: str,
    feedback: str | None = None,
) -> dict[str, str]:
    """Save report as .md and .pdf. Returns paths dict."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"{_slugify(topic)}_{timestamp}"

    md_path = OUTPUT_DIR / f"{base}.md"
    pdf_path = OUTPUT_DIR / f"{base}.pdf"

    md_content = f"# {topic}\n\n*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n\n{report}"
    if feedback:
        md_content += f"\n\n---\n\n## Critic Review\n\n{feedback}"

    md_path.write_text(md_content, encoding="utf-8")

    html = markdown.markdown(md_content, extensions=["tables", "fenced_code"])
    styled_html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
  body {{ font-family: Georgia, serif; margin: 40px; line-height: 1.6; color: #222; }}
  h1 {{ color: #1a1a2e; border-bottom: 2px solid #16213e; padding-bottom: 8px; }}
  h2 {{ color: #16213e; margin-top: 24px; }}
  li {{ margin-bottom: 4px; }}
  code {{ background: #f4f4f4; padding: 2px 4px; }}
</style></head><body>{html}</body></html>"""

    with open(pdf_path, "wb") as pdf_file:
        pisa.CreatePDF(styled_html, dest=pdf_file)

    return {"markdown": str(md_path), "pdf": str(pdf_path)}
