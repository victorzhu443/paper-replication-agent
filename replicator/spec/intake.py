"""Intake: PDF -> per-page PNGs + per-page text. The PDF itself is the source of truth for the
spec call; the derived artifacts serve companion discovery, claim evidence, and the re-reader."""
from __future__ import annotations

import re
from pathlib import Path

import httpx
import pymupdf


def fetch_arxiv(arxiv_id: str, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    url = f"https://arxiv.org/pdf/{arxiv_id}"
    r = httpx.get(url, timeout=120, follow_redirects=True)
    r.raise_for_status()
    p = dest / f"{arxiv_id.replace('/', '_')}.pdf"
    p.write_bytes(r.content)
    return p


def fetch_html_as_text(url: str, dest: Path, name: str) -> Path:
    """HTML-only papers: strip tags to text; no page images, so claim re-reads are skipped."""
    import re as _re
    dest.mkdir(parents=True, exist_ok=True)
    r = httpx.get(url, timeout=120, follow_redirects=True, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    html = r.text
    html = _re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=_re.S | _re.I)
    text = _re.sub(r"<[^>]+>", " ", html)
    text = _re.sub(r"&nbsp;", " ", text)
    text = _re.sub(r"[ \t]+", " ", text)
    text = _re.sub(r"\n\s*\n+", "\n\n", text)
    p = dest / f"{name}.txt"
    p.write_text(text)
    return p


def render(pdf: Path, out: Path, dpi: int = 110) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    if pdf.suffix.lower() != ".pdf":
        full = pdf.read_text()
        (out / "paper.txt").write_text(full)
        return {"n_pages": 0, "pages": [], "companions": find_companions(full), "arxiv_version": _arxiv_version(full)}
    doc = pymupdf.open(pdf)
    pages = []
    text_all = []
    for i, page in enumerate(doc):
        png = out / f"p{i+1:03d}.png"
        if not png.exists():
            page.get_pixmap(dpi=dpi).save(png)
        txt = page.get_text()
        (out / f"p{i+1:03d}.txt").write_text(txt)
        pages.append({"page": i + 1, "png": str(png), "chars": len(txt)})
        text_all.append(txt)
    full = "\n".join(text_all)
    (out / "paper.txt").write_text(full)
    return {"n_pages": len(pages), "pages": pages, "companions": find_companions(full),
            "arxiv_version": _arxiv_version(full)}


_LINK = re.compile(r"https?://(?:github\.com|gitlab\.com|zenodo\.org|osf\.io|huggingface\.co)/[\w\-./]+")


def find_companions(text: str) -> list[str]:
    return sorted(set(m.rstrip(".)") for m in _LINK.findall(text)))


def _arxiv_version(text: str) -> str | None:
    """The paper's own id is stamped in the left margin of page 1 as 'arXiv:ID [cat] date';
    citations in the body look different. Prefer the stamped form; fall back to the first mention."""
    m = re.search(r"arXiv:(\d{4}\.\d{4,5})(v\d+)?\s*\[[\w.\-]+\]\s+\d{1,2}\s+\w{3}\s+\d{4}", text)
    if not m:
        m = re.search(r"arXiv:\s*(\d{4}\.\d{4,5})(v\d+)?", text)
    return (m.group(1) + (m.group(2) or "")) if m else None
