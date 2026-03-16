from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PDF_DIR = REPO_ROOT / "data" / "financebench" / "pdfs"

_YEAR_RE = re.compile(r"^\d{4}(Q[1-4])?$")


@dataclass
class DocSpec:
    path: Path
    company: str
    year: str
    doc_type: str = "10-K"


def parse_pdf_filename(pdf_path: Path) -> DocSpec:
    """Parse {COMPANY}_{YEAR}_{DOCTYPE}.pdf into a DocSpec.

    Falls back gracefully for filenames that do not match the convention:
    company defaults to the full stem, year to "0000", doc_type to "OTHER".
    Never returns None — all PDFs are accepted.
    """
    parts = pdf_path.stem.split("_")
    if len(parts) >= 3:
        doc_type_raw = parts[-1]
        year = parts[-2]
        if _YEAR_RE.match(year):
            company = "_".join(parts[:-2])
            if len(doc_type_raw) >= 3 and doc_type_raw[:-1].isdigit():
                doc_type = doc_type_raw[:-1] + "-" + doc_type_raw[-1]
            else:
                doc_type = doc_type_raw
            return DocSpec(path=pdf_path, company=company, year=year, doc_type=doc_type)

    # Fallback: use the full stem as company, sentinel year, generic doc_type
    return DocSpec(path=pdf_path, company=pdf_path.stem, year="0000", doc_type="OTHER")


def discover_pdfs(pdf_dir: Path = PDF_DIR) -> list[DocSpec]:
    """Return DocSpecs for every PDF found in pdf_dir, regardless of filename shape."""
    if not pdf_dir.exists():
        return []

    return [parse_pdf_filename(pdf_path) for pdf_path in sorted(pdf_dir.glob("*.pdf"))]
