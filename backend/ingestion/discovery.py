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


def parse_pdf_filename(pdf_path: Path) -> DocSpec | None:
    """Parse {COMPANY}_{YEAR}_{DOCTYPE}.pdf into a DocSpec."""
    parts = pdf_path.stem.split("_")
    if len(parts) < 3:
        return None

    doc_type_raw = parts[-1]
    year = parts[-2]
    if not _YEAR_RE.match(year):
        return None

    company = "_".join(parts[:-2])
    if len(doc_type_raw) >= 3 and doc_type_raw[:-1].isdigit():
        doc_type = doc_type_raw[:-1] + "-" + doc_type_raw[-1]
    else:
        doc_type = doc_type_raw

    return DocSpec(path=pdf_path, company=company, year=year, doc_type=doc_type)


def discover_pdfs(pdf_dir: Path = PDF_DIR) -> list[DocSpec]:
    """Return only PDFs that match the ingestion naming convention."""
    if not pdf_dir.exists():
        return []

    specs: list[DocSpec] = []
    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        spec = parse_pdf_filename(pdf_path)
        if spec is not None:
            specs.append(spec)
    return specs
