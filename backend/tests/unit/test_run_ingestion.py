from __future__ import annotations

from pathlib import Path

import pytest

from ingestion.run_ingestion import _parse_pdf_filename


def _path(name: str) -> Path:
    return Path(name)


def test_standard_three_part_filename():
    spec = _parse_pdf_filename(_path("3M_2022_10K.pdf"))
    assert spec is not None
    assert spec.company == "3M"
    assert spec.year == "2022"
    assert spec.doc_type == "10-K"


def test_company_with_underscore():
    spec = _parse_pdf_filename(_path("JOHNSON_JOHNSON_2022_10K.pdf"))
    assert spec is not None
    assert spec.company == "JOHNSON_JOHNSON"
    assert spec.year == "2022"
    assert spec.doc_type == "10-K"


def test_four_part_numeric_suffix_rejected():
    # "3M_2022_10K_10.pdf" → parts[-2]="10K" → fails year regex → None
    spec = _parse_pdf_filename(_path("3M_2022_10K_10.pdf"))
    assert spec is None


def test_too_few_parts_rejected():
    spec = _parse_pdf_filename(_path("BADNAME.pdf"))
    assert spec is None


def test_10q_doc_type_hyphenated():
    spec = _parse_pdf_filename(_path("APPLE_2023_10Q.pdf"))
    assert spec is not None
    assert spec.doc_type == "10-Q"


def test_10q_with_quarter_year():
    spec = _parse_pdf_filename(_path("MSFT_2023Q1_10Q.pdf"))
    assert spec is not None
    assert spec.year == "2023Q1"
    assert spec.doc_type == "10-Q"
