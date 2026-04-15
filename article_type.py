import re
from typing import List


EBM_TYPE_MAP = {
    "practice guideline": "clinical_practice_guideline",
    "guideline": "clinical_practice_guideline",
    "systematic review": "systematic_review",
    "meta-analysis": "meta_analysis",
    "meta analysis": "meta_analysis",
    "randomized controlled trial": "rct",
    "randomised controlled trial": "rct",
    "controlled trial": "rct",
    "case report": "case_report",
    "case series": "case_report",
    "cohort": "observational_study",
    "retrospective": "observational_study",
    "prospective": "observational_study",
    "cross-sectional": "observational_study",
    "cross sectional": "observational_study",
    "observational": "observational_study",
    "survey": "observational_study",
}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def infer_article_type(title: str, publication_types: List[str]) -> str:
    """
    Priority:
    1. explicit publication types from XML
    2. title-based inference
    3. unknown
    """
    publication_blob = _normalize(" ".join(publication_types))
    title_blob = _normalize(title)

    for key, mapped in EBM_TYPE_MAP.items():
        if key in publication_blob:
            return mapped

    for key, mapped in EBM_TYPE_MAP.items():
        pattern = rf"\b{re.escape(key)}\b"
        if re.search(pattern, title_blob):
            return mapped

    return "unknown"