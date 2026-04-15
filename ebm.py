EBM_RANK = {
    "clinical_practice_guideline": 1,
    "systematic_review": 2,
    "meta_analysis": 3,
    "rct": 4,
    "observational_study": 5,
    "case_report": 6,
    "unknown": 99,
}


def get_ebm_rank(article_type: str) -> int:
    return EBM_RANK.get(article_type, 99)


def choose_best_doc_by_ebm(docs):
    if not docs:
        return None
    return sorted(
        docs,
        key=lambda d: (
            get_ebm_rank(d.metadata.get("article_type", "unknown")),
            -(d.metadata.get("year") or 0),
        ),
    )[0]