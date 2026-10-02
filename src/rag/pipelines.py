"""RAG 파이프라인 조합 (Modular RAG).

각 파이프라인은 단계별 모듈을 고른다.
  index     : 청킹 방식이자 벡터 컬렉션 이름 (baseline / structured)
  retriever : 검색 방식 (dense / hybrid)

새 개선은 모듈을 추가하고 여기에 조합을 하나 더 등록한다. 평가는 조합 이름으로 실행한다.
"""

PIPELINES = {
    "baseline":          {"index": "baseline",   "retriever": "dense"},
    "hybrid":            {"index": "baseline",   "retriever": "hybrid"},
    "structured":        {"index": "structured", "retriever": "dense"},
    "structured_hybrid": {"index": "structured", "retriever": "hybrid"},
}
DEFAULT_PIPELINE = "baseline"


def get_pipeline(name: str) -> dict:
    if name not in PIPELINES:
        raise ValueError(f"알 수 없는 파이프라인: {name} (가능: {list(PIPELINES)})")
    return PIPELINES[name]
