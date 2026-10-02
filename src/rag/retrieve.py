"""검색기. 모든 검색은 get_retriever(org)를 거쳐서 기관 필터가 빠지지 않게 한다 (기획서 9-3).

실행: uv run python -m src.rag.retrieve "출장 일비는 얼마인가?"
"""
import sys

from langchain_core.vectorstores import VectorStoreRetriever

from src.lib.store import get_vector_store

DEFAULT_ORG = "한국지역난방공사"
TOP_K = 4


def get_retriever(org: str, k: int = TOP_K) -> VectorStoreRetriever:
    if not org:
        raise ValueError("org는 필수입니다")
    return get_vector_store().as_retriever(
        search_kwargs={"k": k, "filter": {"org": {"$eq": org}}},
    )


def print_docs(docs) -> None:
    for i, d in enumerate(docs, 1):
        m = d.metadata
        preview = d.page_content[:150].replace("\n", " ")
        print(f"[{i}] {m['rule_name']} p.{m['page']} (chunk {m['chunk_id']})\n    {preview}…")


if __name__ == "__main__":
    question = sys.argv[1] if len(sys.argv) > 1 else "출장 일비는 얼마인가?"
    print_docs(get_retriever(DEFAULT_ORG).invoke(question))
