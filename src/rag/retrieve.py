"""검색기. 모든 검색은 get_retriever(org)를 거쳐서 기관 필터가 빠지지 않게 한다 (기획서 9-3).

검색 방식(method)
  dense  : 벡터 검색만 (Baseline)
  hybrid : 벡터 + BM25 키워드 검색을 RRF로 합친다 (개선 1)
index는 청킹 방식이자 컬렉션 이름이다 (baseline / structured).

실행: uv run python -m src.rag.retrieve "여비규정 제13조 내용은?" --method hybrid --index structured
"""
import argparse

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from src.lib.store import get_vector_store
from src.rag.bm25 import get_bm25_index

DEFAULT_ORG = "한국지역난방공사"
TOP_K = 4
METHODS = ("dense", "hybrid")
DEFAULT_METHOD = "dense"
DEFAULT_INDEX = "baseline"

HYBRID_CANDIDATES = 20  # 각 검색기에서 가져올 후보 수
RRF_K = 60  # RRF 상수 (일반적으로 쓰는 값)


def get_dense_retriever(org: str, k: int, index: str) -> BaseRetriever:
    return get_vector_store(index).as_retriever(
        search_kwargs={"k": k, "filter": {"org": {"$eq": org}}},
    )


class HybridRetriever(BaseRetriever):
    """벡터 검색과 BM25 결과를 Reciprocal Rank Fusion으로 합친다.
    점수 = Σ 1 / (RRF_K + 순위). 두 검색기에서 모두 상위면 점수가 높아진다."""

    org: str
    index: str
    k: int = TOP_K

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        dense = get_dense_retriever(self.org, HYBRID_CANDIDATES, self.index).invoke(query)
        sparse = get_bm25_index(self.org, self.index).search(query, HYBRID_CANDIDATES)

        scores: dict[int, float] = {}
        docs: dict[int, Document] = {}
        for results in (dense, sparse):
            for rank, doc in enumerate(results, 1):
                cid = doc.metadata["chunk_id"]
                scores[cid] = scores.get(cid, 0) + 1 / (RRF_K + rank)
                docs[cid] = doc
        top = sorted(scores, key=scores.get, reverse=True)[: self.k]
        return [docs[cid] for cid in top]


def get_retriever(
    org: str, k: int = TOP_K, method: str = DEFAULT_METHOD, index: str = DEFAULT_INDEX
) -> BaseRetriever:
    if not org:
        raise ValueError("org는 필수입니다")
    if method == "dense":
        return get_dense_retriever(org, k, index)
    if method == "hybrid":
        return HybridRetriever(org=org, index=index, k=k)
    raise ValueError(f"알 수 없는 검색 방식: {method} (가능: {METHODS})")


def print_docs(docs) -> None:
    for i, d in enumerate(docs, 1):
        m = d.metadata
        preview = d.page_content[:150].replace("\n", " ")
        label = f" {m['section_label']}" if m.get("section_label") else ""
        print(f"[{i}] {m['rule_name']}{label} p.{m['page']} (chunk {m['chunk_id']})\n    {preview}…")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="?", default="출장 일비는 얼마인가?")
    parser.add_argument("--method", choices=METHODS, default=DEFAULT_METHOD)
    parser.add_argument("--index", default=DEFAULT_INDEX)
    args = parser.parse_args()
    print_docs(get_retriever(DEFAULT_ORG, method=args.method, index=args.index).invoke(args.question))
