"""RAG: 검색 → Prompt → LLM. 답변과 출처(규정명·페이지)를 함께 돌려준다.

실행: uv run python -m src.rag.rag "출장 일비는 얼마인가?" --pipeline structured
"""
import argparse

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.lib.config import OPENAI_CHAT_MODEL
from src.rag.pipelines import DEFAULT_PIPELINE, PIPELINES, get_pipeline
from src.rag.retrieve import DEFAULT_ORG, TOP_K, get_retriever, print_docs

NO_ANSWER = "규정에서 근거를 찾을 수 없습니다. 담당 부서에 문의해 주세요."

PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "너는 {org} 직원의 사내 규정 질문에 답하는 어시스턴트다.\n"
     "규칙:\n"
     "1. 아래 [규정 발췌]에 있는 내용만 근거로 답한다. 일반 상식이나 추측으로 보충하지 않는다.\n"
     "2. 답변 끝에 근거를 '(규정명 제N조, p.페이지)' 형식으로 적는다.\n"
     "3. 금액·기간·조건·예외(다만 ~)가 있으면 빠뜨리지 않는다.\n"
     f"4. 발췌에 근거가 없으면 정확히 이렇게만 답한다: {NO_ANSWER}"),
    ("human", "[규정 발췌]\n{context}\n\n[질문]\n{question}"),
])


def format_context(docs: list[Document]) -> str:
    return "\n\n".join(f"<{source_label(d)}>\n{d.page_content}" for d in docs)


def source_label(d: Document) -> str:
    """구조 기반 청크는 조문번호와 걸친 페이지까지 출처에 보여준다."""
    m = d.metadata
    pages = m.get("pages") or [m["page"]]
    page = f"p.{pages[0]}" if len(pages) == 1 else f"p.{pages[0]}-{pages[-1]}"
    article = f" {m['article_no']}" if m.get("article_no") else ""
    return f"{m['rule_name']}{article} {page}"


def ask(question: str, org: str, k: int = TOP_K, pipeline: str = DEFAULT_PIPELINE) -> dict:
    """RAG 진입점. 나중에 FastAPI 엔드포인트도 이 함수를 그대로 부른다 (기획서 9-2)."""
    p = get_pipeline(pipeline)
    docs = get_retriever(org, k, method=p["retriever"], index=p["index"]).invoke(question)
    chain = PROMPT | ChatOpenAI(model=OPENAI_CHAT_MODEL, temperature=0) | StrOutputParser()
    answer = chain.invoke({"org": org, "context": format_context(docs), "question": question})
    return {
        "question": question,
        "answer": answer,
        "sources": [
            {"rule_name": d.metadata["rule_name"], "page": d.metadata["page"],
             "chunk_id": d.metadata["chunk_id"]}
            for d in docs
        ],
        "docs": docs,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="?", default="출장 일비는 얼마인가?")
    parser.add_argument("--pipeline", choices=PIPELINES, default=DEFAULT_PIPELINE)
    args = parser.parse_args()
    result = ask(args.question, DEFAULT_ORG, pipeline=args.pipeline)
    print("=== 검색 문서")
    print_docs(result["docs"])
    print("\n=== 답변")
    print(result["answer"])
