"""RAG: 검색 → Prompt → LLM → (답변이 법령에 넘기면 법령 조회 후 다시 답변). 답변과 출처(규정명·페이지, 법령 조문)를 함께 돌려준다.

실행: uv run python -m src.rag.rag "출산휴가는 며칠 쓸 수 있어?" --pipeline structured_hybrid_law
"""
import argparse

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.lib.config import OPENAI_CHAT_MODEL
from src.rag.corrective import format_laws, gather_laws
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

# 법령 재검색을 쓰는 파이프라인 전용. 규칙 1·2·4만 법령까지 넓히고 나머지는 같다.
LAW_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "너는 {org} 직원의 사내 규정 질문에 답하는 어시스턴트다.\n"
     "규칙:\n"
     "1. 아래 [규정 발췌]와 [관련 법령]에 있는 내용만 근거로 답한다. 일반 상식이나 추측으로 보충하지 않는다.\n"
     "2. 규정이 '법령에 따른 기준', '법령에서 정한 일수'처럼 법령을 가리키면, [관련 법령]에서 그 기준의 구체적인 내용(일수·금액·범위·대상)을 찾아 답변에 반드시 적는다. '법령에 따른다'로만 끝내지 않는다.\n"
     "3. 답변 끝에 근거를 '(규정명 제N조, p.페이지)', 법령은 '(법령명 제N조)' 형식으로 적는다.\n"
     "4. 금액·기간·조건·예외(다만 ~)가 있으면 빠뜨리지 않는다.\n"
     f"5. 발췌와 법령 모두에 근거가 없으면 정확히 이렇게만 답한다: {NO_ANSWER}"),
    ("human", "[규정 발췌]\n{context}\n\n[관련 법령]\n{laws}\n\n[질문]\n{question}"),
])


def format_context(docs: list[Document]) -> str:
    return "\n\n".join(f"<{source_label(d)}>\n{d.page_content}" for d in docs)


def source_label(d: Document) -> str:
    """구조 기반 청크는 조문번호와 걸친 페이지까지, 법령은 법령명과 조문번호를 보여준다."""
    m = d.metadata
    if m.get("source_type") == "law":
        return f"{m['rule_name']} {m['article_no']}"
    pages = m.get("pages") or [m["page"]]
    page = f"p.{pages[0]}" if len(pages) == 1 else f"p.{pages[0]}-{pages[-1]}"
    article = f" {m['article_no']}" if m.get("article_no") else ""
    return f"{m['rule_name']}{article} {page}"


def ask(question: str, org: str, k: int = TOP_K, pipeline: str = DEFAULT_PIPELINE) -> dict:
    """RAG 진입점. 나중에 FastAPI 엔드포인트도 이 함수를 그대로 부른다 (기획서 9-2)."""
    p = get_pipeline(pipeline)
    docs = get_retriever(org, k, method=p["retriever"], index=p["index"]).invoke(question)
    context = format_context(docs)
    llm = ChatOpenAI(model=OPENAI_CHAT_MODEL, temperature=0)

    answer = (PROMPT | llm | StrOutputParser()).invoke({"org": org, "context": context, "question": question})

    law_docs, trace = [], []
    if p.get("law"):
        # 규정 답변이 법령에 넘기고 있으면 법령을 조회해서 다시 답한다. 아니면 규정 답변을 그대로 쓴다
        law_docs, trace = gather_laws(question, answer, docs)
        if law_docs:
            answer = (LAW_PROMPT | llm | StrOutputParser()).invoke(
                {"org": org, "context": context, "laws": format_laws(law_docs), "question": question})
    return {
        "question": question,
        "answer": answer,
        "sources": [source_label(d) for d in docs + law_docs],
        "docs": docs,  # 규정 검색 결과 (검색 hit 평가 대상)
        "law_docs": law_docs,  # 재검색으로 가져온 법령 조문
        "trace": trace,  # 법령 재검색 판단 과정
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="?", default="출장 일비는 얼마인가?")
    parser.add_argument("--pipeline", choices=PIPELINES, default=DEFAULT_PIPELINE)
    args = parser.parse_args()
    result = ask(args.question, DEFAULT_ORG, pipeline=args.pipeline)
    print("=== 검색 문서")
    print_docs(result["docs"])
    if result["trace"]:
        print("\n=== 법령 재검색")
        print("\n".join(result["trace"]))
    print("\n=== 답변")
    print(result["answer"])
