"""상위 법령 재검색 (Corrective RAG).

규정 검색 결과를 보고 판단한다.
  1. 발췌에 질문이 묻는 구체 내용(일수·금액·범위·대상)이 직접 있는가?
  2. 없고, 그 내용을 법령에 맡기고 있으면("근로기준법에 따른다") 그 법령 조문을 API로 조회해 근거에 추가한다.
조회한 법령이 다시 다른 법령에 맡기는 경우가 있어서 (근로기준법 제34조 → 근로자퇴직급여 보장법) 최대 MAX_ROUNDS번 반복한다.

LLM이 법령 이름을 지어내지 않도록, 조회 대상은 발췌·법령 본문에서 정규식으로 뽑은 후보 중에서만 고르게 한다.
후보는 법령 API에서 이름이 정확히 일치하는 법령만 남긴다.
"""
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.lib.config import OPENAI_PLANNER_MODEL
from src.rag import law

MAX_ROUNDS = 2  # 판단·조회 반복 횟수. 무한 반복을 막는 종료 조건
MAX_LOOKUPS_PER_ROUND = 2
MAX_ARTICLES_PER_LAW = 2


class Plan(BaseModel):
    sufficient: bool = Field(description="질문이 묻는 구체 내용이 지금까지의 발췌에 직접 적혀 있으면 true")
    candidate_ids: list[int] = Field(default_factory=list, description="조회할 후보 번호. 필요 없으면 빈 목록")
    reason: str = Field(description="판단 이유 한 문장")


class ArticleChoice(BaseModel):
    article_nos: list[str] = Field(description="목차에서 고른 조문번호. 예: ['제74조']. 관련 조문이 없으면 빈 목록")


# 예시는 평가셋(eval/questions.csv) 문항과 겹치지 않게 일반적인 형태로만 둔다. 겹치면 정답을 알려주는 셈이다
PLAN_PROMPT = """너는 사내 규정 QA 시스템의 검색 판단기다.

[질문]
{question}

[규정 발췌]
{regulation}

[이미 조회한 법령]
{laws}

[조회 후보] (발췌·법령 본문에 이름이 나온 법령)
{candidates}

판단 기준:
- 질문이 묻는 구체 내용(일수·금액·범위·대상)이 발췌에 직접 적혀 있으면 sufficient=true, candidate_ids는 비운다.
- "○○법에 따른다", "○○법에서 정한 기준을 적용한다", "○○법 시행령 제N조를 준용한다"는 구체 내용이 아니다. 질문이 묻는 내용이 이렇게 법령으로 넘어가 있으면 sufficient=false로 하고, 그 법령의 후보 번호를 candidate_ids에 넣는다.
- 이미 조회한 법령 조문이 다시 「다른 법령」이 정하는 대로 따른다고 하면, 그 다른 법령도 고른다.
- 예:
  질문 "연장근로 수당은 얼마?" + 발췌 "가산임금은 근로기준법에 따른다" → 근로기준법 후보
  질문 "보상 대상은 누구?" + 발췌 "대상의 범위는 ○○법 시행령 제N조를 준용한다" → 그 시행령 제N조 후보
- 질문과 관련 없는 후보는 고르지 않는다. 조회 후보에 없는 법령은 고를 수 없다."""

CHOOSE_PROMPT = """질문에 답하는 데 필요한 조문을 아래 법령 목차에서 최대 {k}개 고른다. 목차에 있는 조문번호만 쓴다.

[질문]
{question}

[참고: 규정이 이 법령을 인용한 문맥]
{hint}

[{law_name} 목차]
{toc}"""


def to_document(a: law.Article) -> Document:
    return Document(
        page_content=f"[{a.law_name} {a.article_no}({a.title})]\n{a.text}",
        metadata={"source_type": "law", "rule_name": a.law_name, "article_no": a.article_no,
                  "effective_date": a.effective_date},
    )


def format_laws(docs: list[Document]) -> str:
    return "\n\n".join(d.page_content for d in docs) or "(없음)"


def hint_for(name: str, text: str, width: int = 80) -> str:
    """후보 법령이 언급된 주변 문장. 조문을 고를 때 참고로 준다."""
    i = text.find(name)
    return text[max(0, i - width): i + len(name) + width].replace("\n", " ") if i >= 0 else ""


def gather_laws(question: str, regulation_context: str) -> tuple[list[Document], list[str]]:
    """필요한 법령 조문을 모은다. (법령 Document 목록, 판단 과정 기록)을 돌려준다."""
    llm = ChatOpenAI(model=OPENAI_PLANNER_MODEL, temperature=0)
    planner = llm.with_structured_output(Plan)
    chooser = llm.with_structured_output(ArticleChoice)

    docs: list[Document] = []
    seen: set[tuple[str, str]] = set()
    trace: list[str] = []
    for round_no in range(1, MAX_ROUNDS + 1):
        source_text = regulation_context + "\n" + format_laws(docs)
        candidates = [c for c in law.extract_law_refs(source_text) if c not in seen]
        if not candidates:
            trace.append(f"{round_no}차: 조회할 법령 후보 없음")
            break
        listing = "\n".join(f"{i}. {name} {art}".strip() for i, (name, art) in enumerate(candidates))
        plan = planner.invoke(PLAN_PROMPT.format(
            question=question, regulation=regulation_context, laws=format_laws(docs), candidates=listing,
        ))
        picked = [candidates[i] for i in plan.candidate_ids if 0 <= i < len(candidates)][:MAX_LOOKUPS_PER_ROUND]
        trace.append(f"{round_no}차 판단: sufficient={plan.sufficient}, 후보 {len(candidates)}개 중 선택 {picked} ({plan.reason})")
        # 판단(sufficient)과 선택이 어긋나면 선택을 따른다. 고른 게 없을 때만 멈춘다
        if not picked:
            break

        added = 0
        for name, article_no in picked:
            seen.add((name, article_no))
            if article_no:
                article_nos = [article_no]
            else:
                choice = chooser.invoke(CHOOSE_PROMPT.format(
                    question=question, hint=hint_for(name, source_text), law_name=name,
                    toc=law.table_of_contents(name), k=MAX_ARTICLES_PER_LAW,
                ))
                article_nos = choice.article_nos[:MAX_ARTICLES_PER_LAW]
            for no in article_nos:
                article = law.get_article(name, no)
                if not article or (name, article.article_no) in seen:
                    continue
                seen.add((name, article.article_no))
                docs.append(to_document(article))
                added += 1
                trace.append(f"  조회: {article.law_name} {article.article_no}({article.title})")
        if not added:  # 새로 얻은 조문이 없으면 더 반복해도 같은 판단이 나온다
            break
    return docs, trace
