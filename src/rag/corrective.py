"""상위 법령 재검색 (Corrective RAG).

규정만으로 먼저 답한 뒤, 그 답이 법령에 넘기고 있는지 확인한다.
  1. 답변 본문, 그리고 답변이 출처로 인용한 규정 조문에서 법령 이름을 찾는다.
     - 답변에 나온 법령은 그대로 후보로 쓴다.
     - 인용 조문에 나온 법령은 "준용", "따른다", "정하는 대로"처럼 맡기는 표현이 붙은 것만 쓴다.
  2. 후보가 있으면 법령 API로 조문을 조회한다. 조문번호가 없으면 LLM이 법령 목차를 보고 고른다.
  3. 조회한 조문이 다시 다른 법령에 맡기면(근로기준법 제34조 → 근로자퇴직급여 보장법) 한 번 더 따라간다.

"근거가 충분한가"를 LLM이 미리 판단하던 방식은 판단이 흔들려서(비슷한 숫자에 혼동 등) 답변에 드러난 신호로 바꿨다.
법령 이름은 정규식 후보 + 법령 API 이름 정확 일치로만 정해서 LLM이 지어낼 수 없다.
"""
import re

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.lib.config import OPENAI_PLANNER_MODEL
from src.rag import law

MAX_HOPS = 2  # 법령 → 법령으로 따라가는 최대 횟수. 무한 반복을 막는 종료 조건
MAX_LAWS_PER_HOP = 3
MAX_ARTICLES_PER_LAW = 2
DELEGATION_RE = re.compile(r"준용|따른다|따라|정하는 대로|정한 바에|적용한다|적용하며")
DELEGATION_WINDOW = 60  # 법령 이름 뒤 몇 글자 안에 맡기는 표현이 있어야 하는가


class ArticleChoice(BaseModel):
    article_nos: list[str] = Field(description="목차에서 고른 조문번호. 예: ['제74조']. 관련 조문이 없으면 빈 목록")


CHOOSE_PROMPT = """질문에 답하는 데 필요한 조문을 아래 법령 목차에서 최대 {k}개 고른다. 목차에 있는 조문번호만 쓴다.

[질문]
{question}

[참고: 이 법령이 언급된 문맥]
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


def context_of(name: str, text: str, width: int = 80) -> str:
    i = text.find(name)
    return text[max(0, i - width): i + len(name) + width].replace("\n", " ") if i >= 0 else ""


def delegated_refs(text: str) -> list[tuple[str, str]]:
    """텍스트에서 '맡기는 표현'이 뒤따르는 법령만 고른다. 지나가듯 언급한 법령은 뺀다."""
    refs = []
    for name, article in law.extract_law_refs(text):
        for m in re.finditer(re.escape(name), text):
            if DELEGATION_RE.search(text[m.end(): m.end() + DELEGATION_WINDOW]):
                refs.append((name, article))
                break
    return refs


def cited_chunks(answer: str, docs: list[Document]) -> list[Document]:
    """답변이 출처로 인용한 규정 조문. 규정명과 조문번호가 모두 답변에 나온 청크."""
    flat = re.sub(r"\s", "", answer)
    return [
        d for d in docs
        if d.metadata.get("article_no")
        and re.sub(r"\s", "", d.metadata["rule_name"]) in flat
        and d.metadata["article_no"] in flat
    ]


def initial_candidates(answer: str, docs: list[Document]) -> list[tuple[str, str]]:
    candidates = law.extract_law_refs(answer)
    for d in cited_chunks(answer, docs):
        for ref in delegated_refs(d.page_content):
            if ref not in candidates:
                candidates.append(ref)
    return candidates


def lookup(question: str, name: str, article_no: str, hint: str, chooser) -> list[law.Article]:
    if article_no:
        article = law.get_article(name, article_no)
        return [article] if article else []
    choice = chooser.invoke(CHOOSE_PROMPT.format(
        question=question, hint=hint, law_name=name, toc=law.table_of_contents(name), k=MAX_ARTICLES_PER_LAW,
    ))
    articles = [law.get_article(name, no) for no in choice.article_nos[:MAX_ARTICLES_PER_LAW]]
    return [a for a in articles if a]


def gather_laws(question: str, answer: str, docs: list[Document]) -> tuple[list[Document], list[str]]:
    """규정만으로 만든 답변을 보고 필요한 법령 조문을 모은다. (법령 Document 목록, 과정 기록)을 돌려준다."""
    candidates = initial_candidates(answer, docs)
    if not candidates:
        return [], ["법령 후보 없음: 규정 답변 그대로 사용"]

    chooser = ChatOpenAI(model=OPENAI_PLANNER_MODEL, temperature=0).with_structured_output(ArticleChoice)
    hint_text = answer + "\n" + "\n".join(d.page_content for d in cited_chunks(answer, docs))
    law_docs: list[Document] = []
    seen_laws: set[tuple[str, str]] = set()
    seen_articles: set[tuple[str, str]] = set()
    trace: list[str] = []

    for hop in range(1, MAX_HOPS + 1):
        todo = [c for c in candidates if c not in seen_laws][:MAX_LAWS_PER_HOP]
        if not todo:
            break
        trace.append(f"{hop}차 후보: {todo}")
        new_docs = []
        for name, article_no in todo:
            seen_laws.add((name, article_no))
            for a in lookup(question, name, article_no, context_of(name, hint_text), chooser):
                if (a.law_name, a.article_no) in seen_articles:
                    continue
                seen_articles.add((a.law_name, a.article_no))
                new_docs.append(to_document(a))
                trace.append(f"  조회: {a.law_name} {a.article_no}({a.title})")
        if not new_docs:
            break
        law_docs += new_docs
        # 조회한 조문이 다시 다른 법령에 맡기면 다음 단계 후보로
        fetched = "\n".join(d.page_content for d in new_docs)
        hint_text += "\n" + fetched
        candidates = delegated_refs(fetched)
    return law_docs, trace
