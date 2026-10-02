"""평가셋(eval/questions.csv)으로 RAG를 측정하고 results/<run>/ 에 기록한다.

채점은 세 겹이다.
  ① 검색: 정답 규정·페이지가 Top-K 안에 있는가 (hit, 순위)
  ② key_facts: 답변에 꼭 들어가야 할 값이 모두 있는가
  ③ LLM 채점: 정답 기준과 의미가 맞는가, 검색 문서에 근거하는가
문서없음 문항은 ①②에서 빼고, 답변을 거절했는지로 판정한다 (교재 14.8).

실행: uv run python -m src.eval.evaluate --run baseline --method dense
      uv run python -m src.eval.evaluate --run hybrid --method hybrid
"""
import argparse
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Literal

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.lib.config import OPENAI_CHAT_MODEL, OPENAI_EMBEDDING_MODEL, OPENAI_JUDGE_MODEL
from src.lib.store import COLLECTION
from src.rag.ingest import CHUNK_OVERLAP, CHUNK_SIZE
from src.rag.rag import NO_ANSWER, ask
from src.rag.retrieve import METHODS, TOP_K

QUESTIONS = Path("eval/questions.csv")
RESULTS_DIR = Path("results")
NO_DOC_TYPE = "문서없음"

DETAIL_FIELDS = [
    "id", "question_type", "question", "target", "retrieved", "hit", "hit_rank", "other_org",
    "answer", "key_facts", "key_facts_ok", "refused", "judge", "grounded", "judge_reason",
]


# ---------- ① 검색 ----------

def target_pages(row: dict) -> set[tuple[str, int]]:
    rule = Path(row["target_file_name"]).stem
    return {(rule, int(p)) for p in row["target_page_no"].split(";") if p}


def hit_rank(docs, targets: set[tuple[str, int]]) -> int | None:
    """정답 페이지가 처음 나온 순위 (1부터). 없으면 None."""
    for rank, d in enumerate(docs, 1):
        if (d.metadata["rule_name"], d.metadata["page"]) in targets:
            return rank
    return None


# ---------- ② key_facts ----------

def normalize(text: str) -> str:
    return re.sub(r"[\s,]", "", text).lower()


def key_facts_ok(answer: str, key_facts: str) -> bool:
    """'|'로 나눈 값이 모두 있어야 통과. 한 값 안의 ' or '는 그중 하나만 있으면 된다."""
    ans = normalize(answer)
    return all(
        any(normalize(alt) in ans for alt in fact.split(" or "))
        for fact in key_facts.split("|") if fact
    )


def is_refusal(answer: str) -> bool:
    return NO_ANSWER in answer or "찾을 수 없" in answer


# ---------- ③ LLM 채점 ----------

class Judgement(BaseModel):
    verdict: Literal["정답", "부분정답", "오답"] = Field(description="정답 기준 대비 답변의 정확성")
    grounded: bool = Field(description="답변 내용이 모두 검색 문서에 근거하는가")
    reason: str = Field(description="판정 이유 한두 문장")


JUDGE_PROMPT = """너는 사내 규정 QA 시스템의 답변을 채점한다.

[질문]
{question}

[정답 기준]
{target_answer}

[검색 문서]
{context}

[시스템 답변]
{answer}

채점 기준:
- verdict: 정답 기준의 핵심(금액·기간·조건·예외)이 모두 맞으면 "정답", 일부만 맞거나 중요한 조건이 빠지면 "부분정답", 틀리거나 답하지 못하면 "오답". 표현이 달라도 의미가 같으면 맞은 것으로 본다.
- 정답 기준이 "규정에서 찾을 수 없음"이면, 답변이 근거 없음을 밝히고 추측하지 않았을 때만 "정답"이다.
- 정답 기준에 답이 있는데 시스템이 "찾을 수 없음"으로 거절만 했다면 무조건 "오답"이다.
- grounded: 답변이 검색 문서에 없는 내용을 지어냈는지만 본다. 지어냈으면 false, 아니면 true. 거절 답변은 지어낸 내용이 없으므로 항상 true다 (정답 여부는 verdict에서만 따진다)."""


def make_judge():
    llm = ChatOpenAI(model=OPENAI_JUDGE_MODEL, temperature=0)
    return llm.with_structured_output(Judgement)


# ---------- 실행 ----------

def evaluate_one(row: dict, judge, k: int, method: str) -> dict:
    result = ask(row["question"], row["org"], k=k, method=method)
    docs, answer = result["docs"], result["answer"]
    no_doc = row["question_type"] == NO_DOC_TYPE

    rank = None if no_doc else hit_rank(docs, target_pages(row))
    context = "\n\n".join(f"<{d.metadata['rule_name']} p.{d.metadata['page']}>\n{d.page_content}" for d in docs)
    j = judge.invoke(JUDGE_PROMPT.format(
        question=row["question"], target_answer=row["target_answer"], context=context, answer=answer,
    ))
    return {
        "id": row["id"],
        "question_type": row["question_type"],
        "question": row["question"],
        "target": "" if no_doc else f"{Path(row['target_file_name']).stem} p.{row['target_page_no']}",
        "retrieved": " / ".join(f"{d.metadata['rule_name']} p.{d.metadata['page']}" for d in docs),
        "hit": "" if no_doc else rank is not None,
        "hit_rank": rank or "",
        "other_org": sum(d.metadata["org"] != row["org"] for d in docs),
        "answer": answer,
        "key_facts": row["key_facts"],
        "key_facts_ok": "" if no_doc else key_facts_ok(answer, row["key_facts"]),
        "refused": is_refusal(answer),
        "judge": j.verdict,
        "grounded": j.grounded,
        "judge_reason": j.reason,
    }


def pct(n: int, d: int) -> str:
    return f"{n / d:.0%} ({n}/{d})" if d else "-"


def summarize(run: str, k: int, method: str, rows: list[dict]) -> str:
    answerable = [r for r in rows if r["question_type"] != NO_DOC_TYPE]
    no_doc = [r for r in rows if r["question_type"] == NO_DOC_TYPE]
    hits = [r for r in answerable if r["hit"]]
    mrr = sum(1 / r["hit_rank"] for r in hits) / len(answerable) if answerable else 0
    verdicts = Counter(r["judge"] for r in rows)

    lines = [
        f"# 평가 결과: {run}",
        "",
        f"- 설정: 검색 `{method}`, chunk {CHUNK_SIZE}/{CHUNK_OVERLAP}, Top-K {k}, 컬렉션 `{COLLECTION}`",
        f"- 모델: 답변 `{OPENAI_CHAT_MODEL}`, 임베딩 `{OPENAI_EMBEDDING_MODEL}`, 채점 `{OPENAI_JUDGE_MODEL}`",
        f"- 문항: {len(rows)}개 (답변 가능 {len(answerable)}, 문서없음 {len(no_doc)})",
        "",
        "## 전체",
        "",
        "| 구분 | 지표 | 값 |",
        "|---|---|---|",
        f"| 검색 | Hit Rate@{k} | {pct(len(hits), len(answerable))} |",
        f"| 검색 | MRR | {mrr:.2f} |",
        f"| 검색 | 다른 기관 문서 혼입 | {sum(r['other_org'] for r in rows)}건 |",
        f"| 답변 | key_facts 통과 | {pct(sum(bool(r['key_facts_ok']) for r in answerable), len(answerable))} |",
        f"| 답변 | LLM 채점 정답 | {pct(verdicts['정답'], len(rows))} (부분정답 {verdicts['부분정답']}, 오답 {verdicts['오답']}) |",
        f"| 답변 | 근거 충실 (grounded) | {pct(sum(r['grounded'] for r in rows), len(rows))} |",
        f"| 거절 | 문서없음 문항 거절 | {pct(sum(r['refused'] for r in no_doc), len(no_doc))} |",
        f"| 거절 | 답변 가능 문항을 잘못 거절 | {pct(sum(r['refused'] for r in answerable), len(answerable))} |",
        "",
        "## 유형별",
        "",
        f"| 유형 | 문항 | Hit@{k} | key_facts | LLM 정답 |",
        "|---|---|---|---|---|",
    ]
    by_type = defaultdict(list)
    for r in rows:
        by_type[r["question_type"]].append(r)
    for qtype, rs in by_type.items():
        if qtype == NO_DOC_TYPE:
            hit_s = kf_s = "-"
        else:
            hit_s = pct(sum(bool(r["hit"]) for r in rs), len(rs))
            kf_s = pct(sum(bool(r["key_facts_ok"]) for r in rs), len(rs))
        lines.append(f"| {qtype} | {len(rs)} | {hit_s} | {kf_s} | {pct(sum(r['judge'] == '정답' for r in rs), len(rs))} |")

    failed = [r for r in rows if r["judge"] != "정답" or r["hit"] is False]
    lines += ["", "## 실패·부분정답 문항", "",
              "| ID | 유형 | 질문 | 검색 hit | LLM 채점 | 이유 |", "|---|---|---|---|---|---|"]
    for r in failed:
        hit_s = "-" if r["hit"] == "" else ("O" if r["hit"] else "X")
        lines.append(f"| {r['id']} | {r['question_type']} | {r['question']} | {hit_s} | {r['judge']} | {r['judge_reason']} |")
    lines += ["", "문제점 분석은 같은 폴더의 `analysis.md`에 직접 작성한다 (재실행해도 덮어쓰지 않음)."]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="결과 폴더 이름 (예: baseline, hybrid)")
    parser.add_argument("--k", type=int, default=TOP_K)
    parser.add_argument("--method", choices=METHODS, required=True, help="검색 방식")
    args = parser.parse_args()

    with QUESTIONS.open(encoding="utf-8-sig") as f:
        questions = list(csv.DictReader(f))

    judge = make_judge()
    rows = []
    for q in questions:
        r = evaluate_one(q, judge, args.k, args.method)
        mark = "-" if r["hit"] == "" else ("O" if r["hit"] else "X")
        print(f"{r['id']} [{r['question_type']}] hit={mark} key={r['key_facts_ok']} judge={r['judge']}")
        rows.append(r)

    out = RESULTS_DIR / args.run
    out.mkdir(parents=True, exist_ok=True)
    with (out / "details.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=DETAIL_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    (out / "summary.md").write_text(summarize(args.run, args.k, args.method, rows), encoding="utf-8")
    print(f"\n저장: {out}/details.csv, {out}/summary.md")


if __name__ == "__main__":
    main()
