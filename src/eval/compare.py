"""두 평가 결과를 문항별로 비교해 results/evaluation.md 를 만든다 (교재 14.7).

실행: uv run python -m src.eval.compare baseline hybrid
"""
import argparse
import csv
from pathlib import Path

from src.eval.evaluate import NO_DOC_TYPE, RESULTS_DIR

VERDICT_SCORE = {"오답": 0, "부분정답": 1, "정답": 2}


def load(run: str) -> dict[str, dict]:
    with (RESULTS_DIR / run / "details.csv").open(encoding="utf-8-sig") as f:
        return {r["id"]: r for r in csv.DictReader(f)}


def change(before: dict, after: dict) -> str:
    diff = VERDICT_SCORE[after["judge"]] - VERDICT_SCORE[before["judge"]]
    return "좋아짐" if diff > 0 else "나빠짐" if diff < 0 else "동일"


def hit_mark(row: dict) -> str:
    if row["question_type"] == NO_DOC_TYPE:
        return "-"
    return f"O({row['hit_rank']}위)" if row["hit"] == "True" else "X"


def metrics(rows: dict[str, dict]) -> dict[str, str]:
    answerable = [r for r in rows.values() if r["question_type"] != NO_DOC_TYPE]
    n = len(answerable)
    hits = [r for r in answerable if r["hit"] == "True"]
    return {
        "Hit Rate": f"{len(hits) / n:.0%}",
        "MRR": f"{sum(1 / int(r['hit_rank']) for r in hits) / n:.2f}",
        "key_facts 통과": f"{sum(r['key_facts_ok'] == 'True' for r in answerable) / n:.0%}",
        "LLM 채점 정답": f"{sum(r['judge'] == '정답' for r in rows.values()) / len(rows):.0%}",
        "잘못된 거절": f"{sum(r['refused'] == 'True' for r in answerable) / n:.0%}",
    }


def build(before_run: str, after_run: str) -> str:
    before, after = load(before_run), load(after_run)
    mb, ma = metrics(before), metrics(after)

    lines = [
        f"# {before_run} vs {after_run}",
        "",
        "## 지표",
        "",
        f"| 지표 | {before_run} | {after_run} |",
        "|---|---|---|",
        *[f"| {k} | {mb[k]} | {ma[k]} |" for k in mb],
        "",
        "## 유형별 검색 hit",
        "",
        f"| 유형 | {before_run} | {after_run} |",
        "|---|---|---|",
    ]
    types = dict.fromkeys(r["question_type"] for r in before.values() if r["question_type"] != NO_DOC_TYPE)
    for t in types:
        b = [r for r in before.values() if r["question_type"] == t]
        a = [r for r in after.values() if r["question_type"] == t]
        lines.append(f"| {t} | {sum(r['hit'] == 'True' for r in b)}/{len(b)} | {sum(r['hit'] == 'True' for r in a)}/{len(a)} |")

    lines += [
        "",
        "## 문항별 비교",
        "",
        "검색 hit의 순위는 정답 페이지가 몇 번째로 검색됐는지다. 판단은 LLM 채점(정답 > 부분정답 > 오답) 기준이다.",
        "",
        f"| ID | 유형 | 질문 | {before_run} 검색 | {after_run} 검색 | 답변 변화 | 판단 |",
        "|---|---|---|---|---|---|---|",
    ]
    for qid, b in before.items():
        a = after[qid]
        lines.append(
            f"| {qid} | {b['question_type']} | {b['question']} | {hit_mark(b)} | {hit_mark(a)} "
            f"| {b['judge']} → {a['judge']} | {change(b, a)} |"
        )

    counts = {c: sum(change(before[q], after[q]) == c for q in before) for c in ("좋아짐", "동일", "나빠짐")}
    lines += ["", f"좋아짐 {counts['좋아짐']} / 동일 {counts['동일']} / 나빠짐 {counts['나빠짐']}"]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("before")
    parser.add_argument("after")
    args = parser.parse_args()
    out = Path(RESULTS_DIR) / "evaluation.md"
    out.write_text(build(args.before, args.after), encoding="utf-8")
    print(f"저장: {out}")
