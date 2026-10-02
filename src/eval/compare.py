"""여러 평가 결과를 비교해 results/evaluation.md 를 만든다 (교재 14.7).
첫 번째 실행을 기준(Baseline)으로 삼아 나머지의 문항별 변화를 표시한다.

실행: uv run python -m src.eval.compare baseline hybrid structured structured_hybrid
"""
import argparse
import csv

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
    return f"O({row['hit_rank']})" if row["hit"] == "True" else "X"


def pct(n: int, d: int) -> str:
    return f"{n / d:.0%}" if d else "-"


def metrics(rows: dict[str, dict]) -> dict[str, str]:
    all_rows = list(rows.values())
    answerable = [r for r in all_rows if r["question_type"] != NO_DOC_TYPE]
    n = len(answerable)
    hits = [r for r in answerable if r["hit"] == "True"]
    cited = [r for r in all_rows if r["citation_ok"] != ""]
    return {
        "검색 Hit Rate@4": pct(len(hits), n),
        "검색 MRR": f"{sum(1 / int(r['hit_rank']) for r in hits) / n:.2f}",
        "key_facts 통과": pct(sum(r["key_facts_ok"] == "True" for r in answerable), n),
        "출처 조문 정확도": pct(sum(r["citation_ok"] == "True" for r in cited), len(cited)),
        "LLM 채점 정답": pct(sum(r["judge"] == "정답" for r in all_rows), len(all_rows)),
        "잘못된 거절": pct(sum(r["refused"] == "True" for r in answerable), n),
    }


def build(runs: list[str]) -> str:
    data = {run: load(run) for run in runs}
    base_run, base = runs[0], data[runs[0]]
    header = "| " + " | ".join(runs) + " |"
    divider = "|---" * len(runs) + "|"

    lines = [f"# 평가 비교: {' / '.join(runs)}", "", f"기준: `{base_run}`", "", "## 지표", "",
             "| 지표 " + header, "|---" + divider]
    per_run = {run: metrics(rows) for run, rows in data.items()}
    for name in per_run[base_run]:
        lines.append(f"| {name} | " + " | ".join(per_run[run][name] for run in runs) + " |")

    lines += ["", "## 유형별 검색 hit", "", "| 유형 " + header, "|---" + divider]
    types = dict.fromkeys(r["question_type"] for r in base.values() if r["question_type"] != NO_DOC_TYPE)
    for t in types:
        cells = []
        for run in runs:
            rs = [r for r in data[run].values() if r["question_type"] == t]
            cells.append(f"{sum(r['hit'] == 'True' for r in rs)}/{len(rs)}")
        lines.append(f"| {t} | " + " | ".join(cells) + " |")

    lines += ["", f"## {base_run} 대비 변화 (LLM 채점 기준)", "", "| 실행 | 좋아짐 | 동일 | 나빠짐 |", "|---|---|---|---|"]
    for run in runs[1:]:
        c = [change(base[q], data[run][q]) for q in base]
        lines.append(f"| {run} | {c.count('좋아짐')} | {c.count('동일')} | {c.count('나빠짐')} |")

    lines += [
        "", "## 문항별 비교", "",
        "검색 칸은 정답 페이지 hit 여부와 순위, 채점 칸은 LLM 채점 결과다.", "",
        "| ID | 유형 | 질문 | " + " | ".join(f"{run} 검색 | {run} 채점" for run in runs) + " |",
        "|---|---|---" + "|---|---" * len(runs) + "|",
    ]
    for qid, b in base.items():
        cells = []
        for run in runs:
            r = data[run][qid]
            mark = "" if run == base_run else f" ({change(b, r)})"
            cells.append(f"{hit_mark(r)} | {r['judge']}{mark}")
        lines.append(f"| {qid} | {b['question_type']} | {b['question']} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", help="비교할 실행 이름. 첫 번째가 기준")
    args = parser.parse_args()
    out = RESULTS_DIR / "evaluation.md"
    out.write_text(build(args.runs), encoding="utf-8")
    print(f"저장: {out}")
