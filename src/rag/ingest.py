"""적재: 최신본 PDF → 페이지 단위 Document → 청킹 → PGVector.

청킹 방식마다 컬렉션을 따로 만든다. 그래서 Baseline과 개선 결과를 같은 DB에서 나란히 비교할 수 있다.

실행: uv run python -m src.rag.ingest --index baseline
      uv run python -m src.rag.ingest --index structured
"""
import argparse
import csv
from collections import Counter
from pathlib import Path

import pymupdf
from langchain_core.documents import Document

from src.collect.collect_rules import MANIFEST, safe_name
from src.collect.convert_latest import PDF_DIR
from src.lib.store import get_vector_store
from src.rag.chunking import baseline_chunks, structured_chunks

INDEXES = ("baseline", "structured")


def load_pdf(pdf: Path, meta: dict) -> list[Document]:
    """PDF 한 페이지를 Document 하나로 만든다. page는 1부터 센다."""
    with pymupdf.open(pdf) as doc:
        return [
            Document(page_content=text, metadata={**meta, "page": i})
            for i, page in enumerate(doc, 1)
            if (text := page.get_text().strip())
        ]


def load_latest_rules() -> list[list[Document]]:
    """규정별 페이지 목록. 구조 기반 청킹은 규정 하나를 통째로 다뤄야 해서 규정 단위로 묶는다."""
    with MANIFEST.open(encoding="utf-8-sig") as f:
        latest = [r for r in csv.DictReader(f) if r["is_latest"] == "True"]

    rules = []
    for row in latest:
        pdf = PDF_DIR / safe_name(row["org"]) / f"{safe_name(row['rule_name'])}.pdf"
        meta = {
            "org": row["org"],
            "rule_name": row["rule_name"].strip(),
            "category": row["category"],
            "rule_date": row["rule_date"],
            "source": str(pdf),
        }
        rules.append(load_pdf(pdf, meta))
    return rules


def build_chunks(index: str) -> list[Document]:
    rules = load_latest_rules()
    if index == "baseline":
        chunks = baseline_chunks([page for pages in rules for page in pages])
    else:
        chunks = [c for pages in rules for c in structured_chunks(pages)]
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", choices=INDEXES, required=True)
    args = parser.parse_args()

    chunks = build_chunks(args.index)
    print(f"[{args.index}] chunk {len(chunks)}개, 평균 {sum(len(c.page_content) for c in chunks) // len(chunks)}자")
    if args.index == "structured":
        print("섹션 종류:", dict(Counter(c.metadata["section_type"] for c in chunks)))
    print("예시 metadata:", chunks[0].metadata)

    # 다시 실행해도 중복되지 않도록 컬렉션을 새로 만든다
    store = get_vector_store(args.index, pre_delete_collection=True)
    store.add_documents(chunks, ids=[f"{args.index}-{c.metadata['org']}-{c.metadata['chunk_id']}" for c in chunks])
    print(f"적재 완료: 컬렉션 `{args.index}`")


if __name__ == "__main__":
    main()
