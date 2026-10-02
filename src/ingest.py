"""Baseline 적재: 최신본 PDF → 페이지 단위 Document → RecursiveCharacterTextSplitter → PGVector.

실행: uv run python -m src.ingest
"""
import csv
from pathlib import Path

import pymupdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.collect_rules import MANIFEST, safe_name
from src.convert_latest import PDF_DIR
from src.store import get_vector_store

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def load_pdf(pdf: Path, meta: dict) -> list[Document]:
    """PDF 한 페이지를 Document 하나로 만든다. page는 1부터 센다."""
    with pymupdf.open(pdf) as doc:
        return [
            Document(page_content=text, metadata={**meta, "page": i})
            for i, page in enumerate(doc, 1)
            if (text := page.get_text().strip())
        ]


def load_latest_documents() -> list[Document]:
    with MANIFEST.open(encoding="utf-8-sig") as f:
        latest = [r for r in csv.DictReader(f) if r["is_latest"] == "True"]

    docs = []
    for row in latest:
        pdf = PDF_DIR / safe_name(row["org"]) / f"{safe_name(row['rule_name'])}.pdf"
        meta = {
            "org": row["org"],
            "rule_name": row["rule_name"].strip(),
            "category": row["category"],
            "rule_date": row["rule_date"],
            "source": str(pdf),
        }
        docs += load_pdf(pdf, meta)
    return docs


def main() -> None:
    docs = load_latest_documents()
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(docs)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i

    rules = {d.metadata["rule_name"] for d in docs}
    print(f"규정 {len(rules)}개, 페이지 {len(docs)}개 → chunk {len(chunks)}개")
    print(f"chunk 길이 평균 {sum(len(c.page_content) for c in chunks) // len(chunks)}자")
    print("예시 metadata:", chunks[0].metadata)

    # 다시 실행해도 중복되지 않도록 컬렉션을 새로 만든다
    store = get_vector_store(pre_delete_collection=True)
    store.add_documents(chunks, ids=[f"{c.metadata['org']}-{c.metadata['chunk_id']}" for c in chunks])
    print(f"적재 완료: {len(chunks)}개")


if __name__ == "__main__":
    main()
