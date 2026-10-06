"""BM25 키워드 검색. pgvector에 적재된 청크를 그대로 읽어 기관별 인덱스를 메모리에 만든다.

BM25는 필터 기능이 없어서, 다른 기관 문서가 섞이지 않도록 기관마다 인덱스를 따로 둔다 (기획서 9-3).
"""
import re
from functools import lru_cache

import psycopg
from kiwipiepy import Kiwi
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from src.lib.config import pg_url

# 내용어만 남긴다: 명사, 동사·형용사 어간, 어근, 숫자, 외국어
KEEP_TAGS = ("NNG", "NNP", "NNB", "NR", "VV", "VA", "XR", "SN", "SL")
# "제13조", "제24조의2", "별표1"은 형태소 분석이 쪼개 버리므로 통째로 토큰을 하나 더 만든다
ARTICLE_RE = re.compile(r"제\s*\d+\s*조(?:\s*의\s*\d+)?|별표\s*\d+(?:\s*의\s*\d+)?")

kiwi = Kiwi()


def tokenize(text: str) -> list[str]:
    tokens = [t.form for t in kiwi.tokenize(text) if t.tag.startswith(KEEP_TAGS)]
    tokens += [re.sub(r"\s", "", m) for m in ARTICLE_RE.findall(text)]
    return tokens


def load_chunks(org: str, index: str) -> list[Document]:
    sql = """
        SELECT e.document, e.cmetadata
        FROM langchain_pg_embedding e
        JOIN langchain_pg_collection c ON e.collection_id = c.uuid
        WHERE c.name = %s AND e.cmetadata->>'org' = %s
        ORDER BY (e.cmetadata->>'chunk_id')::int
    """
    with psycopg.connect(pg_url(driver="")) as conn:
        rows = conn.execute(sql, (index, org)).fetchall()
    return [Document(page_content=text, metadata=meta) for text, meta in rows]


class BM25Index:
    def __init__(self, docs: list[Document]):
        self.docs = docs
        self.bm25 = BM25Okapi([tokenize(d.page_content) for d in docs])

    def search(self, query: str, k: int) -> list[Document]:
        scores = self.bm25.get_scores(tokenize(query))
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self.docs[i] for i in top if scores[i] > 0]


@lru_cache
def get_bm25_index(org: str, index: str) -> BM25Index:
    """(기관, 청킹 인덱스)마다 한 번만 만들고 재사용한다. 재적재(ingest) 후에는 프로세스를 다시 띄운다."""
    docs = load_chunks(org, index)
    if not docs:
        raise ValueError(f"{org} / {index} 청크가 없습니다. 먼저 ingest --index {index}를 실행하세요.")
    return BM25Index(docs)
