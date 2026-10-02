"""0단계 환경 점검: DB(pgvector), OpenAI, 국가법령정보 API 연결을 한 번씩 확인한다.

실행: uv run python -m src.lib.check_env
"""
import os

import psycopg
import requests

from src.lib.config import LAW_OC, OPENAI_CHAT_MODEL, OPENAI_EMBEDDING_MODEL, pg_url


def check_db() -> None:
    with psycopg.connect(pg_url(driver="")) as conn:
        version = conn.execute(
            "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
        ).fetchone()
    if version is None:
        raise RuntimeError("vector 확장이 설치되지 않았습니다 (db/init.sql 확인)")
    print(f"[OK] DB 연결, pgvector {version[0]}")


def check_openai() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        print("[SKIP] OPENAI_API_KEY 없음")
        return
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    reply = ChatOpenAI(model=OPENAI_CHAT_MODEL).invoke("한 단어로 답해: 안녕")
    dim = len(OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL).embed_query("테스트"))
    print(f"[OK] OpenAI chat={reply.content!r}, embedding dim={dim}")


def check_law_api() -> None:
    if not LAW_OC:
        print("[SKIP] LAW_OC 없음")
        return
    resp = requests.get(
        "https://www.law.go.kr/DRF/lawSearch.do",
        params={"OC": LAW_OC, "target": "law", "type": "JSON", "query": "근로기준법"},
        timeout=10,
    )
    resp.raise_for_status()
    try:
        data = resp.json()["LawSearch"]
    except (ValueError, KeyError):
        raise RuntimeError(f"예상과 다른 응답 (OC/IP 등록 확인): {resp.text[:200]}")
    print(f"[OK] 법령 API, '근로기준법' 검색 결과 {data.get('totalCnt')}건")


if __name__ == "__main__":
    for check in (check_db, check_openai, check_law_api):
        try:
            check()
        except Exception as e:
            print(f"[FAIL] {check.__name__}: {e}")
