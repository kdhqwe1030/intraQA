# Intra QA — 사내 규정 QA 어시스턴트

공공기관 내부규정을 근거로 인사·복무·보수·출장·복리후생 질문에 조문 출처와 함께 답하는 RAG.

## 1. 문제 정의
## 2. 사용 데이터
## 3. RAG 구조

## 4. 설치 방법
```bash
uv sync
cp .env.example .env        # OPENAI_API_KEY, LAW_OC 입력
docker compose up -d        # pgvector (localhost:5445)
uv run python -m src.lib.check_env
```

## 5. 실행 방법
## 6. 테스트 질문
## 7. Baseline 결과
## 8. 개선 방법과 결과
## 9. 한계와 추가 개선 방향
