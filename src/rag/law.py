"""국가법령정보 OPEN API 클라이언트.

법령명으로 현행 법령을 찾고, 조문 단위로 꺼낸다. 조문번호를 모를 때 쓰도록 목차(조문 제목 목록)도 제공한다.
같은 법령을 반복 조회하지 않도록 data/law_cache/ 에 본문을 캐시한다 (평가 때 같은 법령을 여러 번 부른다).
"""
import json
import re
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import requests

from src.lib.config import LAW_OC

BASE_URL = "https://www.law.go.kr/DRF"
CACHE_DIR = Path("data/law_cache")
REQUEST_INTERVAL = 0.5  # 초. 과도한 호출 금지 (API 이용 주의사항)


@dataclass
class Article:
    law_name: str
    article_no: str  # 제74조, 제24조의2
    title: str
    text: str
    effective_date: str


def _get(endpoint: str, **params) -> dict:
    if not LAW_OC:
        raise RuntimeError("LAW_OC가 없습니다 (.env 확인)")
    time.sleep(REQUEST_INTERVAL)
    resp = requests.get(f"{BASE_URL}/{endpoint}", params={"OC": LAW_OC, "type": "JSON", **params}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _as_list(value) -> list:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _text(value) -> str:
    """API는 내용을 문자열 또는 문자열 리스트(줄 단위)로 준다."""
    if isinstance(value, list):
        return "\n".join(_text(v) for v in value)
    return str(value or "").strip()


@lru_cache
def find_law(name: str) -> tuple[str, str] | None:
    """법령명 → (법령일련번호 MST, 정식 법령명). 이름이 정확히 같은 현행 법령을 우선한다."""
    data = _get("lawSearch.do", target="law", query=name)
    laws = _as_list(data.get("LawSearch", {}).get("law"))
    if not laws:
        return None
    # 이름이 정확히 같은 법령만 인정한다. 부분 일치를 허용하면 "여비규정"으로 "공무원 여비 규정"을 가져온다
    exact = [l for l in laws if l["법령명한글"].replace(" ", "") == name.replace(" ", "")]
    if not exact:
        return None
    return exact[0]["법령일련번호"], exact[0]["법령명한글"]


def _load_law(mst: str) -> dict:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"{mst}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    data = _get("lawService.do", target="law", MST=mst)
    cache.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return data


@lru_cache
def get_articles(mst: str) -> tuple[Article, ...]:
    law = _load_law(mst)["법령"]
    name = law["기본정보"]["법령명_한글"]
    effective = law["기본정보"]["시행일자"]
    articles = []
    for unit in _as_list(law["조문"]["조문단위"]):
        if unit.get("조문여부") != "조문":  # 장·절 제목 행은 건너뛴다
            continue
        no = f"제{unit['조문번호']}조" + (f"의{unit['조문가지번호']}" if unit.get("조문가지번호") else "")
        lines = [_text(unit.get("조문내용"))]
        for hang in _as_list(unit.get("항")):
            lines.append(_text(hang.get("항내용")))
            for ho in _as_list(hang.get("호")):
                lines.append("  " + _text(ho.get("호내용")))
                for mok in _as_list(ho.get("목")):
                    lines.append("    " + _text(mok.get("목내용")))
        articles.append(Article(name, no, _text(unit.get("조문제목")), "\n".join(l for l in lines if l.strip()), effective))
    return tuple(articles)


def get_law(law_name: str) -> tuple[Article, ...]:
    found = find_law(law_name)
    return get_articles(found[0]) if found else ()


def table_of_contents(law_name: str) -> str:
    """조문번호와 제목 목록. 조문번호를 모를 때 LLM이 이 목차를 보고 조문을 고른다."""
    return "\n".join(f"{a.article_no}({a.title})" for a in get_law(law_name))


def normalize_article_no(text: str) -> str:
    """'74', '74조', '제 74 조', '제24조의2' → '제74조', '제24조의2'. 형식을 못 알아보면 빈 문자열."""
    m = re.search(r"(\d+)\s*조?(?:\s*의\s*(\d+))?", text or "")
    if not m:
        return ""
    return f"제{m.group(1)}조" + (f"의{m.group(2)}" if m.group(2) else "")


def get_article(law_name: str, article_no: str) -> Article | None:
    target = normalize_article_no(article_no)
    return next((a for a in get_law(law_name) if a.article_no == target), None)


# 법령 이름 후보. 「」로 감싼 이름, 또는 "…법 / …법 시행령 / …법 시행규칙" 형태. 뒤에 조문번호가 붙을 수 있다
LAW_REF_RE = re.compile(
    r"「([^」]{2,40})」(?:\s*(제\s*\d+\s*조(?:\s*의\s*\d+)?))?"
    r"|([가-힣]+법(?:\s*시행령|\s*시행규칙)?)(?:\s*(제\s*\d+\s*조(?:\s*의\s*\d+)?))?"
)


def extract_law_refs(text: str) -> list[tuple[str, str]]:
    """텍스트에 나온 법령 (이름, 조문번호) 후보. 실제 존재하는 법령(이름 정확 일치)만 남긴다."""
    refs = []
    for m in LAW_REF_RE.finditer(text):
        name = (m.group(1) or m.group(3)).strip()
        article = normalize_article_no(m.group(2) or m.group(4) or "")
        if (name, article) not in refs:
            refs.append((name, article))
    return [(n, a) for n, a in refs if find_law(n)]
