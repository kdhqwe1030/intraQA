"""ALIO 공공기관 내부규정 수집.

기관명으로 규정 목록을 조회하고, 규정별 첨부파일(개정본 전체)을 내려받는다.
zip 파일은 같은 폴더에 압축을 푼다. 수집 결과는 data/manifest.csv 에 기록한다.

ALIO 웹 화면이 내부적으로 쓰는 엔드포인트(공식 Open API 아님)라서 요청 사이에 간격을 둔다.

실행: uv run python -m src.collect_rules 한국지역난방공사
"""
import csv
import re
import sys
import time
import zipfile
from pathlib import Path

import requests

BASE_URL = "https://www.alio.go.kr"
RAW_DIR = Path("data/raw")
MANIFEST = Path("data/manifest.csv")
REQUEST_INTERVAL = 1.0  # 초

MANIFEST_FIELDS = [
    "org", "apba_id", "seq", "rule_name", "category", "rule_date",
    "file_no", "file_name", "path", "from_zip", "version_date", "is_latest",
]

# 파일명 예: 직원보수규정(2025년도 12월 30일 개정).hwp, 직제규정시행세칙(2024년도 01월 ② 개정).hwp
VERSION_RE = re.compile(r"(\d{4})년도?\s*(\d{1,2})월(?:\s*(\d{1,2})일)?\s*([\u2460-\u2473])?")

session = requests.Session()
session.headers["User-Agent"] = "intra-qa-collector (personal study project)"


def get_json(path: str, **params) -> dict:
    time.sleep(REQUEST_INTERVAL)
    resp = session.get(BASE_URL + path, params=params, timeout=30)
    resp.raise_for_status()
    body = resp.json()
    if body.get("status") != "success":
        raise RuntimeError(f"{path} 실패: {body}")
    return body["data"]


def list_rules(org: str) -> list[dict]:
    """기관명으로 검색한 규정 목록 (전체 페이지)."""
    rules, page = [], 1
    while True:
        data = get_json("/occasional/findRuleList.json", type="apbaNa", word=org, pageNo=page)
        # 기관명 검색은 부분 일치라서 정확히 같은 기관만 남긴다
        rules += [r for r in data["result"] if r["pname"] == org]
        # nextPage_is는 페이지 묶음(10페이지) 단위라서 totalPage로 판단한다
        if page >= data["page"]["totalPage"]:
            return rules
        page += 1


def list_files(seq: str) -> list[tuple[str, str]]:
    """규정 상세의 첨부파일 목록 [(file_no, file_name), ...]. 개정 순서대로 나온다."""
    b_files = get_json("/occasional/findRuleDtl.json", seq=seq).get("bFiles") or ""
    files = []
    for item in b_files.split(","):
        if "|" in item:
            file_no, file_name = item.split("|", 1)
            files.append((file_no.strip(), file_name.strip()))
    return files


def safe_name(name: str) -> str:
    return re.sub(r'[\\/:*?"<>|]', "_", name).strip()


def download(file_no: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    time.sleep(REQUEST_INTERVAL)
    resp = session.get(f"{BASE_URL}/download/rulefiledown.json", params={"fileNo": file_no}, timeout=60)
    resp.raise_for_status()
    dest.write_bytes(resp.content)


def zip_member_name(info: zipfile.ZipInfo) -> str:
    """UTF-8 플래그가 없는 zip은 파일명이 cp949라서 다시 디코딩한다."""
    if info.flag_bits & 0x800:
        return info.filename
    try:
        return info.filename.encode("cp437").decode("cp949")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return info.filename


def extract_zip(zip_path: Path) -> list[Path]:
    out_dir = zip_path.with_suffix("")
    extracted = []
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            # zip 안 폴더가 여러 겹이라 파일명만 꺼내 한 단계로 편다 (경로 탈출도 막힌다)
            target = out_dir / Path(zip_member_name(info)).name
            out_dir.mkdir(parents=True, exist_ok=True)
            target.write_bytes(zf.read(info))
            extracted.append(target)
    return extracted


def collect(org: str) -> list[dict]:
    rows = []
    rules = list_rules(org)
    print(f"{org}: 규정 {len(rules)}건")
    for i, rule in enumerate(rules, 1):
        rule_dir = RAW_DIR / safe_name(org) / safe_name(rule["title"])
        rule_dir.mkdir(parents=True, exist_ok=True)
        files = list_files(rule["seq"])
        print(f"  [{i}/{len(rules)}] {rule['title']} ({rule['insdRuleDivis']}) - 파일 {len(files)}개")

        base = {
            "org": org, "apba_id": rule["apbaId"], "seq": rule["seq"],
            "rule_name": rule["title"], "category": rule["insdRuleDivis"],
            "rule_date": rule["ruleStDa"],
        }
        for file_no, file_name in files:
            dest = rule_dir / safe_name(file_name)
            try:
                download(file_no, dest)
            except requests.RequestException as e:
                print(f"    [FAIL] {file_name}: {e}")
                continue
            # hwpx도 내부 구조가 zip이라서 확장자로만 판단한다
            if dest.suffix.lower() == ".zip":
                for member in extract_zip(dest):
                    rows.append({**base, "file_no": file_no, "file_name": member.name,
                                 "path": str(member), "from_zip": file_name})
            else:
                rows.append({**base, "file_no": file_no, "file_name": file_name,
                             "path": str(dest), "from_zip": ""})
    return rows


def parse_version(file_name: str) -> tuple[int, int, int, int] | None:
    """파일명에서 (연, 월, 일, 같은 달 순번)을 뽑는다. 날짜가 없으면 None."""
    m = VERSION_RE.search(file_name)
    if not m:
        return None
    year, month, day, circled = m.groups()
    seq = ord(circled) - 0x2460 + 1 if circled else 0
    return int(year), int(month), int(day or 0), seq


def mark_latest(rows: list[dict]) -> None:
    """규정별로 파일명 날짜가 가장 늦은 파일을 최신본으로 표시한다.
    날짜가 있는 파일이 하나도 없으면 (단일 파일 규정) 그 파일을 최신본으로 본다."""
    by_rule: dict[str, list[dict]] = {}
    for row in rows:
        version = parse_version(row["file_name"])
        if version is None:
            row["version_date"] = ""
        elif version[2]:
            row["version_date"] = f"{version[0]}-{version[1]:02d}-{version[2]:02d}"
        else:  # 일자 없이 "2024년도 12월 개정"만 있는 파일
            row["version_date"] = f"{version[0]}-{version[1]:02d}"
        row["is_latest"] = False
        by_rule.setdefault(row["seq"], []).append(row)
    for rule_rows in by_rule.values():
        dated = [r for r in rule_rows if parse_version(r["file_name"])]
        candidates = dated or rule_rows
        latest = max(parse_version(r["file_name"]) or (0, 0, 0, 0) for r in candidates)
        for r in candidates:
            if (parse_version(r["file_name"]) or (0, 0, 0, 0)) == latest:
                r["is_latest"] = True


def write_manifest(org: str, rows: list[dict]) -> None:
    """같은 기관의 기존 행은 교체하고 다른 기관 행은 유지한다."""
    existing = []
    if MANIFEST.exists():
        with MANIFEST.open(encoding="utf-8-sig") as f:
            existing = [r for r in csv.DictReader(f) if r["org"] != org]
    with MANIFEST.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(existing + rows)


if __name__ == "__main__":
    org_name = sys.argv[1] if len(sys.argv) > 1 else "한국지역난방공사"
    result = collect(org_name)
    mark_latest(result)
    write_manifest(org_name, result)
    print(f"완료: 파일 {len(result)}개 → {MANIFEST}")
