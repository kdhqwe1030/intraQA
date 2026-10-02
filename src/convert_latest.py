"""manifest.csv에서 최신본(is_latest)만 골라 LibreOffice로 PDF 변환한다.

HWP 5.x / HWPX 변환에는 LibreOffice 확장 H2Orestart가 필요하다.
확장이 없으면 HWP 5.x가 바이너리 그대로 찍힌 깨진 PDF가 나오므로, 변환 결과에 한글이 있는지 검사한다.

실행: uv run python -m src.convert_latest
"""
import csv
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pymupdf

from src.collect_rules import MANIFEST, safe_name

PDF_DIR = Path("data/pdf")
SOFFICE = shutil.which("soffice") or "/Applications/LibreOffice.app/Contents/MacOS/soffice"
HANGUL = re.compile(r"[가-힣]")


def looks_valid(pdf: Path) -> bool:
    """첫 페이지 텍스트에서 한글 비율로 정상 변환 여부를 판단한다."""
    with pymupdf.open(pdf) as doc:
        text = doc[0].get_text() if doc.page_count else ""
    text = re.sub(r"\s", "", text)
    return bool(text) and len(HANGUL.findall(text)) / len(text) > 0.3


def convert(src: Path, dest: Path) -> None:
    # 출력 파일명이 원본 이름을 따르므로 임시 폴더에서 변환한 뒤 옮긴다
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(
            [SOFFICE, "--headless", "--convert-to", "pdf", "--outdir", tmp, str(src)],
            check=True, capture_output=True, timeout=300,
        )
        out = Path(tmp) / f"{src.stem}.pdf"
        if not out.exists():
            raise RuntimeError("LibreOffice가 파일을 열지 못했습니다")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(out, dest)


def main() -> None:
    with MANIFEST.open(encoding="utf-8-sig") as f:
        latest = [r for r in csv.DictReader(f) if r["is_latest"] == "True"]

    ok, failed = 0, []
    for row in latest:
        src = Path(row["path"])
        dest = PDF_DIR / safe_name(row["org"]) / f"{safe_name(row['rule_name'])}.pdf"
        try:
            if not (dest.exists() and looks_valid(dest)):
                convert(src, dest)
            if not looks_valid(dest):
                dest.unlink()
                raise RuntimeError("변환 결과가 깨졌습니다 (H2Orestart 확인)")
            ok += 1
            print(f"[OK]   {row['rule_name']}")
        except Exception as e:
            failed.append(row["rule_name"])
            print(f"[FAIL] {row['rule_name']} ({src.suffix}): {e}")
    print(f"완료: 성공 {ok} / 실패 {len(failed)} → {PDF_DIR}")


if __name__ == "__main__":
    main()
