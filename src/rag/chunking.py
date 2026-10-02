"""청킹 방식.

  baseline   : 페이지 단위 → RecursiveCharacterTextSplitter (교재 기본)
  structured : 규정 전체를 이어 붙인 뒤 조문·별표·별지·부칙 단위로 자른다 (개선 2)

structured는 각 청크 앞에 "[규정명 제N조(제목)]" 머리말을 붙이고, 조문번호와 걸친 페이지를
metadata로 남긴다. 페이지 경계에서 조문이 잘리지 않고, 출처 조문번호를 지어내지 않게 하려는 것이다.
"""
import re
from dataclasses import dataclass, field

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
MAX_SECTION = 1500  # 조문이 이보다 길면 줄 단위로 나누되, 나눈 조각마다 머리말을 다시 붙인다
OVERLAP_LINES = 2  # 나눈 조각 사이에 겹치는 줄 수

PAGE_HEADER_RE = re.compile(r"^\d{4}\s+\S")  # "5300 직원보수규정" 같은 쪽 머리말
PAGE_NO_RE = re.compile(r"^\d+$")  # 쪽 번호

# 섹션 시작 패턴. 줄 맨 앞에 올 때만 인정한다 (본문 중간의 "제4조의 규정에 따라" 같은 인용은 제외)
ARTICLE_RE = re.compile(r"^제\s*(\d+)\s*조(?:\s*의\s*(\d+))?\s*(?:\(([^)]*)\)|(?=\s+삭제))")
APPENDIX_TABLE_RE = re.compile(r"^[\[(]\s*별\s*표\s*(\d+(?:\s*의\s*\d+)?)?\s*[\])]")
FORM_RE = re.compile(r"^[\[(]\s*별\s*지")
SUPPLEMENT_RE = re.compile(r"^부\s*칙")
CHAPTER_RE = re.compile(r"^제\s*\d+\s*[장절]\s*\S")


def baseline_chunks(pages: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return splitter.split_documents(pages)


@dataclass
class Section:
    kind: str  # article / table / form / supplement / preamble
    label: str  # "제13조(국내출장여비지급 기준)", "별표 1" 등
    article_no: str
    lines: list[tuple[str, int]] = field(default_factory=list)  # (줄, 페이지)

    def add(self, line: str, page: int) -> None:
        self.lines.append((line, page))


def clean_lines(text: str) -> list[str]:
    lines = [l.strip() for l in text.split("\n")]
    lines = [l for l in lines if l]
    if lines and PAGE_HEADER_RE.match(lines[0]):
        lines = lines[1:]
    if lines and PAGE_NO_RE.match(lines[-1]):
        lines = lines[:-1]
    return lines


def detect(line: str) -> Section | None:
    if m := ARTICLE_RE.match(line):
        no = f"제{m.group(1)}조" + (f"의{m.group(2)}" if m.group(2) else "")
        title = f"({m.group(3)})" if m.group(3) else " 삭제"
        return Section("article", no + title, no)
    if m := APPENDIX_TABLE_RE.match(line):
        no = f"별표{re.sub(r'\s', '', m.group(1) or '')}"
        return Section("table", no, no)
    if FORM_RE.match(line):
        return Section("form", "별지 서식", "")
    if SUPPLEMENT_RE.match(line):
        return Section("supplement", "부칙", "")
    return None


def split_sections(pages: list[Document]) -> list[Section]:
    sections = [Section("preamble", "머리말", "")]
    chapter = ""
    for page in pages:
        for line in clean_lines(page.page_content):
            if CHAPTER_RE.match(line) and sections[-1].kind in ("article", "preamble"):
                chapter = line  # 장·절 제목은 따로 청크로 만들지 않고 이후 조문 머리말에 쓴다
                continue
            new = detect(line)
            # 부칙 안의 조문("제1조(시행일)")과 이어지는 부칙은 부칙 하나로 묶는다
            if new and sections[-1].kind == "supplement" and new.kind in ("supplement", "article"):
                new = None
            if new:
                if new.kind != "article":
                    chapter = ""  # 부칙·별표·별지에 들어가면 장 이름을 더 이상 붙이지 않는다
                if new.kind == "article" and chapter:
                    new.label = f"{chapter} > {new.label}"
                sections.append(new)
            sections[-1].add(line, page.metadata["page"])
    return [s for s in sections if s.lines]


def pack_lines(lines: list[tuple[str, int]]) -> list[list[tuple[str, int]]]:
    """긴 섹션을 MAX_SECTION 이하 조각으로 나눈다. 줄 단위로 나눠서 조각마다 실제 페이지를 알 수 있다."""
    parts, current, size = [], [], 0
    for line in lines:
        if current and size + len(line[0]) > MAX_SECTION:
            parts.append(current)
            current = current[-OVERLAP_LINES:]
            size = sum(len(l) for l, _ in current)
        current.append(line)
        size += len(line[0]) + 1
    parts.append(current)
    return parts


def structured_chunks(pages: list[Document]) -> list[Document]:
    """한 규정의 페이지들을 받아 조문 단위 청크로 만든다."""
    if not pages:
        return []
    base = {k: v for k, v in pages[0].metadata.items() if k != "page"}

    chunks = []
    for s in split_sections(pages):
        header = f"[{base['rule_name']} {s.label}]"
        for part in pack_lines(s.lines):
            part_pages = list(dict.fromkeys(p for _, p in part))
            chunks.append(Document(
                page_content=header + "\n" + "\n".join(l for l, _ in part),
                metadata={
                    **base,
                    "page": part_pages[0],
                    "pages": part_pages,
                    "section_type": s.kind,
                    "article_no": s.article_no,
                    "section_label": s.label,
                },
            ))
    return chunks
