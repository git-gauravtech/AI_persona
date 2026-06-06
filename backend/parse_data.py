"""
parse_data.py
Parses resume + all repo files into structured chunks with metadata.
Output: data/processed/parsed_chunks.json

RAG-optimized version:
- Resume is split into section chunks, FAQ chunks, paragraph chunks, fact chunks, and anchor chunks.
- README files are split by markdown headings.
- commits.txt and prs.txt are split by semantic blocks like Commit Group, PR, Important technical evidence, and RAG answer support.
"""

import json
import re
from pathlib import Path

try:
    import pdfplumber
except ImportError:
    pdfplumber = None


# ── CONFIG ──────────────────────────────────────────────────────────────────

DATA_DIR = Path("data")
OUTPUT_FILE = DATA_DIR / "processed" / "parsed_chunks.json"

MAX_CHARS = 1100
OVERLAP_CHARS = 220


# ── BASIC HELPERS ────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Normalize whitespace but keep useful newlines."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_long_text(
    text: str,
    max_chars: int = MAX_CHARS,
    overlap: int = OVERLAP_CHARS
) -> list[str]:
    """
    Split long text into overlapping chunks.
    Tries paragraph boundary first, then sentence/line boundary.
    """
    text = clean_text(text)

    if not text:
        return []

    if len(text) <= max_chars:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + max_chars

        if end >= len(text):
            final_chunk = clean_text(text[start:])
            if final_chunk:
                chunks.append(final_chunk)
            break

        window = text[start:end]

        split_at = window.rfind("\n\n")

        if split_at < max_chars * 0.45:
            split_at = max(window.rfind(". "), window.rfind("\n"))

        if split_at < max_chars * 0.45:
            split_at = len(window)

        chunk = clean_text(text[start:start + split_at])
        if chunk:
            chunks.append(chunk)

        new_start = start + split_at - overlap

        if new_start <= start:
            new_start = start + split_at

        start = max(new_start, 0)

    return chunks


def safe_json_load(path: Path) -> dict:
    if not path.exists():
        return {}

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"    [warn] Could not read {path}: {e}")
        return {}


def load_repo_metadata(repo_dir: Path) -> dict:
    return safe_json_load(repo_dir / "metadata.json")


# ── RESUME PARSING ───────────────────────────────────────────────────────────

RESUME_HEADINGS = [
    "CONTACT INFORMATION",
    "SUMMARY",
    "PROFILE SUMMARY",
    "QUICK VERIFIED FACTS",
    "CURRENT EDUCATION AND STATUS",
    "TARGET ROLE",
    "EDUCATION",
    "COURSEWORK",
    "SKILLS",
    "TECHNICAL SKILLS",
    "PROJECTS",
    "ACHIEVEMENTS",
    "HONOURS AND AWARDS",
    "HONORS AND AWARDS",
    "PATENT",
    "PUBLICATION",
    "WHAT MAKES GAURAV A GOOD FIT FOR AI ENGINEER INTERN ROLE",
    "EXPECTATIONS FROM INTERNSHIP",
    "PERSONAL WORK STYLE",
    "FREQUENTLY ASKED VERIFIED ANSWERS",
    "UNKNOWN OR NOT VERIFIED INFORMATION",
]


def is_heading_line(line: str) -> bool:
    line = line.strip()

    if not line:
        return False

    normalized = line.upper().strip()

    if normalized in RESUME_HEADINGS:
        return True

    # Generic uppercase heading support
    if len(line) <= 90 and normalized == line and re.search(r"[A-Z]", line):
        return True

    return False


def split_resume_sections(text: str) -> list[tuple[str, str]]:
    """
    Split resume into sections based on standalone heading lines.
    Returns list of (section_title, section_text).
    """
    lines = text.splitlines()

    sections = []
    current_title = "INTRO"
    current_lines = []

    for line in lines:
        stripped = line.strip()

        if is_heading_line(stripped):
            if current_lines:
                section_text = clean_text("\n".join(current_lines))
                if section_text:
                    sections.append((current_title, section_text))

            current_title = stripped
            current_lines = [stripped]

        else:
            current_lines.append(line)

    if current_lines:
        section_text = clean_text("\n".join(current_lines))
        if section_text:
            sections.append((current_title, section_text))

    return sections


def make_resume_chunk(
    text: str,
    txt_path: Path,
    section_title: str,
    section_index: int,
    chunk_index: int,
    chunk_style: str
) -> dict:
    return {
        "text": text,
        "metadata": {
            "source": "resume_txt",
            "type": "resume_section" if chunk_style == "full_section" else "resume_detail",
            "file": str(txt_path),
            "section": section_title,
            "section_index": section_index,
            "chunk_index": chunk_index,
            "chunk_style": chunk_style,
            "on_resume": True,
            "ownership": "personal",
            "priority": "deep",
        }
    }


def looks_like_important_resume_line(line: str) -> bool:
    """
    Detect dense resume facts that should become separate retrieval chunks.
    This helps indirect questions like:
    - Why is Gaurav fit for AI Engineer?
    - What backend experience does he have?
    - What achievements does he have?
    """
    stripped = line.strip()
    lowered = stripped.lower()

    if len(stripped) < 45:
        return False

    starts_good = stripped.startswith(("-", "*", "•"))

    keyword_hit = any(k in stripped for k in [
        "Project:",
        "Skills:",
        "Tech Stack:",
        "Achievement:",
        "Role:",
        "Built",
        "Developed",
        "Implemented",
        "Integrated",
        "Optimized",
        "Deployed",
        "Created",
        "Designed",
        "Worked",
        "Experience",
        "GitHub",
        "LinkedIn",
        "Hackathon",
        "Amazon ML Summer School",
        "DSA",
        "RAG",
        "LLM",
        "AI",
        "ML",
        "Backend",
        "Flask",
        "FastAPI",
        "MongoDB",
        "Pinecone",
        "Groq",
        "Vapi",
        "Cal.com",
    ])

    signal_hit = any(k in lowered for k in [
        "b.tech",
        "computer science",
        "artificial intelligence",
        "machine learning",
        "backend developer",
        "ai integration",
        "400+",
        "leetcode",
        "codeforces",
        "hackathon",
        "intern",
        "rag",
        "fine-tuning",
        "prompt engineering",
        "vector database",
    ])

    return starts_good or keyword_hit or signal_hit


def parse_txt_resume(txt_path: Path) -> list[dict]:
    """
    Parse resume.txt into many focused chunks for better retrieval.
    """
    chunks = []
    text = clean_text(txt_path.read_text(encoding="utf-8", errors="ignore"))

    if not text:
        return chunks

    sections = split_resume_sections(text)

    for section_index, (section_title, section_text) in enumerate(sections):
        if len(section_text) < 30:
            continue

        # 1. Full section chunks
        full_parts = split_long_text(section_text, max_chars=1500, overlap=220)

        for part_index, part in enumerate(full_parts):
            chunks.append(
                make_resume_chunk(
                    text=part,
                    txt_path=txt_path,
                    section_title=section_title,
                    section_index=section_index,
                    chunk_index=part_index,
                    chunk_style="full_section",
                )
            )

        # 2. FAQ chunks: each Question/Answer becomes separate chunk
        if "Question:" in section_text:
            faq_blocks = re.split(r"(?=Question:)", section_text)

            for faq_index, block in enumerate(faq_blocks):
                block = clean_text(block)

                if len(block) < 50:
                    continue

                chunks.append(
                    make_resume_chunk(
                        text=f"{section_title}\n\n{block}",
                        txt_path=txt_path,
                        section_title=section_title,
                        section_index=section_index,
                        chunk_index=1000 + faq_index,
                        chunk_style="faq",
                    )
                )

        # 3. Paragraph chunks
        paragraphs = re.split(r"\n\s*\n", section_text)

        for para_index, para in enumerate(paragraphs):
            para = clean_text(para)

            if len(para) < 80:
                continue

            if para == section_text:
                continue

            paragraph_text = f"{section_title}\n\n{para}"
            para_parts = split_long_text(paragraph_text, max_chars=950, overlap=150)

            for sub_index, para_part in enumerate(para_parts):
                chunks.append(
                    make_resume_chunk(
                        text=para_part,
                        txt_path=txt_path,
                        section_title=section_title,
                        section_index=section_index,
                        chunk_index=(2000 + para_index * 100) + sub_index,
                        chunk_style="paragraph",
                    )
                )

        # 4. Important resume fact chunks
        important_lines = []

        for line in section_text.splitlines():
            line = clean_text(line)

            if looks_like_important_resume_line(line):
                important_lines.append(line)

        for line_index, line in enumerate(important_lines):
            chunks.append(
                make_resume_chunk(
                    text=f"{section_title}\n\nVerified resume fact:\n{line}",
                    txt_path=txt_path,
                    section_title=section_title,
                    section_index=section_index,
                    chunk_index=9000 + line_index,
                    chunk_style="resume_fact",
                )
            )

        # 5. Section anchor chunk for indirect questions
        summary_anchor = clean_text(
            f"Resume section: {section_title}\n"
            f"This section contains verified information about Gaurav Saklani related to {section_title}. "
            f"Use this section when answering recruiter questions about Gaurav's {section_title.lower()}.\n\n"
            f"{section_text[:1200]}"
        )

        if len(summary_anchor) > 120:
            chunks.append(
                make_resume_chunk(
                    text=summary_anchor,
                    txt_path=txt_path,
                    section_title=section_title,
                    section_index=section_index,
                    chunk_index=9900,
                    chunk_style="section_anchor",
                )
            )

    return chunks


def parse_resume(pdf_path: Path) -> list[dict]:
    """Extract text from resume PDF and chunk by sections."""
    chunks = []

    if pdfplumber is None:
        print("  [warn] pdfplumber not installed, skipping PDF resume")
        return chunks

    with pdfplumber.open(pdf_path) as pdf:
        pages = []

        for page in pdf.pages:
            page_text = page.extract_text() or ""
            pages.append(page_text)

    full_text = clean_text("\n".join(pages))

    if not full_text:
        print("  [warn] No text extracted from resume PDF")
        return chunks

    temp_txt_path = pdf_path.with_suffix(".txt")
    temp_txt_path.write_text(full_text, encoding="utf-8")

    chunks = parse_txt_resume(temp_txt_path)

    for chunk in chunks:
        chunk["metadata"]["file"] = str(pdf_path)
        chunk["metadata"]["source"] = "resume"

    try:
        temp_txt_path.unlink()
    except Exception:
        pass

    return chunks


# ── REPO PARSING ─────────────────────────────────────────────────────────────

def infer_file_type(stem: str) -> str:
    if stem == "readme":
        return "readme_section"

    if stem == "commits":
        return "commits"

    if stem == "prs":
        return "pull_request"

    return stem


def split_structured_repo_text(text: str, file_type: str) -> list[str]:
    """
    Split commits.txt and prs.txt using meaningful RAG boundaries.
    Works with:
    - Project Summary
    - Commit Group 1 / Commit 1
    - PR #1 / PR 1
    - Important technical evidence
    - RAG answer support
    """
    text = clean_text(text)

    if not text:
        return []

    if file_type == "commits":
        pattern = (
            r"(?=\n?(?:"
            r"Project Summary:|"
            r"Commit Group \d+|"
            r"Commit \d+|"
            r"Important technical evidence|"
            r"ML Model Evidence|"
            r"Dataset Evidence|"
            r"RAG answer support"
            r"):?)"
        )

    elif file_type == "prs":
        pattern = (
            r"(?=\n?(?:"
            r"PR Evidence Status:|"
            r"PR #\d+|"
            r"PR \d+|"
            r"Known contribution evidence|"
            r"Contribution Summary|"
            r"Supported contribution areas|"
            r"Important caution for RAG|"
            r"AI/ML caution|"
            r"RAG answer support"
            r"):?)"
        )

    else:
        return split_long_text(text, max_chars=1100, overlap=180)

    parts = re.split(pattern, text)
    chunks = []

    # Keep repository/role header attached to each chunk
    header_lines = []

    for line in text.splitlines():
        line = line.strip()

        if not line:
            continue

        if re.match(
            r"^(Project Summary:|Commit Group \d+|Commit \d+|PR Evidence Status:|PR #\d+|PR \d+|Known contribution evidence|Contribution Summary|Important technical evidence|RAG answer support)",
            line
        ):
            break

        header_lines.append(line)

    header = clean_text("\n".join(header_lines[:10]))

    for part in parts:
        part = clean_text(part)

        if len(part) < 40:
            continue

        if header and not part.startswith("Repository:") and header not in part[:500]:
            part = clean_text(header + "\n\n" + part)

        sub_parts = split_long_text(part, max_chars=1200, overlap=180)
        chunks.extend(sub_parts)

    return chunks


def parse_txt_file(
    file_path: Path,
    repo_name: str,
    file_type: str,
    ownership: str,
    on_resume: bool,
    repo_metadata: dict,
) -> list[dict]:
    """
    Parse repo readme/commits/prs into chunks.
    """
    chunks = []
    raw_text = file_path.read_text(encoding="utf-8", errors="ignore")
    text = clean_text(raw_text)

    if not text:
        return chunks

    parsed_type = infer_file_type(file_type)

    # README: split by markdown headings
    if file_type == "readme":
        parts = re.split(r"\n(?=#{1,4}\s+)", text)

        if len(parts) <= 1:
            parts = split_long_text(text, max_chars=1300, overlap=180)

        for idx, part in enumerate(parts):
            part = clean_text(part)

            if len(part) < 40:
                continue

            sub_parts = split_long_text(part, max_chars=1300, overlap=180)

            for sub_idx, sub_part in enumerate(sub_parts):
                chunks.append({
                    "text": sub_part,
                    "metadata": {
                        **repo_metadata,
                        "source": repo_name,
                        "type": parsed_type,
                        "file": str(file_path),
                        "chunk_index": (idx * 100) + sub_idx,
                        "ownership": ownership,
                        "on_resume": on_resume,
                        "priority": "deep" if on_resume else repo_metadata.get("priority", "medium"),
                    }
                })

    # commits.txt / prs.txt: semantic chunks
    else:
        semantic_parts = split_structured_repo_text(text, file_type=file_type)

        for idx, part in enumerate(semantic_parts):
            part = clean_text(part)

            if len(part) < 40:
                continue

            chunks.append({
                "text": part,
                "metadata": {
                    **repo_metadata,
                    "source": repo_name,
                    "type": parsed_type,
                    "file": str(file_path),
                    "chunk_index": idx,
                    "ownership": ownership,
                    "on_resume": on_resume,
                    "priority": "deep" if on_resume else repo_metadata.get("priority", "medium"),
                }
            })

    return chunks


def parse_repos(repos_dir: Path) -> list[dict]:
    """Walk through personal/ and contributed/ and parse all repo files."""
    all_chunks = []

    for ownership in ["personal", "contributed"]:
        owner_dir = repos_dir / ownership

        if not owner_dir.exists():
            print(f"  [skip] {owner_dir} not found")
            continue

        for repo_dir in sorted(owner_dir.iterdir()):
            if not repo_dir.is_dir():
                continue

            folder_name = repo_dir.name
            on_resume = folder_name.startswith("resume_")

            repo_metadata = load_repo_metadata(repo_dir)

            clean_name = repo_metadata.get("repo_name") or folder_name.replace("resume_", "")

            repo_metadata["priority"] = repo_metadata.get(
                "priority",
                "deep" if on_resume else "medium"
            )

            print(
                f"  Parsing repo: {clean_name} | "
                f"ownership={ownership} | on_resume={on_resume}"
            )

            for file_path in sorted(repo_dir.iterdir()):
                if not file_path.is_file():
                    continue

                stem = file_path.stem.lower()

                if stem in ("readme", "commits", "prs"):
                    chunks = parse_txt_file(
                        file_path=file_path,
                        repo_name=clean_name,
                        file_type=stem,
                        ownership=ownership,
                        on_resume=on_resume,
                        repo_metadata=repo_metadata,
                    )

                    all_chunks.extend(chunks)
                    print(f"    {file_path.name} → {len(chunks)} chunks")

                elif file_path.name == "metadata.json":
                    continue

                else:
                    print(f"    [skip] unrecognized file: {file_path.name}")

    return all_chunks


# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    all_chunks = []
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    resume_txt_path = DATA_DIR / "resume" / "resume.txt"
    resume_pdf_path = DATA_DIR / "resume" / "resume.pdf"

    if resume_txt_path.exists():
        print("Parsing resume.txt...")
        resume_chunks = parse_txt_resume(resume_txt_path)
        all_chunks.extend(resume_chunks)
        print(f"  → {len(resume_chunks)} chunks from resume.txt")

    elif resume_pdf_path.exists():
        print("Parsing resume.pdf...")
        resume_chunks = parse_resume(resume_pdf_path)
        all_chunks.extend(resume_chunks)
        print(f"  → {len(resume_chunks)} chunks from resume.pdf")

    else:
        print(f"[warn] No resume found at {resume_txt_path} or {resume_pdf_path}")

    repos_dir = DATA_DIR / "repos"

    if repos_dir.exists():
        print("\nParsing repos...")
        repo_chunks = parse_repos(repos_dir)
        all_chunks.extend(repo_chunks)
        print(f"\n  → {len(repo_chunks)} total chunks from repos")

    else:
        print(f"[warn] Repos dir not found at {repos_dir}")

    # Add global chunk index for easier debugging
    for idx, chunk in enumerate(all_chunks):
        chunk["metadata"]["global_chunk_index"] = idx

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Done. Total chunks: {len(all_chunks)}")
    print(f"   Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()