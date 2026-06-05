"""
parse_data.py
Parses resume + all repo files into structured chunks with metadata.
Output: data/processed/parsed_chunks.json
"""

import json
import re
from pathlib import Path
import pdfplumber  # pip install pdfplumber

# ── CONFIG ──────────────────────────────────────────────────────────────────
DATA_DIR = Path("data")
OUTPUT_FILE = DATA_DIR / "processed" / "parsed_chunks.json"
# ─────────────────────────────────────────────────────────────────────────────


def clean_text(text: str) -> str:
    """Normalize whitespace."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def load_repo_metadata(repo_dir: Path) -> dict:
    """Load optional metadata.json from repo folder."""
    metadata_path = repo_dir / "metadata.json"

    if not metadata_path.exists():
        return {}

    try:
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"    [warn] Could not read metadata.json in {repo_dir}: {e}")
        return {}


def parse_resume(pdf_path: Path) -> list[dict]:
    """Extract text from resume PDF and chunk by likely resume sections."""
    chunks = []

    with pdfplumber.open(pdf_path) as pdf:
        pages = []
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            pages.append(page_text)

    full_text = clean_text("\n".join(pages))

    if not full_text:
        print("  [warn] No text extracted from resume PDF")
        return chunks

    section_pattern = re.compile(
        r"\n(?=\s*(?:EDUCATION|EXPERIENCE|PROJECTS|SKILLS|SUMMARY|CERTIFICATIONS|ACHIEVEMENTS|POSITION OF RESPONSIBILITY|COURSEWORK)\s*\n)",
        re.IGNORECASE,
    )

    sections = section_pattern.split(full_text)

    for idx, section in enumerate(sections):
        section = clean_text(section)
        if len(section) < 30:
            continue

        chunks.append({
            "text": section,
            "metadata": {
                "source": "resume",
                "type": "resume_section",
                "file": str(pdf_path),
                "section_index": idx,
                "on_resume": True,
                "ownership": "personal",
                "priority": "deep"
            }
        })

    return chunks


def parse_txt_resume(txt_path: Path) -> list[dict]:
    """Optional: parse clean resume.txt if available."""
    chunks = []
    text = clean_text(txt_path.read_text(encoding="utf-8", errors="ignore"))

    if not text:
        return chunks

    section_pattern = re.compile(
        r"\n(?=\s*(?:EDUCATION|EXPERIENCE|PROJECTS|SKILLS|SUMMARY|CERTIFICATIONS|ACHIEVEMENTS|POSITION OF RESPONSIBILITY|COURSEWORK)\s*\n)",
        re.IGNORECASE,
    )

    sections = section_pattern.split(text)

    for idx, section in enumerate(sections):
        section = clean_text(section)
        if len(section) < 30:
            continue

        chunks.append({
            "text": section,
            "metadata": {
                "source": "resume_txt",
                "type": "resume_section",
                "file": str(txt_path),
                "section_index": idx,
                "on_resume": True,
                "ownership": "personal",
                "priority": "deep"
            }
        })

    return chunks


def parse_txt_file(
    file_path: Path,
    repo_name: str,
    file_type: str,
    ownership: str,
    on_resume: bool,
    repo_metadata: dict
) -> list[dict]:
    """Parse readme / commits / prs into chunks."""
    chunks = []
    text = clean_text(file_path.read_text(encoding="utf-8", errors="ignore"))

    if not text:
        return chunks

    base_metadata = {
        "source": repo_name,
        "type": file_type,
        "file": str(file_path),
        "on_resume": on_resume,
        "ownership": ownership,
        "github_url": repo_metadata.get("github_url"),
        "role": repo_metadata.get("role"),
        "priority": repo_metadata.get("priority", "deep" if on_resume else "medium"),
        "tech_stack": repo_metadata.get("tech_stack"),
        "project_type": repo_metadata.get("project_type"),
        "contribution_scope": repo_metadata.get("contribution_scope"),
    }

    # Remove None values
    base_metadata = {k: v for k, v in base_metadata.items() if v is not None}

    if file_type == "readme":
        sections = re.split(r"\n(?=#{1,3}\s+)", text)

        for idx, section in enumerate(sections):
            section = clean_text(section)
            if len(section) < 30:
                continue

            chunks.append({
                "text": f"Repository: {repo_name}\nFile: README\n\n{section}",
                "metadata": {
                    **base_metadata,
                    "type": "readme_section",
                    "section_index": idx,
                }
            })

    elif file_type == "commits":
        lines = [line.strip() for line in text.splitlines() if line.strip()]

        for i in range(0, len(lines), 10):
            batch = "\n".join(lines[i:i + 10])
            if not batch:
                continue

            chunks.append({
                "text": f"Repository: {repo_name}\nCommit history:\n{batch}",
                "metadata": {
                    **base_metadata,
                    "type": "commits",
                    "commit_batch_start": i,
                    "commit_batch_end": min(i + 10, len(lines)),
                }
            })

    elif file_type == "prs":
        prs = re.split(r"\n{2,}", text)

        for idx, pr in enumerate(prs):
            pr = clean_text(pr)
            if len(pr) < 20:
                continue

            chunks.append({
                "text": f"Repository: {repo_name}\nPull request / contribution note:\n{pr}",
                "metadata": {
                    **base_metadata,
                    "type": "pull_request",
                    "pr_index": idx,
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


def main():
    all_chunks = []
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # 1. Prefer resume.txt if available, fallback to resume.pdf
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

    # 2. Repos
    repos_dir = DATA_DIR / "repos"

    if repos_dir.exists():
        print("\nParsing repos...")
        repo_chunks = parse_repos(repos_dir)
        all_chunks.extend(repo_chunks)
        print(f"\n  → {len(repo_chunks)} total chunks from repos")
    else:
        print(f"[warn] Repos dir not found at {repos_dir}")

    # 3. Save output
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Done. Total chunks: {len(all_chunks)}")
    print(f"   Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()