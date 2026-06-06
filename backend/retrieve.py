import os
import re
from dotenv import load_dotenv
from pinecone import Pinecone
from fastembed import TextEmbedding

load_dotenv()

PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]

INDEX_NAME = "ai-persona-local"
NAMESPACE = "gaurav-ai-persona"

# Must match embed_and_upsert.py
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

TOP_K_CHAT = 8
TOP_K_VOICE = 5

print(f"Loading FastEmbed retrieval model: {EMBEDDING_MODEL}")
embedding_model = TextEmbedding(model_name=EMBEDDING_MODEL)

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)


# ── PROJECT / REPO ALIASES ────────────────────────────────────────────────────

REPO_ALIASES = {
    "vocalis-ai": [
        "vocalis",
        "vocalis-ai",
        "vocalis ai",
        "intelligent ai interview",
        "interview simulation",
        "mock interview",
        "ai interview",
        "voice interview",
        "resume based interview",
        "behavioral analysis",
    ],
    "flask-ml-app": [
        "flask-ml-app",
        "vehicle breakdown",
        "vehicle breakdown predictor",
        "breakdown predictor",
        "flask ml app",
        "vehicle ml",
        "voting classifier",
        "maintenance prediction",
        "ensemble learning",
    ],
    "QuestForge": [
        "questforge",
        "question paper creator",
        "question paper generator",
        "ai assessment",
        "assessment creator",
        "exam creator",
        "paper generator",
        "bullmq",
        "dsl parser",
        "mermaid",
        "question paper",
        "assignment generation",
    ],
    "Indoor-plant-health-detection": [
        "indoor plant",
        "plant health",
        "plant health detection",
        "leaf health",
        "mobilenet",
        "grad-cam",
        "grad cam",
        "plant disease",
        "streamlit plant",
        "plant leaf",
    ],
    "Agroshakti": [
        "agroshakti",
        "agro shakti",
        "smart farming",
        "ai farming",
        "agriculture platform",
        "crop disease",
        "farmer",
        "elevenlabs",
        "stt",
        "tts",
        "scheme search",
        "agritech",
    ],
    "micromatch": [
        "micromatch",
        "micro match",
        "micro influencer",
        "influencer marketing",
        "fake influencer",
        "influencer verification",
        "meta api",
        "campaign analytics",
    ],
}


# ── BASIC HELPERS ─────────────────────────────────────────────────────────────

def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def detect_repo(query: str) -> str | None:
    q = normalize_text(query)

    for source, aliases in REPO_ALIASES.items():
        for alias in aliases:
            if alias.lower() in q:
                return source

    return None


def embed_query(query: str) -> list[float]:
    embedding = next(embedding_model.embed([query]))

    if hasattr(embedding, "tolist"):
        return embedding.tolist()

    return list(embedding)


def retrieve(
    query: str,
    top_k: int = TOP_K_CHAT,
    metadata_filter: dict | None = None
) -> list[dict]:
    query_embedding = embed_query(query)

    query_params = {
        "vector": query_embedding,
        "top_k": top_k,
        "include_metadata": True,
        "namespace": NAMESPACE,
    }

    if metadata_filter:
        query_params["filter"] = metadata_filter

    results = index.query(**query_params)

    chunks = []

    for match in results.matches:
        metadata = match.metadata or {}

        chunks.append({
            "text": metadata.get("text", ""),
            "score": round(match.score, 4),
            "source": metadata.get("source", ""),
            "type": metadata.get("type", ""),
            "ownership": metadata.get("ownership", ""),
            "on_resume": metadata.get("on_resume", False),
            "github_url": metadata.get("github_url", ""),
            "role": metadata.get("role", ""),
            "priority": metadata.get("priority", ""),
            "project_type": metadata.get("project_type", ""),
            "contribution_scope": metadata.get("contribution_scope", ""),
            "section": metadata.get("section", ""),
            "chunk_style": metadata.get("chunk_style", ""),
            "metadata": metadata,
        })

    return chunks


def dedupe_chunks(chunks: list[dict]) -> list[dict]:
    seen = set()
    unique = []

    for chunk in chunks:
        key = (
            chunk.get("source", ""),
            chunk.get("type", ""),
            chunk.get("section", ""),
            chunk.get("chunk_style", ""),
            chunk.get("text", "")[:280],
        )

        if key not in seen:
            seen.add(key)
            unique.append(chunk)

    return unique


def retrieve_many(searches: list[tuple[str, int, dict | None]]) -> list[dict]:
    """
    Run multiple retrieval routes and combine results.
    Each item is: (query, top_k, metadata_filter)
    """
    all_chunks = []

    for query, top_k, metadata_filter in searches:
        try:
            all_chunks.extend(
                retrieve(
                    query=query,
                    top_k=top_k,
                    metadata_filter=metadata_filter
                )
            )
        except Exception as e:
            print(f"[warn] Retrieval route failed: {e}")

    return dedupe_chunks(all_chunks)


# ── QUERY TYPE DETECTION ─────────────────────────────────────────────────────

def detect_query_types(query: str) -> dict:
    q = normalize_text(query)

    return {
        "broad": any(p in q for p in [
            "tell me about",
            "explain",
            "overview",
            "walk me through",
            "what is",
            "what does",
            "describe",
            "summary",
        ]),
        "role": any(p in q for p in [
            "role",
            "contribution",
            "contributed",
            "what did gaurav do",
            "what part",
            "worked on",
            "built",
            "responsibility",
            "responsibilities",
        ]),
        "education": any(p in q for p in [
            "education",
            "currently doing",
            "current status",
            "currently pursuing",
            "pursuing",
            "studying",
            "student",
            "college",
            "university",
            "btech",
            "b.tech",
            "degree",
            "specialization",
            "cgpa",
            "12th",
            "class 12",
            "intermediate",
            "10th",
            "class 10",
            "high school",
            "schooling",
        ]),
        "achievement": any(p in q for p in [
            "achievement",
            "achievements",
            "award",
            "awards",
            "honours",
            "honors",
            "amazon ml",
            "microsoft sefa",
            "leetcode",
            "hackathon",
            "patent",
            "publication",
            "research paper",
            "paper",
            "conference",
            "400",
            "dsa",
        ]),
        "fit": any(p in q for p in [
            "fit",
            "good fit",
            "why hire",
            "why should",
            "right fit",
            "suitable",
            "strength",
            "strengths",
            "expectation",
            "expectations",
            "from us",
            "role",
            "internship",
            "ai engineer",
            "scaler",
        ]),
        "architecture": any(p in q for p in [
            "architecture",
            "system design",
            "flow",
            "pipeline",
            "backend",
            "api",
            "database",
            "websocket",
            "socket",
            "queue",
            "worker",
            "deployment",
            "render",
            "vercel",
        ]),
        "ai": any(p in q for p in [
            "ai",
            "llm",
            "model",
            "rag",
            "prompt",
            "fine tuning",
            "fine-tuning",
            "groq",
            "ml",
            "machine learning",
            "deep learning",
            "tensorflow",
            "keras",
            "pinecone",
            "vector",
            "embedding",
        ]),
        "commit": any(p in q for p in [
            "commit",
            "commits",
            "github history",
            "git history",
            "pull request",
            "pull requests",
            "pr",
            "prs",
            "what do his commits show",
            "code contribution",
        ]),
        "project": any(p in q for p in [
            "project",
            "projects",
            "github",
            "repo",
            "repository",
            "portfolio",
            "built",
            "developed",
        ]),
        "contributed": any(p in q for p in [
            "contributed",
            "contribution",
            "team project",
            "worked with team",
            "what part did",
            "what did gaurav do in",
            "did he build alone",
            "built alone",
            "solely built",
            "contributed project",
        ]),
    }


# ── RERANKING ────────────────────────────────────────────────────────────────

def rerank_chunks(query: str, chunks: list[dict]) -> list[dict]:
    query_lower = normalize_text(query)
    qtypes = detect_query_types(query)
    detected_repo = detect_repo(query)

    def score_chunk(chunk: dict) -> float:
        text = chunk.get("text", "").lower()
        source = chunk.get("source", "")
        ctype = chunk.get("type", "")
        chunk_style = chunk.get("chunk_style", "")
        section = chunk.get("section", "").lower()

        score = float(chunk.get("score", 0))

        # Exact repo boost
        if detected_repo and source == detected_repo:
            score += 0.45

        # Priority/on-resume boosts
        if chunk.get("on_resume"):
            score += 0.08

        if chunk.get("priority") == "deep":
            score += 0.05

        # New parse_data.py resume chunk style boosts
        if chunk_style == "resume_fact":
            score += 0.18

        if chunk_style == "section_anchor":
            score += 0.12

        if chunk_style == "faq":
            score += 0.20

        # Metadata type boosts
        if ctype == "readme_section":
            score += 0.04

        if ctype == "pull_request":
            score += 0.08

        if ctype == "commits":
            if qtypes["commit"] or qtypes["role"] or qtypes["contributed"]:
                score += 0.12
            else:
                score -= 0.01

        # General resume/background boosts
        if "quick verified facts" in text:
            score += 0.35

        if "current education and status" in text:
            score += 0.35

        if "frequently asked verified answers" in text:
            score += 0.30

        if "education" in text or section == "education":
            score += 0.16

        if "graphic era hill university" in text:
            score += 0.25

        if "b.tech" in text or "btech" in text:
            score += 0.22

        if "cgpa" in text:
            score += 0.16

        if "amazon ml summer school" in text:
            score += 0.18

        if "microsoft sefa" in text:
            score += 0.14

        if "patent" in text:
            score += 0.16

        if "publication" in text or "research paper" in text:
            score += 0.16

        if "what makes gaurav a good fit" in text:
            score += 0.28

        if "expectations from internship" in text:
            score += 0.15

        # Project-level boosts
        if "project summary" in text:
            score += 0.18

        if "overview" in text:
            score += 0.14

        if "my role" in text or "gaurav's role" in text:
            score += 0.25

        if "gaurav's contribution" in text:
            score += 0.25

        if "contribution scope" in text:
            score += 0.25

        if "important technical evidence" in text:
            score += 0.20

        if "rag answer support" in text:
            score += 0.15

        if "technical stack" in text or "tech stack" in text:
            score += 0.08

        if "architecture" in text:
            score += 0.08

        if "ai integration" in text:
            score += 0.10

        if "backend" in text:
            score += 0.07

        # Query-specific boosts
        if qtypes["broad"]:
            if "summary" in text:
                score += 0.12
            if "overview" in text:
                score += 0.16
            if "project summary" in text:
                score += 0.16

        if qtypes["role"]:
            if "role" in text:
                score += 0.22
            if "contribution" in text:
                score += 0.25
            if "team project" in text:
                score += 0.12
            if "not build the entire project alone" in text or "not built alone" in text:
                score += 0.14

        if qtypes["education"]:
            if "education" in text:
                score += 0.35
            if "current education and status" in text:
                score += 0.40
            if "graphic era hill university" in text:
                score += 0.35
            if "b.tech" in text or "btech" in text:
                score += 0.30
            if "cgpa" in text:
                score += 0.20
            if "class 12" in text or "intermediate" in text:
                score += 0.16
            if "class 10" in text or "high school" in text:
                score += 0.16

        if qtypes["achievement"]:
            if "achievements" in text or section in ["achievements", "honours and awards", "honors and awards"]:
                score += 0.55
            if "amazon ml summer school" in text:
                score += 0.30
            if "microsoft sefa" in text:
                score += 0.20
            if "patent" in text:
                score += 0.25
            if "publication" in text or "research paper" in text:
                score += 0.25
            if "leetcode" in text or "400+" in text:
                score += 0.18
            if "hackathon" in text:
                score += 0.25

        if qtypes["fit"]:
            if "what makes gaurav a good fit" in text:
                score += 0.38
            if "ai engineer intern" in text:
                score += 0.22
            if "scaler" in text:
                score += 0.12
            if "production-ready ai applications" in text:
                score += 0.12
            if "rag" in text:
                score += 0.08
            if "voice agents" in text:
                score += 0.08
            if "backend" in text:
                score += 0.06
            if "ai integration" in text:
                score += 0.08

        if qtypes["architecture"]:
            if "architecture" in text:
                score += 0.22
            if "backend" in text:
                score += 0.10
            if "database" in text or "mongodb" in text or "postgresql" in text:
                score += 0.08
            if "websocket" in text or "socket.io" in text:
                score += 0.08
            if "bullmq" in text or "redis" in text:
                score += 0.08
            if "flask" in text or "fastapi" in text:
                score += 0.06
            if "deployment" in text or "render" in text or "vercel" in text:
                score += 0.08

        if qtypes["ai"]:
            if "ai integration" in text:
                score += 0.18
            if "llm" in text:
                score += 0.09
            if "prompt" in text:
                score += 0.09
            if "rag" in text:
                score += 0.10
            if "model" in text:
                score += 0.06
            if "machine learning" in text:
                score += 0.08
            if "deep learning" in text:
                score += 0.08
            if "tensorflow" in text or "keras" in text:
                score += 0.06
            if "pinecone" in text or "vector" in text:
                score += 0.08

        if qtypes["commit"]:
            if ctype == "commits":
                score += 0.35
            if ctype == "pull_request":
                score += 0.30
            if "original commit message" in text:
                score += 0.20
            if "expanded meaning" in text:
                score += 0.22
            if "important technical evidence" in text:
                score += 0.25

        if qtypes["contributed"]:
            if chunk.get("ownership") == "contributed":
                score += 0.35
            if ctype == "pull_request":
                score += 0.22
            if "contributed repository" in text:
                score += 0.20
            if "did not build the entire project alone" in text:
                score += 0.25
            if "not build the entire project alone" in text:
                score += 0.25
            if "team" in text:
                score += 0.10

        # Small exact keyword match boost from user query terms
        important_terms = [
            term for term in query_lower.split()
            if len(term) >= 5 and term not in {
                "about", "gaurav", "tell", "explain", "what", "which", "where", "their", "there"
            }
        ]

        for term in important_terms[:8]:
            if term in text:
                score += 0.025

        return score

    reranked = sorted(chunks, key=score_chunk, reverse=True)
    return reranked


# ── CONTEXT FORMATTING ──────────────────────────────────────────────────────

def format_context(chunks: list[dict]) -> str:
    if not chunks:
        return "No relevant context found."

    parts = []

    for i, chunk in enumerate(chunks, 1):
        header = (
            f"[Chunk {i} | source={chunk['source']} | "
            f"type={chunk['type']} | score={chunk['score']} | "
            f"ownership={chunk['ownership']} | on_resume={chunk['on_resume']}]"
        )

        extra = []

        if chunk.get("section"):
            extra.append(f"Section: {chunk['section']}")

        if chunk.get("chunk_style"):
            extra.append(f"Chunk Style: {chunk['chunk_style']}")

        if chunk.get("role"):
            extra.append(f"Role: {chunk['role']}")

        if chunk.get("project_type"):
            extra.append(f"Project Type: {chunk['project_type']}")

        if chunk.get("github_url"):
            extra.append(f"GitHub: {chunk['github_url']}")

        if chunk.get("contribution_scope"):
            extra.append(f"Contribution Scope: {chunk['contribution_scope']}")

        extra_text = "\n".join(extra)

        parts.append(
            f"{header}\n"
            f"{extra_text}\n\n"
            f"{chunk['text']}"
        )

    return "\n\n---\n\n".join(parts)


# ── SMART RETRIEVAL ─────────────────────────────────────────────────────────

def smart_retrieve_chunks(query: str, top_k: int = TOP_K_CHAT) -> list[dict]:
    query_lower = normalize_text(query)
    qtypes = detect_query_types(query)
    detected_repo = detect_repo(query)

    searches = []

    # 1. Repo-specific retrieval
    if detected_repo:
        expanded_repo_query = (
            f"{query} project summary overview my role Gaurav contribution contribution scope "
            f"backend work AI integration technical stack architecture design decisions challenges "
            f"tradeoffs what would improve commits pull requests important technical evidence"
        )

        searches.append((expanded_repo_query, max(top_k, 14), {"source": detected_repo}))

    # 2. Contribution/team-project retrieval
    if qtypes["contributed"] or qtypes["role"]:
        expanded_contribution_query = (
            f"{query} Gaurav contribution contribution scope role team project not built alone "
            f"backend feature work pull request commits important technical evidence"
        )

        searches.append((expanded_contribution_query, max(top_k, 12), {"ownership": "contributed"}))

    # 3. Resume/background retrieval
    resume_keywords = [
        "experience",
        "background",
        "education",
        "degree",
        "skills",
        "resume",
        "cv",
        "job",
        "role",
        "fit",
        "why should",
        "hire",
        "right person",
        "good fit",
        "ai engineer",
        "scaler",
        "projects in resume",
        "resume projects",
        "candidate",
        "profile",
        "cgpa",
        "college",
        "university",
        "technical skills",
        "currently doing",
        "current status",
        "currently pursuing",
        "pursuing",
        "studying",
        "student",
        "btech",
        "b.tech",
        "specialization",
        "12th",
        "class 12",
        "intermediate",
        "10th",
        "class 10",
        "high school",
        "schooling",
        "achievements",
        "achievement",
        "awards",
        "honours",
        "honors",
        "amazon ml summer school",
        "microsoft sefa",
        "patent",
        "publication",
        "research paper",
        "leetcode",
        "hackathon",
        "expectation",
        "expectations",
        "from us",
        "internship expectation",
        "salary expectation",
        "stipend",
    ]

    should_search_resume = (
        any(keyword in query_lower for keyword in resume_keywords)
        or qtypes["education"]
        or qtypes["achievement"]
        or qtypes["fit"]
    )

    if should_search_resume:
        expanded_resume_query = (
            f"{query} quick verified facts resume fact section anchor current education status "
            f"technical skills projects achievements fit AI Engineer Intern Scaler "
            f"RAG LLM prompt engineering backend AI integration voice agents"
        )

        searches.append((expanded_resume_query, max(top_k, 14), {"on_resume": True}))

    # 4. AI/backend/project broad retrieval across all repos
    if qtypes["ai"] or qtypes["architecture"] or qtypes["project"] or qtypes["commit"]:
        expanded_project_query = (
            f"{query} project summary overview technical evidence architecture backend AI ML "
            f"deployment WebSocket queue worker database commits pull request contribution"
        )

        searches.append((expanded_project_query, max(top_k, 14), None))

    # 5. Fallback general retrieval
    searches.append((query, max(top_k, 10), None))

    chunks = retrieve_many(searches)
    chunks = dedupe_chunks(chunks)
    chunks = rerank_chunks(query, chunks)

    return chunks[:top_k]


def smart_retrieve_context(query: str, top_k: int = TOP_K_CHAT) -> str:
    chunks = smart_retrieve_chunks(query=query, top_k=top_k)
    return format_context(chunks)


def smart_retrieve(query: str) -> str:
    return smart_retrieve_context(query)


def smart_retrieve_voice_context(query: str) -> str:
    chunks = smart_retrieve_chunks(query=query, top_k=TOP_K_VOICE)
    return format_context(chunks)


# ── CLI TEST ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Tell me about Gaurav's education"

    print(f"\nQuery: {query}")
    print("=" * 80)

    chunks = smart_retrieve_chunks(query)

    print("\nRetrieved chunks:")
    for i, chunk in enumerate(chunks, 1):
        print(
            f"{i}. source={chunk['source']} | "
            f"type={chunk['type']} | "
            f"score={chunk['score']} | "
            f"on_resume={chunk['on_resume']} | "
            f"ownership={chunk['ownership']} | "
            f"section={chunk.get('section', '')} | "
            f"chunk_style={chunk.get('chunk_style', '')}"
        )

    print("\n" + "=" * 80)
    print("Formatted context:\n")
    print(format_context(chunks))