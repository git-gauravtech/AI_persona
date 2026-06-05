"""
retrieve.py
Retrieval layer for Gaurav AI Persona.

Uses:
- Local SentenceTransformer embeddings
- Pinecone vector DB
- Metadata-aware smart routing
- Light reranking for better answer quality

Used by:
- chat_api.py
- future voice endpoint / Vapi tool
"""

import os
import re
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

load_dotenv()

# ── CONFIG ───────────────────────────────────────────────────────────────────
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]

INDEX_NAME = "ai-persona-local"
NAMESPACE = "gaurav-ai-persona"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Chat can use more context.
TOP_K_CHAT = 6

# Voice should use fewer chunks for latency and concise answers.
TOP_K_VOICE = 3
# ─────────────────────────────────────────────────────────────────────────────

print(f"Loading retrieval embedding model: {EMBEDDING_MODEL}")
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)


# ── PROJECT / REPO ALIASES ────────────────────────────────────────────────────
# These source names must match metadata["source"] in Pinecone.
# This is routing logic only, not answer hardcoding.

REPO_ALIASES = {
    "vocalis-ai": [
        "vocalis",
        "vocalis-ai",
        "intelligent ai interview",
        "interview simulation",
        "mock interview",
        "ai interview",
        "voice interview",
        "resume based interview",
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
    ],
    "Indoor-plant-health-detection": [
        "indoor plant",
        "plant health",
        "plant health detection",
        "leaf health",
        "mobilenet",
        "grad-cam",
        "plant disease",
        "streamlit plant",
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


def normalize_text(text: str) -> str:
    """Lowercase and normalize whitespace."""
    return re.sub(r"\s+", " ", text.lower()).strip()


def detect_repo(query: str) -> str | None:
    """Detect if the query refers to a known project/repo."""
    q = normalize_text(query)

    for source, aliases in REPO_ALIASES.items():
        for alias in aliases:
            if alias.lower() in q:
                return source

    return None


def embed_query(query: str) -> list[float]:
    """Embed a single query using local embedding model."""
    embedding = embedding_model.encode(
        query,
        normalize_embeddings=True,
        show_progress_bar=False
    )
    return embedding.tolist()


def retrieve(
    query: str,
    top_k: int = TOP_K_CHAT,
    metadata_filter: dict | None = None
) -> list[dict]:
    """
    Retrieve relevant chunks from Pinecone.
    """
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
            "metadata": metadata,
        })

    return chunks


def dedupe_chunks(chunks: list[dict]) -> list[dict]:
    """Remove duplicate chunks by source/type/text prefix."""
    seen = set()
    unique = []

    for chunk in chunks:
        key = (
            chunk.get("source", ""),
            chunk.get("type", ""),
            chunk.get("text", "")[:250]
        )

        if key not in seen:
            seen.add(key)
            unique.append(chunk)

    return unique


def rerank_chunks(query: str, chunks: list[dict]) -> list[dict]:
    """
    Light reranking to improve broad project answers.
    """
    query_lower = normalize_text(query)

    broad_question = any(
        phrase in query_lower
        for phrase in [
            "tell me about",
            "explain",
            "overview",
            "walk me through",
            "what is",
            "what does",
            "describe",
        ]
    )

    role_question = any(
        phrase in query_lower
        for phrase in [
            "role",
            "contribution",
            "contributed",
            "what did gaurav do",
            "what part",
            "worked on",
            "built",
            "responsibility",
        ]
    )

    architecture_question = any(
        phrase in query_lower
        for phrase in [
            "architecture",
            "system design",
            "flow",
            "pipeline",
            "backend",
            "api",
            "database",
            "websocket",
            "queue",
            "worker",
            "deployment",
        ]
    )

    ai_question = any(
        phrase in query_lower
        for phrase in [
            "ai",
            "llm",
            "model",
            "rag",
            "prompt",
            "fine tuning",
            "fine-tuning",
            "gemini",
            "groq",
            "ml",
            "machine learning",
            "deep learning",
            "tensorflow",
            "keras",
        ]
    )

    def score_chunk(chunk: dict) -> float:
        text = chunk.get("text", "").lower()
        score = float(chunk.get("score", 0))

        # Strong section boosts
        if "overview" in text:
            score += 0.25
        if "my role" in text:
            score += 0.35
        if "gaurav's contribution" in text:
            score += 0.30
        if "contribution scope" in text:
            score += 0.30
        if "problem it solves" in text:
            score += 0.08
        if "technical stack" in text or "tech stack" in text:
            score += 0.07
        if "architecture" in text:
            score += 0.08
        if "ai integration" in text:
            score += 0.08
        if "backend work" in text:
            score += 0.08
        if "design decisions" in text:
            score += 0.05
        if "challenges" in text:
            score += 0.04
        if "tradeoffs" in text or "trade-offs" in text:
            score += 0.04
        if "what i would improve" in text or "what gaurav would improve" in text:
            score += 0.04

        # Query-specific boosts
        if broad_question:
            if "overview" in text:
                score += 0.25
            if "my role" in text:
                score += 0.25
            if "problem it solves" in text:
                score += 0.08

        if role_question:
            if "my role" in text:
                score += 0.25
            if "contribution scope" in text:
                score += 0.22
            if "gaurav's contribution" in text:
                score += 0.22
            if "backend" in text:
                score += 0.05
            if "ai integration" in text:
                score += 0.05

        if architecture_question:
            if "architecture" in text:
                score += 0.22
            if "backend" in text:
                score += 0.08
            if "database" in text:
                score += 0.06
            if "websocket" in text or "socket.io" in text:
                score += 0.06
            if "bullmq" in text or "redis" in text:
                score += 0.06
            if "flask" in text:
                score += 0.05

        if ai_question:
            if "ai integration" in text:
                score += 0.18
            if "llm" in text:
                score += 0.08
            if "prompt" in text:
                score += 0.08
            if "model" in text:
                score += 0.06
            if "machine learning" in text:
                score += 0.06
            if "deep learning" in text:
                score += 0.06
            if "tensorflow" in text or "keras" in text:
                score += 0.05

        # Prefer README and PR/contribution notes over noisy commit batches.
        if chunk.get("type") == "readme_section":
            score += 0.03
        if chunk.get("type") == "pull_request":
            score += 0.04
        if chunk.get("type") == "commits":
            score -= 0.02

        return score

    return sorted(chunks, key=score_chunk, reverse=True)


def format_context(chunks: list[dict]) -> str:
    """
    Format retrieved chunks into context for LLM.
    """
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


def smart_retrieve_chunks(query: str, top_k: int = TOP_K_CHAT) -> list[dict]:
    """
    Smart metadata-aware retrieval.

    Priority:
    1. Specific repo/project mentioned
    2. Contribution/team project questions
    3. Resume/background/fit questions
    4. General search
    """
    query_lower = normalize_text(query)

    # 1. Repo-specific routing
    detected_repo = detect_repo(query)
    if detected_repo:
        expanded_query = (
            f"{query} overview my role Gaurav contribution contribution scope "
            f"backend work AI integration technical stack architecture problem it solves "
            f"design decisions challenges tradeoffs what would improve"
        )

        chunks = retrieve(
            query=expanded_query,
            top_k=max(top_k, 10),
            metadata_filter={"source": detected_repo}
        )

        if chunks:
            chunks = dedupe_chunks(chunks)
            chunks = rerank_chunks(query, chunks)
            return chunks[:top_k]

    # 2. Contributed project routing
    contributed_keywords = [
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
    ]

    if any(keyword in query_lower for keyword in contributed_keywords):
        expanded_query = (
            f"{query} Gaurav contribution contribution scope role "
            f"team project not built alone backend feature work"
        )

        chunks = retrieve(
            query=expanded_query,
            top_k=max(top_k, 8),
            metadata_filter={"ownership": "contributed"}
        )

        if chunks:
            chunks = dedupe_chunks(chunks)
            chunks = rerank_chunks(query, chunks)
            return chunks[:top_k]

    # 3. Resume / background / fit routing
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
    ]

    if any(keyword in query_lower for keyword in resume_keywords):
        expanded_query = (
            f"{query} resume education skills projects achievements "
            f"AI engineer intern fit backend AI integration machine learning RAG"
        )

        chunks = retrieve(
            query=expanded_query,
            top_k=max(top_k, 10),
            metadata_filter={"on_resume": True}
        )

        if chunks:
            chunks = dedupe_chunks(chunks)
            chunks = rerank_chunks(query, chunks)
            return chunks[:top_k]

    # 4. General search across everything
    chunks = retrieve(query=query, top_k=top_k)
    chunks = dedupe_chunks(chunks)
    return rerank_chunks(query, chunks)


def smart_retrieve_context(query: str, top_k: int = TOP_K_CHAT) -> str:
    chunks = smart_retrieve_chunks(query=query, top_k=top_k)
    return format_context(chunks)


# Backward-compatible alias for chat_api.py
def smart_retrieve(query: str) -> str:
    return smart_retrieve_context(query)


def smart_retrieve_voice_context(query: str) -> str:
    """
    Smaller context for voice agent.
    Use fewer chunks to reduce latency and keep spoken answers short.
    """
    chunks = smart_retrieve_chunks(query=query, top_k=TOP_K_VOICE)
    return format_context(chunks)


# ── CLI TEST ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Tell me about Gaurav's AI projects"

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
            f"ownership={chunk['ownership']}"
        )

    print("\n" + "=" * 80)
    print("Formatted context:\n")
    print(format_context(chunks))