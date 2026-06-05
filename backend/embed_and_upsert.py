"""
embed_and_upsert.py
Reads data/processed/parsed_chunks.json, embeds each chunk using a free local
SentenceTransformer model, and upserts into Pinecone.

Requirements:
    pip install sentence-transformers pinecone tqdm python-dotenv
"""

import json
import os
import time
import hashlib
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]

INDEX_NAME = "ai-persona-local"
NAMESPACE = "gaurav-ai-persona"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

BATCH_SIZE = 64
SLEEP_BETWEEN_BATCHES = 0.3

INPUT_FILE = Path("data/processed/parsed_chunks.json")

print(f"Loading local embedding model: {EMBEDDING_MODEL}")
embedding_model = SentenceTransformer(EMBEDDING_MODEL)


def make_id(text: str, metadata: dict) -> str:
    raw = (
        f"{metadata.get('source', '')}::"
        f"{metadata.get('type', '')}::"
        f"{metadata.get('file', '')}::"
        f"{text[:500]}"
    )
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def embed_texts(texts: list[str]) -> list[list[float]]:
    embeddings = embedding_model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False
    )
    return embeddings.tolist()


def init_pinecone_index():
    pc = Pinecone(api_key=PINECONE_API_KEY)

    existing = [idx.name for idx in pc.list_indexes()]

    if INDEX_NAME not in existing:
        print(f"Creating Pinecone index: {INDEX_NAME}")

        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

        while not pc.describe_index(INDEX_NAME).status["ready"]:
            print("  Waiting for index to be ready...")
            time.sleep(2)

    else:
        print(f"Using existing Pinecone index: {INDEX_NAME}")

    return pc.Index(INDEX_NAME)


def chunk_list(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def sanitize_metadata(metadata: dict) -> dict:
    clean = {}

    for key, value in metadata.items():
        if value is None:
            continue

        if isinstance(value, (str, int, float, bool)):
            clean[key] = value

        elif isinstance(value, list):
            clean[key] = [str(v) for v in value]

        else:
            clean[key] = str(value)

    return clean


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"{INPUT_FILE} not found. Run parse_data.py first.")

    with open(INPUT_FILE, encoding="utf-8") as f:
        chunks = json.load(f)

    if not chunks:
        raise ValueError("No chunks found. Check parse_data.py output.")

    print(f"Loaded {len(chunks)} chunks from {INPUT_FILE}")

    index = init_pinecone_index()

    total_upserted = 0

    for batch in tqdm(list(chunk_list(chunks, BATCH_SIZE)), desc="Embedding & upserting"):
        texts = [chunk["text"] for chunk in batch]
        embeddings = embed_texts(texts)

        vectors = []

        for chunk, embedding in zip(batch, embeddings):
            text = chunk["text"]
            metadata = chunk.get("metadata", {})
            vec_id = make_id(text, metadata)

            vector_metadata = sanitize_metadata({
                **metadata,
                "text": text[:3500]
            })

            vectors.append({
                "id": vec_id,
                "values": embedding,
                "metadata": vector_metadata
            })

        index.upsert(vectors=vectors, namespace=NAMESPACE)
        total_upserted += len(vectors)

        time.sleep(SLEEP_BETWEEN_BATCHES)

    print(f"\n✅ Done. Total vectors upserted: {total_upserted}")
    print(f"   Index: {INDEX_NAME}")
    print(f"   Namespace: {NAMESPACE}")

    stats = index.describe_index_stats()
    print(f"   Pinecone index stats: {stats}")


if __name__ == "__main__":
    main()