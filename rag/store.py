"""Chroma vector store: one collection per target, embedded with Gemini (gemini-embedding-001)."""
import os
import re
import time
from typing import Any, Dict, List, Optional

import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from dotenv import load_dotenv

load_dotenv()

INDEX_DIR = "output/rag_index"
BATCH = 100
EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-001")
EMBED_DIM = 768


class GeminiEmbeddings(EmbeddingFunction):
    """Calls Gemini's OpenAI-compatible embeddings endpoint, retrying on rate limits (free tier)."""

    def __init__(self, model: str = EMBED_MODEL, dim: int = EMBED_DIM):
        self.model, self.dim = model, dim

    def __call__(self, input: Documents) -> Embeddings:
        from openai import OpenAI, RateLimitError

        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=os.getenv("OPENAI_BASE_URL"))
        out = []
        for i in range(0, len(input), BATCH):
            for attempt in range(5):
                try:
                    r = client.embeddings.create(model=self.model, input=list(input[i:i + BATCH]), dimensions=self.dim)
                    out.extend(d.embedding for d in r.data)
                    break
                except RateLimitError:
                    time.sleep(2 ** attempt * 5)
            else:
                raise RuntimeError("Gemini embeddings kept hitting the rate limit")
        return out

    @staticmethod
    def name() -> str:
        return "gemini-openai-compat"

    def get_config(self) -> Dict[str, Any]:
        return {"model": self.model, "dim": self.dim}

    @staticmethod
    def build_from_config(config: Dict[str, Any]) -> "GeminiEmbeddings":
        return GeminiEmbeddings(config["model"], config["dim"])


def _client(path: Optional[str]):
    return chromadb.EphemeralClient() if path is None else chromadb.PersistentClient(path=path)


def collection_name(target: str) -> str:
    return "t_" + re.sub(r"[^a-z0-9_-]", "_", target.lower())[:50]


def get_collection(target: str, path: Optional[str] = INDEX_DIR, embedder: Optional[EmbeddingFunction] = None):
    return _client(path).get_or_create_collection(
        collection_name(target), embedding_function=embedder or GeminiEmbeddings(), metadata={"hnsw:space": "cosine"})


def index_chunks(target: str, chunks: List[Dict[str, Any]], path: Optional[str] = INDEX_DIR,
                 embedder: Optional[EmbeddingFunction] = None):
    """Rebuild the target's collection from scratch so re-runs don't leave stale chunks. Returns the collection."""
    client = _client(path)
    name = collection_name(target)
    if name in [c if isinstance(c, str) else c.name for c in client.list_collections()]:
        client.delete_collection(name)
    col = client.get_or_create_collection(name, embedding_function=embedder or GeminiEmbeddings(),
                                          metadata={"hnsw:space": "cosine"})
    for i in range(0, len(chunks), BATCH):
        batch = chunks[i:i + BATCH]
        col.add(
            ids=[c["chunk_id"] for c in batch],
            documents=[c["text"] for c in batch],
            metadatas=[{k: c[k] for k in ("source_id", "source", "title", "year", "phase")} for c in batch],
        )
    return col


def retrieve(col, question: str, k: int = 5, source: Optional[str] = None) -> List[Dict[str, Any]]:
    res = col.query(query_texts=[question], n_results=min(k, col.count()),
                    where={"source": source} if source else None)
    return [
        {"chunk_id": cid, "text": doc, **meta, "score": round(1 - dist, 3)}
        for cid, doc, meta, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0])
    ]
