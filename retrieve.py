# retrieve.py - logic for retrieving relevant document chunks from FAISS index based on query
import os
from pathlib import Path
from typing import Any, Dict, List

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

# retrieve.py
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")

def get_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)

def load_vectorstore(index_dir: Path) -> FAISS:
    return FAISS.load_local(
        str(index_dir),
        embeddings=get_embeddings(),
        allow_dangerous_deserialization=True,
    )

def retrieve(query: str, k: int = 5) -> List[Dict[str, Any]]:
    base_dir = Path(__file__).resolve().parent
    index_dir = base_dir / "data" / "index"

    if not (index_dir / "index.faiss").exists():
        raise FileNotFoundError(f"Missing FAISS index: {index_dir}/index.faiss. Run ingest.py first.")

    db = load_vectorstore(index_dir)

    # Returns List[Tuple[Document, score]] where score is distance (lower is more similar) depending on metric.
    results = db.similarity_search_with_score(query, k=k)

    out = []
    for doc, score in results:
        out.append(
            {
                "score": float(score),
                "text": doc.page_content,
                "metadata": doc.metadata,
            }
        )
    return out


if __name__ == "__main__":
    q = "Summarize this candidate's experience with Python and data engineering."
    hits = retrieve(q, k=5)

    for i, h in enumerate(hits, start=1):
        meta = h["metadata"]
        print(f"\n=== HIT {i} ===")
        print("score:", h["score"])
        print("source_file:", meta.get("source_file"))
        print("candidate_id:", meta.get("candidate_id"))
        print("page:", meta.get("page"))
        print("chunk_id:", meta.get("chunk_id"))
        print(h["text"][:500])
