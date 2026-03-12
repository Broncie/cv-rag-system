# ingest.py - script to load PDFs, split into chunks, create FAISS index with Ollama embeddings, and save to disk
import os
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")


def load_pdfs(pdf_dir: Path) -> List:
    docs = []
    for pdf_path in pdf_dir.glob("*.pdf"):
        loader = PyPDFLoader(str(pdf_path))
        pdf_docs = loader.load()  # one Document per page
        candidate_id = pdf_path.stem  # filename without .pdf
        for d in pdf_docs:
            d.metadata["candidate_id"] = candidate_id
            d.metadata["source_file"] = pdf_path.name
            d.metadata["doc_type"] = "cv"
            d.metadata["page"] = d.metadata.get("page", None)
        docs.extend(pdf_docs)
    return docs


def main():
    BASE_DIR = Path(__file__).resolve().parent
    pdf_dir = BASE_DIR / "data" / "docs"
    index_dir = BASE_DIR / "data" / "index"

    index_dir.mkdir(parents=True, exist_ok=True)

    if not pdf_dir.exists():
        raise FileNotFoundError(f"Missing folder: {pdf_dir.resolve()}")

    raw_docs = load_pdfs(pdf_dir)
    if not raw_docs:
        raise ValueError(f"No PDFs found in {pdf_dir.resolve()}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100,
        separators=["\n\n", "\n", "•", "-", " ", ""],
    )
    chunks = splitter.split_documents(raw_docs)

    for i, c in enumerate(chunks):
        c.metadata["chunk_id"] = i

    print("Pages loaded:", len(raw_docs))
    print("Chunks created:", len(chunks))

    for i in range(min(5, len(chunks))):
        print("\n--- CHUNK", i, "---")
        print("META:", chunks[i].metadata)
        print(chunks[i].page_content[:300])

    # Local embeddings via Ollama
    embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)

    # Build FAISS index
    db = FAISS.from_documents(chunks, embeddings)

    # Persist to disk (creates index.faiss + index.pkl)
    db.save_local(str(index_dir))

    print(f"Ingested PDFs: {len(list(pdf_dir.glob('*.pdf')))}")
    print(f"Total pages loaded: {len(raw_docs)}")
    print(f"Total chunks created: {len(chunks)}")
    print(f"Saved FAISS index to: {index_dir.resolve()}")


if __name__ == "__main__":
    main()
