# app.py - main FastAPI app for CV RAG system
import os

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage
from verify import verify_grounding
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from collections import defaultdict
from typing import DefaultDict

app = FastAPI(title="CV RAG API", version="0.1.0")

# Add these right after app = FastAPI(...)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_private_network=True,
)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def serve_ui():
    return FileResponse("static/index.html")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
CHAT_MODEL = os.getenv("CHAT_MODEL", "llama3.1")

def get_embeddings():
    return OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)

def get_llm():
    return ChatOllama(model=CHAT_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.0, num_predict = 200, repeat_penalty=1.2)

_DB: Optional[FAISS] = None


class AskRequest(BaseModel):
    question: str
    k: int = 5

class AnswerResponse(BaseModel):
    answer: str
    citations: List[Dict[str, Any]]
    confidence: float
    groundedness: float
    unsupported_sentences: List[str]
    attempts: int


def load_vectorstore(index_dir: Path) -> FAISS:
    return FAISS.load_local(
        str(index_dir),
        embeddings=get_embeddings(),
        allow_dangerous_deserialization=True,
    )

def get_db(index_dir: Path) -> FAISS:
    global _DB
    if _DB is None:
        _DB = load_vectorstore(index_dir)
    return _DB

def group_results_by_candidate(results: List[Tuple[Any, float]]) -> Dict[str, List[Tuple[Any, float]]]:
    grouped: DefaultDict[str, List[Tuple[Any, float]]] = defaultdict(list)

    for doc, score in results:
        meta = doc.metadata or {}
        candidate_id = meta.get("candidate_id", "unknown_candidate")
        grouped[candidate_id].append((doc, score))

    return dict(grouped)


def build_grouped_context(
    results: List[Tuple[Any, float]]
) -> Tuple[str, List[Dict[str, Any]], Dict[int, str]]:
    grouped = group_results_by_candidate(results)

    context_blocks: List[str] = []
    citations: List[Dict[str, Any]] = []
    chunk_text_by_id: Dict[int, str] = {}

    for candidate_id, candidate_results in grouped.items():
        candidate_results = sorted(
            candidate_results,
            key=lambda x: (
                (x[0].metadata or {}).get("page", 10**9),
                (x[0].metadata or {}).get("chunk_id", 10**9),
            )
        )

        first_doc = candidate_results[0][0]
        first_meta = first_doc.metadata or {}
        source_file = first_meta.get("source_file", "unknown_source")

        candidate_block_lines = [
            f"<candidate id={candidate_id} source={source_file}>"
        ]

        for doc, score in candidate_results:
            meta = doc.metadata or {}
            chunk_id = meta.get("chunk_id")
            page = meta.get("page")

            candidate_block_lines.append(
                f"<chunk id={chunk_id} candidate_id={candidate_id} source={source_file} page={page}>"
            )
            candidate_block_lines.append(doc.page_content)
            candidate_block_lines.append("</chunk>")

            citations.append({
                "chunk_id": chunk_id,
                "candidate_id": candidate_id,
                "source_file": source_file,
                "page": page,
                "distance": float(score),
            })

            if isinstance(chunk_id, int):
                chunk_text_by_id[chunk_id] = doc.page_content

        candidate_block_lines.append("</candidate>")
        context_blocks.append("\n".join(candidate_block_lines))

    return "\n\n".join(context_blocks), citations, chunk_text_by_id

def rewrite_query(question: str) -> str:
    try:
        prompt = f"""Rewrite the user question into a short keyword-style search query.
Return ONLY the rewritten query, no explanation.
Question: {question}"""
        result = get_llm().invoke([HumanMessage(content=prompt)]).content.strip()
        return result if result else question  # fallback to original
    except Exception:
        return question


@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ask", response_model=AnswerResponse)
async def ask(req: AskRequest) -> AnswerResponse:
    base_dir = Path(__file__).resolve().parent
    index_dir = base_dir / "data" / "index"

    if not (index_dir / "index.faiss").exists():
        raise HTTPException(status_code=400, detail="Missing FAISS index. Run ingest first.")

    db = get_db(index_dir)

    attempts = 2
    ks = [req.k, max(req.k, 10)]

    last_answer = ""
    last_citations: List[Dict[str, Any]] = []
    last_groundedness = 0.0
    last_unsupported: List[str] = []

    for i in range(attempts):
        k = ks[i]
        rq = rewrite_query(req.question)
        results = db.similarity_search_with_score(rq, k=k)

        context, citations, chunk_text_by_id = build_grouped_context(results)
        answer = draft_answer(req.question, context)

        groundedness, unsupported = verify_grounding(answer, chunk_text_by_id)

        last_answer = answer
        last_citations = citations
        last_groundedness = groundedness
        last_unsupported = unsupported

        # Only treat it as a true refusal if the whole answer is exactly the refusal string
        if answer.strip() == "I do not have enough information in the provided documents.":
            return AnswerResponse(
                answer=answer,
                citations=[],
                confidence=0.2,
                groundedness=1.0,
                unsupported_sentences=[],
                attempts=i + 1,
            )

        # Accept moderately grounded answers instead of forcing all-or-nothing refusal
        if groundedness >= 0.6:
            confidence = 0.4 + 0.5 * groundedness
            return AnswerResponse(
                answer=answer,
                citations=citations,
                confidence=float(min(confidence, 0.95)),
                groundedness=float(groundedness),
                unsupported_sentences=unsupported,
                attempts=i + 1,
            )

    # Return the best available answer instead of collapsing everything into refusal
    if last_answer:
        confidence = 0.25 + 0.4 * last_groundedness
        return AnswerResponse(
            answer=last_answer,
            citations=last_citations,
            confidence=float(min(confidence, 0.6)),
            groundedness=float(last_groundedness),
            unsupported_sentences=last_unsupported[:5],
            attempts=attempts,
        )

    return AnswerResponse(
        answer="I do not have enough information in the provided documents.",
        citations=[],
        confidence=0.2,
        groundedness=float(last_groundedness),
        unsupported_sentences=last_unsupported[:5],
        attempts=attempts,
    )


def draft_answer(question: str, context: str) -> str:
    prompt = f"""
You are a strict retrieval-based assistant helping a recruiter analyse candidate CVs.

You are given grouped resume evidence in <candidate> blocks.
Each <candidate> block corresponds to one candidate.
Each <chunk> inside a candidate block is a fragment of that candidate's resume.

Important rules:
- Use ONLY the provided context.
- Never treat chunk_id as a candidate identifier.
- Use candidate_id, source_file, or an explicit candidate name if it appears in the text.
- If multiple chunks belong to the same candidate_id, they refer to the same candidate.

Critical answering rules:
- Every factual sentence MUST end with one or more citations like [chunk_id=12] or [chunk_id=12,45].
- Use ONLY chunk_id values that appear in the provided context.
- Do NOT include any factual sentence you cannot cite.
- Prefer concrete statements that stay very close to the wording in the retrieved text.
- Do NOT write meta-sentences such as:
  "Based on the provided context"
  "This indicates that"
  "However, I can provide"
  "It appears that"
- Do NOT generalize into broad labels such as:
  "machine learning background"
  "data engineering experience"
  "most senior candidate"
  unless the retrieved text explicitly supports that exact phrasing.
- If the question cannot be answered directly, say exactly:
  "I do not have enough information in the provided documents."

Question-specific rules:
- If asked who has experience with something, mention only candidates where the retrieved text explicitly mentions that skill, tool, course, project, or technology.
- If asked for names, provide a name only if it explicitly appears in the retrieved text.
- If asked to summarize a candidate, summarize only explicit education, projects, work, and listed skills from the retrieved text.
- If asked to compare years of experience, do not calculate durations unless the retrieved text clearly provides sufficient dates for a reliable comparison.

Return a concise plain-text answer only.

Context:
{context}

Question:
{question}
""".strip()

    llm = get_llm()
    resp = llm.invoke([HumanMessage(content=prompt)])
    return resp.content.strip()