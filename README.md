# Local CV RAG System
A fully local Retrieval-Augmented Generation (RAG) application for querying and analyzing CVs using FastAPI, Ollama, FAISS, and Docker.

## Introduction
This repository contains a fully local RAG system designed for querying indexed CVs through a FastAPI-based REST interface and an interactive debug UI for inspecting retrieval results, citations, groundedness scores, and raw model responses.

PDF CVs are ingested locally, split into chunks, converted into embeddings using Ollama, and stored in a FAISS vector index. At query time, the system retrieves relevant chunks, aggregates them at candidate level, and generates answers grounded in the retrieved evidence.

The project demonstrates practical skills in:
- LLM application development
- Retrieval-Augmented Generation (RAG)
- Vector similarity search with FAISS
- FastAPI backend development
- Docker-based local deployment
- Grounded answer verification

## Motivation
This project was built to explore how local LLM systems can be used for structured document analysis while maintaining full data privacy. It focuses on improving answer reliability through retrieval grounding and post-generation verification.

## Features
- Fully local RAG pipeline (no external APIs required)
- PDF ingestion and chunking pipeline
- Local embeddings using Ollama
- FAISS vector similarity search
- Candidate-level aggregation of retrieved evidence
- Citation-aware answer generation
- Groundedness verification and hallucination detection
- Interactive debug interface for inspecting retrieval behavior
- Fully containerized deployment with Docker

## Architecture
![RAG  Architecture](RAG.png)

The system is divided into two main pipelines:

- **Indexing pipeline (offline)**: Handles PDF ingestion, chunking, embedding generation, and FAISS indexing.
- **Query pipeline (online)**: Handles retrieval, candidate-level aggregation, answer generation, and grounding verification.

Key design decisions:
- Retrieval is performed at chunk level but aggregated at candidate level to improve reasoning.
- Groundedness verification is applied post-generation to detect unsupported claims.
- All components run locally via Docker to ensure reproducibility and data privacy.

## Technologies Used
- **Programming Language:** Python  
- **API Framework:** FastAPI  
- **LLM Runtime:** Ollama  
- **Embedding Model:** `nomic-embed-text`  
- **Chat Model:** `llama3.1`  
- **Vector Store:** FAISS  
- **PDF Loading:** PyPDF  
- **Containerization:** Docker & Docker Compose  
- **Frontend:** HTML, CSS, JavaScript  

## Quick Start
```bash
git clone <repo>
cd <repo>

docker compose up --build

docker exec -it ollama ollama pull llama3.1
docker exec -it ollama ollama pull nomic-embed-text

docker compose exec rag_api python ingest.py

Then open: http://localhost:8000
```
## System Pipeline
1. **Ingestion**  
   PDF CVs are loaded from the local `data/docs` folder and split into smaller text chunks.

2. **Embedding**  
   Each chunk is converted into a vector embedding using a local embedding model via Ollama.

3. **Indexing**  
   The embedded chunks are stored in a FAISS vector index for efficient similarity search.

4. **Retrieval**  
   Given a query, the system retrieves the most relevant chunks from the index.

5. **Candidate Aggregation**  
   Retrieved chunks are grouped by `candidate_id` to enable reasoning at candidate level instead of isolated fragments.

6. **Candidate Summarization**  
   Evidence for each candidate is summarized into structured bullet points grounded in retrieved chunks.

7. **Generation**  
   The LLM generates an answer based on candidate summaries while preserving citations.

8. **Verification**  
   A post-processing step evaluates whether the generated answer is supported by retrieved evidence.

## Groundedness Verification
The system includes a post-generation verification step that evaluates whether each sentence in the generated response is supported by retrieved evidence.

This is implemented using:
- Sentence-level comparison between generated output and retrieved chunks
- Heuristic scoring based on lexical overlap and citation alignment
- Detection of unsupported or hallucinated statements

The verifier returns:
- groundedness score
- unsupported sentences
- confidence estimate

## Example Output
Query:
"Which candidates have experience with Python and data engineering?"

Response:
- Candidate A: Experience with Python, ETL pipelines, and Airflow.
- Candidate B: Worked with Python for data processing and analytics.

Groundedness Score: 0.87  
Confidence: 0.78  

Unsupported Sentences:
- "Candidate B has production experience with Kubernetes"

## Example Workflow
1. Place CV PDFs in `data/docs/`
2. Start the services with Docker
3. Pull the required Ollama models
4. Run ingestion to build FAISS index
5. Open the local UI
6. Ask questions such as:
   - Which candidates mention Python?
   - Who has data engineering experience?
   - Does any candidate have a Master's degree?

## Project Structure
```text
.
├── app.py               # FastAPI application
├── ingest.py            # PDF ingestion, chunking, embeddings, indexing
├── retrieve.py          # Vector retrieval logic
├── verify.py            # Groundedness verification logic
├── requirements.txt     # Python dependencies
├── Dockerfile           # API container definition
├── docker-compose.yml   # Multi-service setup
├── static/
│   └── index.html       # Debug interface
└── data/
    ├── docs/            # Local CV PDFs (excluded from GitHub)
    └── index/           # FAISS index files (excluded from GitHub)
```

## API Endpoints
GET /: Serves the local debug interface.

GET /health: Returns API health status.

POST /ask: Accepts a JSON request with:

```json
{
  "question": "Which candidates have experience with Python and data engineering?",
  "k": 5
}
```
Response includes:
- Generated answer
- Retrieved citations
- Confidence estimate
- Groundedness score
- Unsupported sentences
- Number of retrieval attempts

## Performance Notes
- Optimized for small to medium document collections (10–100 CVs)
- Retrieval latency depends on embedding model and index size
- Fully local execution avoids network overhead but is limited by hardware

## Data Privacy
This project is designed for local document processing.
Do not commit real CVs or generated vector indexes to GitHub.
Ensure the following are excluded via .gitignore:
`data/docs/`
`data/index/`

## Current Limitations
- Retrieval is limited to indexed content and may miss relevant information outside top-k results
- Grounding verification is heuristic-based and does not perform full semantic entailment
- Debug UI is not production-ready
- Performance is optimized for smaller datasets

## Future Improvements
- Migrate frontend to React
- Improve answer verification using semantic entailment models
- Add metadata filtering and structured queries
- Improve prompt orchestration and agent-based workflows
- Enhance ingestion pipeline and document management
- Add authentication for non-local deployment
