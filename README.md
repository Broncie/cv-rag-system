# Local CV RAG System
A fully local Retrieval-Augmented Generation (RAG) application for querying and analyzing CVs using FastAPI, Ollama, FAISS, and Docker.

## Introduction
This repository contains a fully local RAG system designed for querying indexed CVs through a simple API and interactive debug interface for inspecting retrieval results, citations, groundedness scores, and raw model responses.

PDF CVs are ingested locally, split into chunks, converted into embeddings using Ollama, and stored in a FAISS vector index. When a question is asked, the system retrieves the most relevant chunks, groups them by candidate, and uses a local LLM to generate answers grounded in the retrieved evidence.

The project is intended as a portfolio project to demonstrate practical skills in:
- LLM application development
- Retrieval-Augmented Generation (RAG)
- Vector similarity search with FAISS
- FastAPI backend development
- Docker-based local deployment
- Grounded answer verification

## Features
- Fully local RAG pipeline (no cloud APIs required)
- PDF ingestion and chunking pipeline
- Local embeddings using Ollama
- FAISS vector search
- LLM-based answer generation
- Citation-aware prompting
- Groundedness verification of generated answers
- Interactive debug UI for inspecting retrieval behavior
- Dockerized deployment
- Candidate-aware context construction that groups retrieved chunks by resume

## Architecture
![RAG  Architecture](RAG.png)

## Technologies Used
1. Programming Language - Python  
2. API Framework - FastAPI  
3. LLM Runtime - Ollama  
4. Embedding Model - `nomic-embed-text`
5. Chat Model - `llama3.1`
6. Vector Store - FAISS  
7. PDF Loading - PyPDF  
8. Containerization & Environment Management - Docker & Docker Compose  
9. Frontend - HTML, CSS, JavaScript

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
## How It Works
1. **Ingestion**  
   PDF CVs are loaded from the local `data/docs` folder and split into smaller text chunks.

2. **Embedding**  
   Each chunk is converted into a vector embedding using a local embedding model running through Ollama.

3. **Indexing**  
   The embedded chunks are stored in a FAISS vector index for efficient similarity search.

4. **Retrieval**  
   When a user asks a question, the system retrieves the most relevant chunks from the FAISS index.

5. **Candidate Grouping**  
   Retrieved chunks are grouped by `candidate_id` so the system reasons over candidate-level evidence rather than isolated chunk fragments.

6. **Candidate Summarization**  
   Retrieved evidence for each candidate is summarized into concise bullet points grounded in the original chunks.

7. **Generation**  
   The LLM answers the user’s question using the candidate summaries while preserving chunk-level citations.

8. **Verification**  
   The generated answer is checked using a heuristic grounding verifier to ensure cited sentences are supported by the retrieved chunks.

## Example Workflow
1. Place CV PDFs in `data/docs/`
2. Start the services with Docker
3. Pull the required Ollama models
4. Run ingestion to create the FAISS index
5. Open the local UI
6. Ask questions such as:
   - Which candidates mention Python?
   - Who has data engineering experience?
   - Does any candidate have a Master's degree?

## Project Structure
```text
.
├── app.py               # Main FastAPI application
├── ingest.py            # Loads PDFs, chunks text, creates embeddings, saves FAISS index
├── retrieve.py          # Retrieval logic for querying the vector store
├── verify.py            # Heuristic grounding verification
├── requirements.txt     # Python dependencies
├── Dockerfile           # Docker image definition for the API
├── docker-compose.yml   # Multi-service local setup
├── static/
│   └── index.html       # Local debug interface
└── data/
    ├── docs/            # Local PDF CVs (not intended for GitHub)
    └── index/           # Generated FAISS index files
```

## Dataset Used
This project is designed to work with local PDF CVs provided by the user.

The system does not rely on a public dataset by default. Instead, it indexes CV documents placed in the `data/docs` directory.

## Setup Instructions
1. Clone the repository.

2. Start the services.
```bash
docker compose up --build
```

3. Pull the required Ollama models if needed:
```bash
docker exec -it ollama ollama pull llama3.1
docker exec -it ollama ollama pull nomic-embed-text
```

4. Place PDF CVs inside:
`data/docs/`

5. Run ingestion to build the FAISS vector index (inside the API container):
```bash
docker compose exec rag_api python ingest.py
```
6. Open the interface in the browser:
`http://localhost:8000`

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
Returns:
- generated answer
- retrieved citations
- confidence estimate
- groundedness score
- unsupported sentences
- number of retrieval attempts

## Example Use Cases
- Query CVs semantically instead of manually reading each one
- Compare candidates based on technical skills
- Identify experience with specific tools, frameworks, or domains
- Demonstrate a local end-to-end RAG pipeline as a portfolio project

## Notes
- This project is intended for local use.
- It uses local models through Ollama, meaning no cloud LLM API is required.
- The current interface is a debug-oriented frontend for testing retrieval quality, groundedness, and citations.

## Data Privacy
This project is designed to work with local CV documents.

Do **not** commit real CVs or generated vector indexes to GitHub.  
The `data/docs` and `data/index` folders are intended for local use only and should be excluded using `.gitignore`.

## Current Limitations
- Retrieval is limited to the indexed PDF content and may miss information if the relevant text is not retrieved in the top-k results.
- Grounding verification is heuristic-based and does not perform full semantic entailment.
- The interface is intended for debugging and inspection rather than production use.
- The system currently works best on small local CV collections.

## Future Improvements
- migrate frontend to React
- improve answer verification logic
- add stronger input validation and security hardening
- support metadata filtering
- add better prompt orchestration and agentic workflows
- improve document management and ingestion pipeline
- add authentication if deployed beyond localhost