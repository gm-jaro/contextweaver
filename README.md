# ContextWeaver

Graph RAG system for Polish documents, built for my master's thesis. Three FastAPI services:

- `ingestor_service` - takes PDFs or text, converts them with Docling (optional OCR), splits into chunks
  and stores them in MongoDB
- `lm_service` - model layer: embeddings, reranking, NER, knowledge graph extraction (nodes, relations,
  deduplication), question answering, summaries and OCR with a VLM; models run through Ollama or OpenAI
- `vectorizer_service` - indexing and retrieval: hybrid search in Qdrant, knowledge graph in Neo4j,
  background workers, and a LangGraph agent that curates the graph (merges duplicates, fixes relations)

`lm_service/eval` and the `experiments` folders hold evaluation and experiment notebooks (outputs removed).
Infrastructure is defined in `vectorizer_service/docker/docker-compose.yml`. Passwords come from environment
variables: copy `.env.example` to `.env`, fill it in, and start the stack from that folder with
`docker compose --env-file ../../.env up`.
