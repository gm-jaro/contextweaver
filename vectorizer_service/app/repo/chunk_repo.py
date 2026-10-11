from app.dao.chunk_dao import ChunkDAO
from app.models.chunk import IngestChunk
from typing import List, Dict, Any, Optional


class ChunkRepository:
    def __init__(self):
        self.dao = ChunkDAO()

    def ensure_chunk_exists(self, chunk: IngestChunk):
        self.dao.upsert(chunk.model_dump())

    def update_chunk_summary(self, chunk_id: str, summary: str):
        result = self.dao.update_summary(chunk_id=chunk_id, summary=summary)
        if result.matched_count == 0:
            raise RuntimeError(f"Chunk {chunk_id} not found in Mongo")

    def update_chunk_qa_pairs(self, chunk_id: str, qa_pairs: List[Dict[str, Any]]):
        result = self.dao.update_qa_pairs(chunk_id=chunk_id, qa_pairs=qa_pairs)
        if result.matched_count == 0:
            raise RuntimeError(f"Chunk {chunk_id} not found in Mongo")

    def update_chunk_ingest_status(self, chunk_id: str, status: str):
        self.dao.collection.update_one({"chunk_id": chunk_id}, {"$set": {"ingest_status": status}})

    def update_chunk_graph_status(self, chunk_id: str, status: str):
        self.dao.collection.update_one({"chunk_id": chunk_id}, {"$set": {"graph_status": status}})

    def update_chunk_summary_status(self, chunk_id: str, status: str):
        self.dao.collection.update_one({"chunk_id": chunk_id}, {"$set": {"summary_status": status}})

    def get_chunks_count(self, document_id: str) -> int:
        return self.dao.collection.count_documents({"document_id": document_id})

    def get_ready_chunks_count(self, document_id: str) -> int:
        return self.dao.collection.count_documents({
            "document_id": document_id,
            "ingest_status": "DONE",
            "graph_status": "DONE"
        })

    def get_chunks(self, document_id: str):
        return list(self.dao.collection.find({"document_id": document_id}))