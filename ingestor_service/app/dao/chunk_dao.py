from typing import Dict, Any, List
from app.clients.mongo_client import MongoDBClient
from app.config.costants import MONGO_DB_NAME


class ChunkDAO:
    def __init__(self):
        self.collection = MongoDBClient().get_collection(MONGO_DB_NAME)

    def upsert_chunk(self, chunk: Dict[str, Any]):
        self.collection.update_one(
            {"chunk_id": chunk["chunk_id"]},
            {"$setOnInsert": {
                "chunk_id": chunk["chunk_id"],
                "document_id": chunk.get("document_id"),
                "ingest_status": chunk.get("ingest_status", "PENDING"),
                "graph_status": chunk.get("graph_status", "PENDING"),
                "summary_status": chunk.get("summary_status", "PENDING")
            }},
            upsert=True
        )
