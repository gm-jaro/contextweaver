from typing import Dict, Any, List
from app.clients.mongo_client import MongoDBClient
from app.config.costants import MONGO_DB_NAME


class ChunkDAO:
    def __init__(self):
        self.collection = MongoDBClient().get_collection(MONGO_DB_NAME)

    def update_summary(self, chunk_id: str, summary: str):
        return self.collection.update_one({"chunk_id": chunk_id}, {"$set": {"summary": summary}})

    def update_qa_pairs(self, chunk_id: str, qa_pairs: List[Dict[str, Any]]):
        return self.collection.update_one({"chunk_id": chunk_id}, {"$set": {"qa_pairs": qa_pairs}})

    def upsert(self, chunk: Dict[str, Any]):
        self.collection.update_one({"chunk_id": chunk["chunk_id"]}, {"$setOnInsert": chunk}, upsert=True)
