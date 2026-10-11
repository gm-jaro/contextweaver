from typing import Any, Dict
from app.dao.chunk_dao import ChunkDAO


class ChunkRepository:
    def __init__(self):
        self.dao = ChunkDAO()

    def ensure_chunk_exists(self, chunk: Dict[str, Any]):
        self.dao.upsert_chunk(chunk)