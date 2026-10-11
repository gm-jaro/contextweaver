from pydantic import BaseModel
from typing import Dict, Any, List, Optional


class IngestChunk(BaseModel):
    document_id: str
    document_name: str
    chunk_id: str
    chunk_index: int
    text: str
    metadata: Dict[str, Any]
    created_at: str
    qa_pairs: Optional[List[Dict]] = None
    summary: Optional[str] = None
