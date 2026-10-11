from typing import List, Optional
from pydantic import BaseModel
from enum import Enum


class QARAGMode(str, Enum):
    GRAPH = "graph"
    VANILLA = "vanilla"
    HYBRID = "hybrid"
    GRAPH_CHUNK = "graph_chunk"

class QARequest(BaseModel):
    question: str

class QAResponse(BaseModel):
    answer: str
    source_context: str
    reasoning: Optional[str] = None
    subquestions: Optional[List[str]] = None
