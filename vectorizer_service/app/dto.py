from typing import List, Optional
from pydantic import BaseModel


class QAPair(BaseModel):
    question: str
    answer: str


class Relation(BaseModel):
    head: str
    head_type: str
    relation: str
    tail: str
    tail_type: str


class BuildGraphResponse(BaseModel):
    graph: List[Relation]


class ManualGraphRequest(BaseModel):
    triplets: List[Relation]
    chunk_id: Optional[str] = "manual-debug"
    dry_run: bool = False

class TripletResult(BaseModel):
    head: str
    relation: str
    tail: str
    status: str
    detail: Optional[str] = None

class ManualGraphResponse(BaseModel):
    inserted: int
    failed: int
    results: List[TripletResult]


class QuestionEntity(BaseModel):
    name: str
    type: str
    aliases: List[str] = []


class QuestionGraphResponse(BaseModel):
    nodes: List[str]
    relations: List[str] = []
    entities: List[QuestionEntity] = []      # v7
    keywords: List[str] = []
    answer_type: Optional[str] = None
    canonical_question: Optional[str] = None


class NodeResolution(BaseModel):
    input_name: str
    matched_id: Optional[str]   # None = new node
    matched_name: Optional[str]
    confidence: float
    reasoning: str

class ResolutionResponse(BaseModel):
    resolutions: List[NodeResolution]