from pydantic import BaseModel
from typing import List


class GraphRelation(BaseModel):
    head: str
    head_type: str
    relation: str
    tail: str
    tail_type: str

class CurationRequest(BaseModel):
    chunk_id: str
    subgraph: List[GraphRelation]
