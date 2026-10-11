from typing import Literal, List
from pydantic import BaseModel


class QuestionsRequest(BaseModel):
    questions: List[str]
    rag_mode: Literal["VANILLA", "GRAPH", "HYBRID", "GRAPH_CHUNK"] = "VANILLA"
