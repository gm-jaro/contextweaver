from typing import List, Dict, Optional
from pydantic import Field, BaseModel


class UnstructuredRelation(BaseModel):
    head: str = Field(
        description=(
            "extracted head entity like Microsoft, Apple, John. "
            "Must use human-readable unique identifier."
        )
    )
    head_type: str = Field(
        description="type of the extracted head entity like Person, Company, etc"
    )
    relation: str = Field(description="relation between the head and the tail entities")
    tail: str = Field(
        description=(
            "extracted tail entity like Microsoft, Apple, John. "
            "Must use human-readable unique identifier."
        )
    )
    tail_type: str = Field(
        description="type of the extracted tail entity like Person, Company, etc"
    )


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


class TextRequest(BaseModel):
    text: str


class QuestionEntity(BaseModel):
    name: str
    type: str
    aliases: List[str] = []


class QuestionGraphResponse(BaseModel):
    nodes: List[str]                      # "name: type" (backward compatibility)
    relations: List[str] = []             # v7: unused (relation labels from the question added noise)
    entities: List[QuestionEntity] = []   # v7: canonical entities with aliases
    keywords: List[str] = []
    answer_type: Optional[str] = None
    canonical_question: Optional[str] = None


class NodeResolution(BaseModel):
    input_name: str
    matched_id: Optional[str] = None
    matched_name: Optional[str] = None
    confidence: int
    reasoning: str


class NodeResolutionResponse(BaseModel):
    resolutions: List[NodeResolution]


class ResolveNodeIdentityRequest(BaseModel):
    chunk_text: str
    input_nodes: List[Dict[str, str]]   # [{"name": "Putin", "type": "Osoba"}, ...]
    candidates_map: Dict[str, List[Dict]]  # key: "Osoba: Putin", value: nodes from neo4j
    source_library: Dict[int, str]
