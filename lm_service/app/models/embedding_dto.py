from pydantic import BaseModel
from typing import List


class QueryRequest(BaseModel):
    query: str

class RetrieveRequest(QueryRequest):
    top_k: int = 50

class RetrieveResponseItem(BaseModel):
    text: str
    score: float

class RetrieveResponse(BaseModel):
    results: List[RetrieveResponseItem]

class RerankRequest(BaseModel):
    query: str
    candidates: List[str]

class RerankResponseItem(BaseModel):
    text: str
    score: float

class RerankResponse(BaseModel):
    results: List[RerankResponseItem]

class CountTokensRequest(BaseModel):
    text: str

class CountTokensResponse(BaseModel):
    count: int

class MaxLengthResponse(BaseModel):
    max_length: int

class EncodeRequest(BaseModel):
    text: str

class EncodeResponse(BaseModel):
    token_ids: List[int]

class TokenizeRequest(BaseModel):
    text: str

class TokenizeResponse(BaseModel):
    input_ids: List[int]
    attention_mask: List[int]