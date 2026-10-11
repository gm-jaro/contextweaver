import asyncio
from fastapi import HTTPException, APIRouter
import numpy as np
from app.service.reranker import Reranker
from app.service.retriever import Retriever
from app.models.embedding_dto import *


router = APIRouter(tags=["Embedding"])


retriever = Retriever()
reranker = Reranker()


# -----------------------------
# RETRIEVER UTILITY
# -----------------------------
@router.post("/retriever/count_tokens", response_model=CountTokensResponse)
async def count_tokens(req: CountTokensRequest):
    if not req.text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    count = retriever.count_tokens(req.text)
    return CountTokensResponse(count=count)

@router.get("/retriever/max_length", response_model=MaxLengthResponse)
async def max_length():
    return MaxLengthResponse(max_length=retriever.max_length)

@router.post("/retriever/encode", response_model=EncodeResponse)
async def encode(req: EncodeRequest):
    if not req.text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")  # TODO: encode should only return tokens ids
    token_ids = retriever.encode(req.text)
    return EncodeResponse(token_ids=token_ids)

@router.post("/retriever/tokenize", response_model=TokenizeResponse)
async def tokenize(req: TokenizeRequest):
    if not req.text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")  # TODO: tokenize should return tokens not their ids
    tokens = retriever.tokenize(req.text)
    input_ids = tokens["input_ids"].squeeze().tolist()
    attention_mask = tokens["attention_mask"].squeeze().tolist()
    return TokenizeResponse(input_ids=input_ids, attention_mask=attention_mask)

# -----------------------------
# RETRIEVER MAIN
# -----------------------------
# TODO: embed endpoint + batch embed endpoint

@router.post("/retrieve", response_model=RetrieveResponse)
async def retrieve(req: RetrieveRequest):
    if not req.query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    loop = asyncio.get_event_loop()
    try:
        all_docs = ["Dokument 1...", "Dokument 2...", "Dokument 3..."]  # TODO hybrid search knnsearch? bm25?
        query_embed = await loop.run_in_executor(None, retriever.embed, req.query)
        doc_embeds = await loop.run_in_executor(None, retriever.embed_batch, all_docs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    scores = (query_embed @ doc_embeds.T)
    top_idx = np.argsort(scores)[::-1][:req.top_k]
    results = [RetrieveResponseItem(text=all_docs[i], score=float(scores[i])) for i in top_idx]
    return RetrieveResponse(results=results)

# -----------------------------
# RERANKER
# -----------------------------
@router.post("/rerank", response_model=RerankResponse)
async def rerank(req: RerankRequest):
    if not req.query or not req.candidates:
        raise HTTPException(status_code=400, detail="Query and candidates cannot be empty.")
    loop = asyncio.get_event_loop()
    try:
        scores = await loop.run_in_executor(None, reranker.score_batch, req.query, req.candidates)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    results = [RerankResponseItem(text=txt, score=score) for txt, score in zip(req.candidates, scores)]
    return RerankResponse(results=results)
