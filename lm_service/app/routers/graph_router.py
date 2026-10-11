from typing import List
from fastapi import APIRouter, HTTPException, Body
from app.models.graph_dto import (BuildGraphResponse, TextRequest, QuestionGraphResponse, ResolveNodeIdentityRequest,
                                  NodeResolutionResponse)
from app.service.graph import GraphBuilder


router = APIRouter(tags=["Graph"])
graph_builder = GraphBuilder()


@router.post("/graph/nodes")
async def graph_nodes_endpoint(request: TextRequest = Body(...)):
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    return await graph_builder.extract_nodes(text)


@router.post("/graph/qa_pairs")
async def generate_qa_pairs_endpoint(request: TextRequest = Body(...)):
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    nodes = await graph_builder.extract_nodes(text)
    nodes_str = ", ".join(nodes)
    return await graph_builder.generate_qa_pairs(text, nodes_str)


@router.post("/graph/build", response_model=BuildGraphResponse)
async def build_graph_endpoint(request: TextRequest = Body(...)):
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    res = await graph_builder.build_graph(text)
    return BuildGraphResponse(**res)


@router.post("/graph/questions_graph", response_model=QuestionGraphResponse)
async def gen_questions_graph_endpoint(questions_request: List[TextRequest] = Body(...)):
    questions = [q.text.strip() for q in questions_request if q.text and q.text.strip()]
    if not questions:
        raise HTTPException(status_code=400, detail="Questions cannot be empty.")
    res = await graph_builder.analyze_questions(questions)
    return QuestionGraphResponse(**res)


@router.post("/graph/resolve_nodes", response_model=NodeResolutionResponse)
async def resolve_node_identity_endpoint(request: ResolveNodeIdentityRequest = Body(...)):
    if not request.input_nodes:
        raise HTTPException(status_code=400, detail="input_nodes cannot be empty.")
    if not request.chunk_text.strip():
        raise HTTPException(status_code=400, detail="chunks cannot be empty.")

    return await graph_builder.resolve_node_identity(chunk_text=request.chunk_text, input_nodes=request.input_nodes,
                                                     candidates_map=request.candidates_map,
                                                     source_library=request.source_library)
