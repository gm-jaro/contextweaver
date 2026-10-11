from typing import List
from fastapi import APIRouter, HTTPException, Body
from app.dto import ManualGraphRequest, ManualGraphResponse, TripletResult
from app.models.graph import CurationRequest
from app.agent.workflow import app as agent_app
from loguru import logger
from app.repo import graph_repo

router = APIRouter()

@router.post("/graph/curate")
async def curate_graph(request: CurationRequest):
    """Endpoint receiving the raw subgraph from lm_service."""
    if not request.subgraph:
        return {"status": "skipped", "reason": "Empty subgraph"}

    logger.info(f"Starting graph curation for chunk {request.chunk_id} with {len(request.subgraph)} relations")

    input_subgraph = [rel.model_dump() for rel in request.subgraph]

    initial_state = {
        "input_subgraph": input_subgraph,
        "graph_context": "",
        "schema_summary": "",
        "messages": [],
        "iterations": 0,
        "uuid_map": {}
    }

    try:
        final_state = await agent_app.ainvoke(initial_state, config={"recursion_limit": 50})

        return {
            "status": "success",
            "chunk_id": request.chunk_id,
            "final_iterations": final_state["iterations"]
        }

    except Exception as e:
        logger.error(f"Agent failed for chunk {request.chunk_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/graph/manual", response_model=ManualGraphResponse)
async def insert_manual_graph(request: ManualGraphRequest = Body(...)):
    """Inserts the given triplets directly into Neo4j."""
    if not request.triplets:
        raise HTTPException(status_code=400, detail="triplets cannot be empty")

    results: List[TripletResult] = []
    inserted = 0
    failed = 0

    for t in request.triplets:
        label = f"({t.head_type}: {t.head}) --[{t.relation}]--> ({t.tail_type}: {t.tail})"

        if request.dry_run:
            logger.info(f"[dry_run] {label}")
            results.append(TripletResult(head=t.head, relation=t.relation,
                                         tail=t.tail, status="dry_run"))
            inserted += 1
            continue

        try:
            graph_repo.upsert_triplet(
                head=t.head,
                head_type=t.head_type,
                relation=t.relation,
                tail=t.tail,
                tail_type=t.tail_type,
                chunk_id=request.chunk_id,
                existing_head_id=None,
                existing_tail_id=None,
            )
            logger.success(f"[manual] upserted: {label}")
            results.append(TripletResult(head=t.head, relation=t.relation,
                                         tail=t.tail, status="ok"))
            inserted += 1

        except Exception as e:
            logger.error(f"[manual] failed: {label} → {e}")
            results.append(TripletResult(head=t.head, relation=t.relation,
                                         tail=t.tail, status="error", detail=str(e)))
            failed += 1

    return ManualGraphResponse(inserted=inserted, failed=failed, results=results)


@router.delete("/graph/manual/{chunk_id}", summary="Delete manually inserted triplets")
async def delete_manual_graph(chunk_id: str):
    """Deletes all nodes and relationships linked to the given chunk_id."""
    cypher = """
        MATCH (n)-[r]->()
        WHERE r.chunk_id = $chunk_id
        DELETE r
    """
    cypher_isolated = """
        MATCH (n)
        WHERE n.chunk_id = $chunk_id
          AND NOT (n)--()
        DELETE n
    """
    try:
        with graph_repo.driver.session() as session:
            result = session.run(cypher, chunk_id=chunk_id)
            deleted_rels = result.consume().counters.relationships_deleted
            session.run(cypher_isolated, chunk_id=chunk_id)

        logger.info(f"[manual delete] chunk_id={chunk_id}, relationships deleted: {deleted_rels}")
        return {"chunk_id": chunk_id, "relationships_deleted": deleted_rels}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/graph/manual/preview", summary="Preview graph for chunk_id")
async def preview_manual_graph(chunk_id: str = "manual-debug"):
    """Returns all triplets linked to the given chunk_id - to verify after an insert."""
    cypher = """
        MATCH (h)-[r]->(t)
        WHERE r.chunk_id = $chunk_id
        RETURN h.name AS head, h.type AS head_type,
               r.type  AS relation,
               t.name  AS tail, t.type AS tail_type
        ORDER BY head
    """
    try:
        with graph_repo.driver.session() as session:
            rows = session.run(cypher, chunk_id=chunk_id).data()
        return {"chunk_id": chunk_id, "count": len(rows), "triplets": rows}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
