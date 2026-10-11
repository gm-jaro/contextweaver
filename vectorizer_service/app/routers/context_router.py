import time
from fastapi import APIRouter, HTTPException
from loguru import logger
from app.models.qa import QuestionsRequest
from app.service.context import get_context_for_questions

router = APIRouter()

@router.post("/questions/context")
async def retrieve_context(request: QuestionsRequest):
    if not request.questions or not any(q.strip() for q in request.questions):
        raise HTTPException(status_code=400, detail="Questions cannot be empty")

    start_time = time.perf_counter()
    result = await get_context_for_questions(request.questions, request.rag_mode)
    end_time = time.perf_counter()

    duration_s = end_time - start_time
    logger.info(f"Request for {len(request.questions)} questions, rag_mode={request.rag_mode} "
                f"took {duration_s:.2f} s")

    return result
