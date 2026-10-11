from fastapi import APIRouter, HTTPException, Body, Query
from app.clients.vectorizer import VectorizerClient
from app.config.settings import VECTORIZER_URL
from app.models.qa_dto import QAResponse, QARequest, QARAGMode
from app.service.qa import QAService

router = APIRouter(tags=["QA"])
vectorizer_client = VectorizerClient(base_url=VECTORIZER_URL)
qa_service = QAService(vectorizer_client)


@router.post("/qa", response_model=QAResponse)
async def qa_endpoint(req: QARequest = Body(...), rag_mode: QARAGMode = Query(default=QARAGMode.VANILLA), gen_subquestions: bool = Query(default=False)):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    response = await qa_service.answer_question(req.question, rag_mode, gen_subquestions)
    return response
