from fastapi import APIRouter, HTTPException, Body
from app.models.graph_dto import TextRequest
from app.service.summary import generate_summary


router = APIRouter(tags=["Summary"])

@router.post("/summary")
async def summary_endpoint(request: TextRequest = Body(...)):
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    return generate_summary(text)
