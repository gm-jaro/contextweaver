from fastapi import APIRouter


router = APIRouter(tags=["Healthcheck"])


@router.get("/health")
def health_check():
    return {"status": "ok", "service": "llm"}