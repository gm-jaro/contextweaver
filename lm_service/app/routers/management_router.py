from fastapi import APIRouter
from app.config import settings
from app.service.prompts import load_prompts

router = APIRouter(tags=["Management"])


@router.get("/models")
async def get_models():
    models = {}
    for key in dir(settings):
        if key.endswith("_MODEL_NAME"):
            role = key.replace("_MODEL_NAME", "").lower()
            value = getattr(settings, key)
            if value:
                models[role] = value
    return models

@router.get("/prompts")
async def get_prompts():
    return await load_prompts()