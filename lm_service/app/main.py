import sys, os
from app.routers.qa_router import vectorizer_client

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import uvicorn
from fastapi import FastAPI
from loguru import logger
from app.config import settings
from app.routers import (healthcheck_router, embedding_router, management_router, qa_router,
                         summary_router, graph_router, vlm_router)

logger.configure(handlers=[{"sink": sys.stdout, "format": "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | {level} | {message}", "level": settings.LOG_LEVEL}])


app = FastAPI(title=settings.PROJECT_NAME, version=settings.PROJECT_VERSION, logger=logger, log_level=settings.LOG_LEVEL)

app.include_router(embedding_router.router)
app.include_router(healthcheck_router.router)
app.include_router(management_router.router)
app.include_router(qa_router.router)
app.include_router(summary_router.router)
app.include_router(vlm_router.router)
app.include_router(graph_router.router)


@app.on_event("shutdown")
async def shutdown_event():
    await vectorizer_client.close()

if __name__ == "__main__":
    logger.info(f"Starting {settings.PROJECT_NAME} on http://127.0.0.1:{settings.PORT}")
    uvicorn.run("app.main:app", host="127.0.0.1", port=settings.PORT, reload=True, log_level=settings.LOG_LEVEL.lower())
