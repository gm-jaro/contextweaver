import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import uvicorn
from fastapi import FastAPI
from loguru import logger
from app.config import settings
from app.routers import healthcheck_router, ingest_router, conv_router

logger.configure(handlers=[{"sink": sys.stdout, "format": "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | {level} | {message}", "level": settings.LOG_LEVEL}])


app = FastAPI(title=settings.PROJECT_NAME, version=settings.PROJECT_VERSION)

app.include_router(ingest_router.router)
app.include_router(healthcheck_router.router)
app.include_router(conv_router.router)


if __name__ == "__main__":   
    logger.info(f"Starting {settings.PROJECT_NAME} on http://{settings.HOST}:{settings.PORT}")
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=True)
