import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import uvicorn
from fastapi import FastAPI
from loguru import logger
from app.config import settings
from app.routers import healthcheck_router, context_router
from app.config.costants import PROJECT_NAME
from app.routers import graph

logger.configure(handlers=[{"sink": sys.stdout, "format": "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | {level} | {message}", "level": settings.LOG_LEVEL}])

app = FastAPI(title=PROJECT_NAME, version=settings.PROJECT_VERSION)

app.include_router(context_router.router)
app.include_router(healthcheck_router.router)
app.include_router(graph.router)


if __name__ == "__main__":   
    logger.info(f"Starting {PROJECT_NAME} on http://127.0.0.1:{settings.PORT}")
    uvicorn.run("app.main:app", host="127.0.0.1", port=settings.PORT, reload=True)
