import json
import asyncio
import redis.asyncio as redis
from app.config.costants import RAG_QUEUE_NAME
from app.config.settings import REDIS_URL
from loguru import logger
from app.models.chunk import IngestChunk
from app.repo.chunk_repo import ChunkRepository
from app.service.vectorizer import HybridRetriever


MAX_CONCURRENCY = 5

async def process_entry(redis_client, chunk_repo, retriever, entry_id, entry, semaphore):
    async with semaphore:
        try:
            raw_data = json.loads(entry[b"data"].decode("utf-8"))
            chunk = IngestChunk(**raw_data)

            retriever.add_documents([chunk])

            # Set status for summary job
            chunk_repo.update_chunk_ingest_status(chunk_id=chunk.chunk_id, status="DONE")

            await redis_client.xdel(RAG_QUEUE_NAME, entry_id)  # TODO: groups xack instead of xdel

            logger.success(f"Ingested chunk {chunk.chunk_id}")

        except Exception:
            logger.exception(f"Failed to ingest job {entry_id}")
            chunk_repo.update_chunk_ingest_status(chunk_id=chunk.chunk_id, status="ERROR")


async def run_ingest_worker():
    redis_client = redis.from_url(REDIS_URL)
    retriever = HybridRetriever()
    chunk_repo = ChunkRepository()

    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

    logger.info(f"Ingesting from {RAG_QUEUE_NAME}")

    last_id = "0"

    try:
        while True:
            streams = await redis_client.xread({RAG_QUEUE_NAME: last_id}, block=5000)  # TODO: one, specific stream

            tasks = []
            for _, entries in streams or []:
                for entry_id, entry in entries:
                    last_id = entry_id
                    task = asyncio.create_task(process_entry(redis_client, chunk_repo, retriever, entry_id, entry, semaphore))
                    tasks.append(task)

            if tasks:
                await asyncio.gather(*tasks)

    finally:
        await redis_client.close()


if __name__ == "__main__":
    asyncio.run(run_ingest_worker())
