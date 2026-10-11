import asyncio
import redis.asyncio as redis
import json
from loguru import logger
from app.clients import model_client
from app.config.settings import REDIS_URL
from app.config.costants import SUMMARY_QUEUE_NAME
from app.repo.chunk_repo import ChunkRepository


MAX_CONCURRENCY = 5
LLM_TIMEOUT_SECONDS = 900


async def process_entry(redis_client, chunk_repo, entry_id, entry, semaphore):
    async with semaphore:
        try:
            raw_data = json.loads(entry[b"data"].decode("utf-8"))
            text = raw_data["text"]
            chunk_id = raw_data["chunk_id"]

            logger.info(f"Generating summary for chunk {chunk_id}")

            summary = await asyncio.wait_for(model_client.summarize(text), timeout=LLM_TIMEOUT_SECONDS)

            chunk_repo.update_chunk_summary(chunk_id=chunk_id, summary=summary)
            chunk_repo.update_chunk_summary_status(chunk_id=chunk_id, status="DONE")

            await redis_client.xdel(SUMMARY_QUEUE_NAME, entry_id)

            logger.success(f"Summary completed for chunk {chunk_id}")

        except asyncio.TimeoutError:
            logger.error(f"LLM timeout for job {entry_id}")
            chunk_repo.update_chunk_summary_status(chunk_id=chunk_id, status="ERROR")

        except Exception:
            logger.exception(f"Failed to process summary job {entry_id}")
            chunk_repo.update_chunk_summary_status(chunk_id=chunk_id, status="ERROR")


async def run_summary_worker():
    redis_client = redis.from_url(REDIS_URL)
    chunk_repo = ChunkRepository()

    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

    logger.info(f"Consuming summary jobs from {SUMMARY_QUEUE_NAME}")

    last_id = "$"

    try:
        while True:
            streams = await redis_client.xread({SUMMARY_QUEUE_NAME: last_id}, block=5000)

            tasks = []

            for _, entries in streams:
                for entry_id, entry in entries:
                    last_id = entry_id

                    task = asyncio.create_task(process_entry(redis_client, chunk_repo, entry_id, entry, semaphore))
                    tasks.append(task)

            if tasks:
                await asyncio.gather(*tasks)

    finally:
        await redis_client.close()
        await model_client.aclose()


if __name__ == "__main__":
    asyncio.run(run_summary_worker())
