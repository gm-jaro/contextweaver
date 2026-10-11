from typing import List
import httpx
from loguru import logger
from app.models.qa_dto import QARAGMode


class VectorizerClient:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=15.0))

    async def get_context(self, questions: List[str], rag_mode: QARAGMode) -> str:
        rag_mode_value = rag_mode.value if hasattr(rag_mode, "value") else str(rag_mode)

        payload = {"questions": questions, "rag_mode": rag_mode_value.upper()}
        try:
            response = await self.client.post(f"{self.base_url}/questions/context", json=payload)
            response.raise_for_status()
            return response.json()

        except httpx.RequestError as e:
            logger.error(f"Error connecting to Vectorizer service: {e}")
            return ""

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error from Vectorizer service: {e.response.text}")
            return ""

    async def close(self):
        await self.client.aclose()
