import httpx
from loguru import logger


class IngestorClient:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.client = httpx.Client()

    def count_tokens(self, text: str) -> int:
        response = self.client.post(
            f"{self.base_url}/tokenizer/count",
            json={"text": text}
        )
        if response.status_code == 200:
            return response.json().get("count", 0)
        else:
            logger.error(f"Token count failed: {response.text}")
            return 0
