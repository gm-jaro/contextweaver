import httpx
from loguru import logger


class ModelClient:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.client = httpx.Client(timeout=httpx.Timeout(600.0, connect=30.0, read=30.0))

    def get_models(self) -> dict:
        response = self.client.get(f"{self.base_url}/models")
        response.raise_for_status()

        data = response.json()

        if not isinstance(data, dict):
            logger.error(f"Invalid model config format: {data}")
            return {}

        return {k: str(v).strip() for k, v in data.items() if v}

    def get_prompts(self) -> dict:
        response = self.client.get(f"{self.base_url}/prompts")
        response.raise_for_status()

        data = response.json()

        if not isinstance(data, dict):
            logger.error(f"Invalid prompts config format: {data}")
            return {}

        return {k: str(v).strip() for k, v in data.items() if v}

    def correct_ocr(self, image_bytes: bytes, text: str) -> str:
        response = self.client.post(
            f"{self.base_url}/vlm/correct",
            files={"image": image_bytes},
            data={"text": text}
        )
        if response.status_code == 200:
            return response.json()["corrected_text"]
        else:
            logger.error(f"VLM correction failed: {response.text}")
            return text

    def generate_qa(self, chunk_text: str) -> list[dict]:
        response = self.client.post(
            f"{self.base_url}/qa/generate",
            json={"text": chunk_text}
        )
        if response.status_code == 200:
            return response.json()["qa"]
        else:
            logger.error(f"QA generation failed: {response.text}")
            return []

    def summarize(self, text: str) -> str:
        response = self.client.post(
            f"{self.base_url}/summary",
            json={"text": text}
        )
        if response.status_code == 200:
            return response.json()["summary"]
        else:
            logger.error(f"Summary failed: {response.text}")
            return ""


    # TOKENIZER METHODS
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

    def get_max_tokens(self) -> int:
        response = self.client.get(f"{self.base_url}/tokenizer/max")
        if response.status_code == 200:
            return response.json().get("max_tokens", 0)
        else:
            logger.error(f"Max tokens fetch failed: {response.text}")
            return 0

    def tokenize(self, text: str) -> list[str]:
        response = self.client.post(
            f"{self.base_url}/tokenizer/tokenize",
            json={"text": text},
        )
        if response.status_code == 200:
            return response.json().get("tokens", [])
        logger.error(f"Tokenization failed: {response.text}")
        return []

    def encode(self, text: str) -> list[int]:
        response = self.client.post(
            f"{self.base_url}/tokenizer/encode",
            json={"text": text},
        )
        if response.status_code == 200:
            return response.json().get("ids", [])
        logger.error(f"Encoding failed: {response.text}")
        return []
