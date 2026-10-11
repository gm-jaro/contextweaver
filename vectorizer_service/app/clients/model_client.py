from typing import List, Dict
import httpx
from loguru import logger
from app.dto import BuildGraphResponse, QAPair, QuestionGraphResponse


class ModelClient:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(connect=100.0, read=900.0, write=60.0, pool=30.0),
                                        limits=httpx.Limits(max_connections=50, max_keepalive_connections=20))

    async def get_models(self) -> dict:
        response = await self.client.get(f"{self.base_url}/models")
        response.raise_for_status()

        data = response.json()
        if not isinstance(data, dict):
            logger.error(f"Invalid model config format: {data}")
            return {}

        return {k: str(v).strip() for k, v in data.items() if v}

    async def get_prompts(self) -> dict:
        response = await self.client.get(f"{self.base_url}/prompts")
        response.raise_for_status()

        data = response.json()
        if not isinstance(data, dict):
            logger.error(f"Invalid prompts config format: {data}")
            return {}

        return {k: str(v).strip() for k, v in data.items() if v}

    async def correct_ocr(self, image_bytes: bytes, text: str) -> str:
        response = await self.client.post(f"{self.base_url}/vlm/correct", files={"image": image_bytes}, data={"text": text})
        if response.status_code == 200:
            return response.json()["corrected_text"]
        else:
            logger.error(f"VLM correction failed: {response.text}")
            return text


    # GRAPH METHODS
    async def extract_nodes(self, text: str) -> List[str]:
        response = await self.client.post(f"{self.base_url}/graph/nodes", json={"text": text})
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Node extraction failed [{response.status_code}]: {response.text}")
            return []

    async def generate_qa_pairs(self, text: str) -> List[QAPair]:
        response = await self.client.post(f"{self.base_url}/graph/qa_pairs", json={"text": text})
        if response.status_code == 200:
            return [QAPair(**item) for item in response.json()]
        else:
            logger.error(f"QA pair generation failed [{response.status_code}]: {response.text}")
            return []

    async def build_graph(self, text: str) -> BuildGraphResponse:
        logger.debug(f"Building graph for text (len={len(text)})")

        try:
            response = await self.client.post(f"{self.base_url}/graph/build", json={"text": text})
            response.raise_for_status()
        except Exception as e:
            logger.exception("Graph build request failed")
            raise RuntimeError("Graph build request failed") from e

        try:
            data = response.json()
            return BuildGraphResponse(**data)
        except Exception as e:
            logger.exception("Invalid response payload from graph service")
            raise RuntimeError("Invalid graph service response") from e

    async def generate_questions_graph(self, questions: List[str]) -> QuestionGraphResponse:
        """Calls the /graph/questions_graph endpoint with a list of questions and returns the question graph."""
        payload = [{"text": q} for q in questions]
        try:
            response = await self.client.post(f"{self.base_url}/graph/questions_graph", json=payload)
            response.raise_for_status()
        except httpx.RequestError as e:
            logger.error(f"Failed to connect to model service: {e}")
            return QuestionGraphResponse(nodes=[], relations=[])
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error from model service [{e.response.status_code}]: {e.response.text}")
            return QuestionGraphResponse(nodes=[], relations=[])

        try:
            data = response.json()
            return QuestionGraphResponse(**data)
        except Exception as e:
            logger.error(f"Failed to parse questions graph response: {e}")
            return QuestionGraphResponse(nodes=[], relations=[])


    async def resolve_node_identity(self, chunk_text: str, input_nodes: List[Dict[str, str]],
                                    candidates_map: Dict[str, List[Dict]], source_library: Dict[int, str]) -> dict:
        payload = {"chunk_text": chunk_text, "input_nodes": input_nodes, "candidates_map": candidates_map,
                   "source_library": source_library}

        try:
            response = await self.client.post(f"{self.base_url}/graph/resolve_nodes", json=payload)
            response.raise_for_status()
        except httpx.RequestError as e:
            logger.error(f"Connection error: {e}")
            return {"resolutions": []}
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error [{e.response.status_code}]: {e.response.text}")
            return {"resolutions": []}

        try:
            return response.json()
        except Exception as e:
            logger.error(f"Invalid response: {e}")
            return {"resolutions": []}


    # SUMMARY METHODS
    async def summarize(self, text: str) -> str:
        response = await self.client.post(f"{self.base_url}/summary", json={"text": text})
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Summary failed [{response.status_code}]: {response.text}")
            return ""


    # TOKENIZER METHODS
    async def count_tokens(self, text: str) -> int:
        response = await self.client.post(f"{self.base_url}/tokenizer/count", json={"text": text})
        if response.status_code == 200:
            return response.json().get("count", 0)
        else:
            logger.error(f"Token count failed: {response.text}")
            return 0

    async def get_max_tokens(self) -> int:
        response = await self.client.get(f"{self.base_url}/tokenizer/max")
        if response.status_code == 200:
            return response.json().get("max_tokens", 0)
        else:
            logger.error(f"Max tokens fetch failed: {response.text}")
            return 0

    async def tokenize(self, text: str) -> list[str]:
        response = await self.client.post(f"{self.base_url}/tokenizer/tokenize", json={"text": text},)
        if response.status_code == 200:
            return response.json().get("tokens", [])
        logger.error(f"Tokenization failed: {response.text}")
        return []

    async def encode(self, text: str) -> list[int]:
        response = await self.client.post(f"{self.base_url}/tokenizer/encode", json={"text": text})
        if response.status_code == 200:
            return response.json().get("ids", [])
        logger.error(f"Encoding failed: {response.text}")
        return []

    async def aclose(self):
        await self.client.aclose()
