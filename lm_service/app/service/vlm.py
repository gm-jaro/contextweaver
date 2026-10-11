from loguru import logger
from glmocr import GlmOcr
import tempfile
import os
from app.config.settings import GLM_ONFIG_PATH


async def generate_ocr_from_bytes(image_bytes: bytes) -> str:
    if not image_bytes:
        raise ValueError("Image cannot be empty.")

    logger.info("Running GLM-OCR via GlmOcr class")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(image_bytes)
        tmp_path = tmp.name

    try:
        with GlmOcr(config_path=GLM_ONFIG_PATH) as parser:
            result = parser.parse(tmp_path)

        return result.markdown_result

    finally:
        os.remove(tmp_path)