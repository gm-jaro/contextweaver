from docling.datamodel.pipeline_options import PictureDescriptionApiOptions
from pydantic import AnyUrl
from app.config.settings import OLLAMA_URL
from app.config.costants import VLM_IMAGE_DESCRIPTION_PROMPT, VLM_NAME


def init_vlm():
    options = PictureDescriptionApiOptions(
        url=AnyUrl(f"{OLLAMA_URL}/v1/chat/completions"),
        params=dict(
            model=VLM_NAME,
            num_ctx=150000,
            num_predict=16000,
            temperature=0.1
        ),
        prompt=VLM_IMAGE_DESCRIPTION_PROMPT,
        timeout=300,
        scale=2.0
    )
    return options
