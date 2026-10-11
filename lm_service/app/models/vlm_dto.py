from pydantic import BaseModel


class OcrRequest(BaseModel):
    image_base64: str