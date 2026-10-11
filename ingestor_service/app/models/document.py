from typing import Optional
from pydantic import BaseModel


class Document(BaseModel):
    name: str
    file_bytes: bytes
    id: str
    metadata: Optional[dict[str, str]] = None