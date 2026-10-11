import json
from typing import List
import redis
from weasyprint import HTML
from fastapi import APIRouter, UploadFile, File, Form, Body
from app.config.costants import RAG_QUEUE_NAME, GRAPH_QUEUE_NAME
from app.config.settings import REDIS_URL
from app.models.document import Document
from app.repo.chunk_repo import ChunkRepository
from app.service.pdf_processor import process_document
from app.utils.document import generate_doc_id


router = APIRouter(tags=["Ingest"])
redis_client = redis.from_url(url=REDIS_URL)
chunk_repo = ChunkRepository()


@router.post("/document/ingest")
async def ingest_document(file: UploadFile = File(...), source: str = Form(None), original_key: str = Form(None)):
    document = Document(
        name=file.filename,
        file_bytes=await file.read(),
        id=generate_doc_id(),
        metadata={"source": source, "original_key": original_key}
    )

    chunks_data = process_document(document)
    for chunk in chunks_data:
        chunk["ingest_status"] = "PENDING"
        chunk["graph_status"] = "PENDING"
        chunk["summary_status"] = "PENDING"

        chunk_repo.ensure_chunk_exists(chunk)
        message = json.dumps(chunk)

        redis_client.xadd(RAG_QUEUE_NAME, {"data": message}, maxlen=None, approximate=False)
        redis_client.xadd(GRAPH_QUEUE_NAME, {"data": message}, maxlen=None, approximate=False)

    return {"status": "queued", "chunks_count": len(chunks_data), "document_id": document.id}


@router.post("/document/ingestText")
async def ingest_text(text: str = Body(..., embed=True),
                      name: str = Body("Manual test text", embed=True)):
    pdf_bytes = HTML(string=f"<div style='font-size:12pt; line-height:1.4;'>{text}</div>").write_pdf()
    document = Document(name=name, file_bytes=pdf_bytes, id=generate_doc_id())

    chunks_data = process_document(document)
    for chunk in chunks_data:
        chunk["ingest_status"] = "PENDING"
        chunk["graph_status"] = "PENDING"
        chunk["summary_status"] = "PENDING"

        chunk_repo.ensure_chunk_exists(chunk)
        message = json.dumps(chunk)

        redis_client.xadd(RAG_QUEUE_NAME, {"data": message}, maxlen=None, approximate=False)
        redis_client.xadd(GRAPH_QUEUE_NAME, {"data": message}, maxlen=None, approximate=False)

    return {"status": "queued", "chunks_count": len(chunks_data), "document_id": document.id}


@router.post("/document/batch/ingest")
async def ingest_documents(files: List[UploadFile] = File(...)):
    results = []

    for i, file in enumerate(files):
        document = Document(
            name=file.filename,
            file_bytes=await file.read(),
            id=generate_doc_id()
        )

        chunks_data = process_document(document)
        for chunk in chunks_data:
            chunk["ingest_status"] = "PENDING"
            chunk["graph_status"] = "PENDING"
            chunk["summary_status"] = "PENDING"

            chunk_repo.ensure_chunk_exists(chunk)
            message = json.dumps(chunk)

            redis_client.xadd(RAG_QUEUE_NAME, {"data": message}, maxlen=None, approximate=False)
            redis_client.xadd(GRAPH_QUEUE_NAME, {"data": message}, maxlen=None, approximate=False)

        results.append({
            "document_id": document.id,
            "chunks_count": len(chunks_data)
        })

    return {"status": "queued", "documents": results}
