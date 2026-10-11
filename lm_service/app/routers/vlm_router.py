from fastapi import APIRouter, UploadFile, File, HTTPException
from app.service.vlm import generate_ocr_from_bytes

router = APIRouter(tags=["OCR"])


@router.post("/ocr")
async def ocr_endpoint(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="File is required.")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are supported.")

    try:
        image_bytes = await file.read()
        text = await generate_ocr_from_bytes(image_bytes)
        return {"text": text}

    except Exception as e:
        logger.exception("OCR processing failed")
        raise HTTPException(status_code=500, detail=str(e))
