from fastapi import APIRouter, UploadFile, File
from fastapi.responses import Response
from weasyprint import HTML
import tempfile


router = APIRouter(tags=["Convert"])

@router.post("/convert/html2pdf")
async def html2pdf(file: UploadFile = File(...)):
    html_content = await file.read()

    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_bytes = HTML(string=html_content.decode(), base_url=tmpdir).write_pdf()

    return Response(content=pdf_bytes,
                    media_type="application/pdf",
                    headers={"Content-Disposition": f"attachment; filename={file.filename}.pdf"})