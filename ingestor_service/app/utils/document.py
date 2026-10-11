import uuid
from datetime import datetime, timezone
import fitz
from PIL import Image
from typing import List, Dict
import io
from fastapi import UploadFile


def generate_doc_id():
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    random_suffix = uuid.uuid4().hex[:6]
    return f"{random_suffix}_{timestamp}"


def render_pdf_to_images(file: UploadFile, zoom: float = 2.0) -> List[Image.Image]:
    pdf_bytes = file.file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    images = []

    for page in doc:
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)

    return images


def extract_images(file: UploadFile) -> List[Dict]:
    pdf_bytes = file.file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    images_list = []

    for page_num, page in enumerate(doc, start=0):
        image_list = page.get_images(full=True)
        for img_index, img_info in enumerate(image_list, start=0):
            xref = img_info[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            img = Image.open(io.BytesIO(image_bytes))
            images_list.append({
                "page_num": page_num,
                "image_index": img_index,
                "width": img.width,
                "height": img.height,
                "image": img
            })

    return images_list
