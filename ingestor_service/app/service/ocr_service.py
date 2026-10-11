import os
from loguru import logger
from docling.datamodel.pipeline_options import RapidOcrOptions
from app.config.settings import PADDLE_DIR
from modelscope import snapshot_download


def init_ocr(force_full_page_ocr: bool = True) -> RapidOcrOptions:
    logger.info(f'Initializing OCR from {PADDLE_DIR}')

    # TODO: ocr rec + det models - ensure polish support
    det_model_path = os.path.join(
        PADDLE_DIR, "temp", "det", "ch_PP-OCRv5_server_det.onnx"
    )
    rec_model_path = os.path.join(
        PADDLE_DIR, "temp", "rec", "ch_PP-OCRv5_rec_server_infer.onnx"
    )
    ocr_options = RapidOcrOptions(
        force_full_page_ocr=force_full_page_ocr,
        det_model_path=det_model_path,
        rec_model_path=rec_model_path,
        cls_model_path=None,
        lang=["polish", "english"],
    )
    
    return ocr_options