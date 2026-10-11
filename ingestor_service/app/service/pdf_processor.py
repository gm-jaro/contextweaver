from io import BytesIO
from app.config.settings import DO_OCR, DO_ENRICHMENT
from app.models.document import Document
from app.service.chunker import chunk_document
from app.service.ocr_service import init_ocr
from app.service.visual_enrichment import init_vlm
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode, LayoutOptions
from docling.datamodel.layout_model_specs import DOCLING_LAYOUT_EGRET_XLARGE
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption, WordFormatOption
from docling.pipeline.simple_pipeline import SimplePipeline
from docling.datamodel.base_models import DocumentStream
# from docling_core.types.doc import PictureItem
# from loguru import logger
# from docling_core.types.doc import ImageRefMode


def process_document(document: Document):
    accelerator_options=AcceleratorOptions(num_threads=12, device=AcceleratorDevice.CUDA)
    pipeline_options=PdfPipelineOptions()
    pipeline_options.accelerator_options=accelerator_options
    pipeline_options.do_ocr=DO_OCR
    pipeline_options.enable_remote_services=DO_ENRICHMENT
    pipeline_options.do_formula_enrichment=DO_ENRICHMENT
    pipeline_options.do_table_structure=DO_ENRICHMENT
    pipeline_options.do_picture_description=DO_ENRICHMENT
    pipeline_options.layout_options=LayoutOptions(model_spec=DOCLING_LAYOUT_EGRET_XLARGE)
    pipeline_options.table_structure_options.do_cell_matching=DO_ENRICHMENT
    pipeline_options.table_structure_options.mode=TableFormerMode.ACCURATE
    if DO_ENRICHMENT:
        pipeline_options.picture_description_options=init_vlm()
    if DO_OCR:
        pipeline_options.ocr_options=init_ocr(force_full_page_ocr=DO_OCR)
    pipeline_options.generate_picture_images = DO_ENRICHMENT
    pipeline_options.images_scale = 2.0

    doc_converter=DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options
            ),
            InputFormat.DOCX: WordFormatOption(
                pipeline_cls=SimplePipeline
            ),
        },
    )

    buf=BytesIO(document.file_bytes)
    source=DocumentStream(name=document.name, stream=buf)

    conv_results=doc_converter.convert(source, max_num_pages=100, max_file_size=20971520)

    # html_output = conv_results.document.export_to_markdown(image_mode=ImageRefMode.EMBEDDED,
    #                                                        image_placeholder="![{text}]({link})")
    # logger.debug("Generated DOC for document '{}'\n{}\n--- END OF DOC ---", document.name, html_output)
    #
    # logger.info(f"Found {len(conv_results.document.pictures)} pictures.")
    # for i, pic in enumerate(conv_results.document.pictures):
    #     logger.info(f"Picture {i} annotations: {pic.annotations}")
    #     description = next((a.text for a in pic.annotations if a.kind == 'description'), None)
    #     logger.info(f"Picture {i} generated description: {description}")
    #
    # for element, _level in conv_results.document.iterate_items():
    #     if isinstance(element, PictureItem):
    #         if element.annotations:
    #             for anno in element.annotations:
    #                 logger.info(f"Picture description: {anno.text}")
    #         else:
    #             logger.warning("Picture found, but no VLM annotations (check the Ollama connection)")

    return chunk_document(conv_results.document, document.id, document.name)
