import os
from datetime import datetime, timezone
from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from loguru import logger
from transformers import AutoTokenizer
from app.clients import model_client
from typing import List, Dict, Any
from docling_core.transforms.chunker.hierarchical_chunker import (
    ChunkingDocSerializer,
    ChunkingSerializerProvider,
    DocChunk, TripletTableSerializer
)
from docling_core.transforms.serializer.base import BaseDocSerializer, SerializationResult
from docling_core.transforms.serializer.common import create_ser_result
from docling_core.transforms.serializer.markdown import MarkdownPictureSerializer
from docling_core.types.doc.document import (
    PictureClassificationData, PictureDescriptionData, PictureItem,
    PictureMoleculeData, DoclingDocument, SectionHeaderItem
)
from typing_extensions import override


class AnnotationPictureSerializer(MarkdownPictureSerializer):
    @override
    def serialize(
            self,
            *,
            item: PictureItem,
            doc_serializer: BaseDocSerializer,
            doc: DoclingDocument,
            **kwargs: Any,
    ) -> SerializationResult:
        text_parts: list[str] = []
        for annotation in item.annotations:
            if isinstance(annotation, PictureClassificationData):
                predicted_class = (
                    annotation.predicted_classes[0].class_name
                    if annotation.predicted_classes
                    else None
                )
                if predicted_class is not None:
                    text_parts.append(f"Typ obrazka: {predicted_class}")
            elif isinstance(annotation, PictureMoleculeData):
                text_parts.append(f"SMILES: {annotation.smi}")
            elif isinstance(annotation, PictureDescriptionData):
                text_parts.append(f"Opis obrazka: {annotation.text}")

        text_res = "\n".join(text_parts)
        text_res = doc_serializer.post_process(text=text_res)
        return create_ser_result(text=text_res, span_source=item)


class SerializerProvider(ChunkingSerializerProvider):
    def get_serializer(self, doc: DoclingDocument):
        return ChunkingDocSerializer(
            doc=doc,
            table_serializer=TripletTableSerializer(),
            picture_serializer=AnnotationPictureSerializer()
        )


chunker = HybridChunker(tokenizer=HuggingFaceTokenizer(
        tokenizer=AutoTokenizer.from_pretrained(model_client.get_models()["retrieval"]),
        max_tokens=8192,  # Model specific (sdadas/stella-pl-retrieval-8k from (model_client.get_models()["retrieval"])
    ),
    serializer_provider=SerializerProvider(),
)

logger.debug(f"Max tokens: {chunker.max_tokens}")


def chunk_document(document: DoclingDocument, doc_id: str, doc_name: str) -> List[Dict[str, Any]]:
    headings = [item for item in document.texts if isinstance(item, SectionHeaderItem)]
    for heading in headings:
        print(f"Heading (level {heading.level}): {heading.text}")

    for table in document.tables:
        print("Table:")
        print(table.export_to_markdown(doc=document))

    chunk_iter = chunker.chunk(dl_doc=document)

    results = []
    for index, chunk in enumerate(chunk_iter):
        pages = set()
        for item in chunk.meta.doc_items:
            if item.prov[0].page_no:
                pages.add(item.prov[0].page_no)

        page_number = min(pages) if pages else 1
        section_name = None
        logger.debug(f"Section Headings: {chunk.meta.headings}")

        if chunk.meta.headings:
            section_name = chunk.meta.headings[-1]

        _, ext = os.path.splitext(doc_name)
        source_type = ext.lstrip('.').lower() if ext else "unknown"

        chunk_data = {
            "document_id": doc_id,
            "document_name": doc_name,
            "chunk_id": f"{doc_id}_{index:04d}",
            "chunk_index": index,
            "text": chunker.contextualize(chunk=chunk),
            "metadata": {
                "page": page_number,
                "section": section_name,
                "source_type": source_type
            },
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        }

        results.append(chunk_data)

    return results
