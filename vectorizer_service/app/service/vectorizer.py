from typing import List
from app import DENSE_MODEL, RERANKER
from app.config.settings import QDRANT_URL
from app.config.costants import SPARSE_MODEL_NAME, COLLECTION_NAME
from fastembed.common.model_description import SparseModelDescription, ModelSource
from fastembed.sparse.splade_pp import supported_splade_models
supported_splade_models.append(SparseModelDescription(
    model=SPARSE_MODEL_NAME,
    vocab_size=50001,
    description="Polish export to onnx.",
    license="apache-2.0",
    size_in_GB=0.325,
    sources=ModelSource(hf=SPARSE_MODEL_NAME),
    model_file="model.onnx"
))
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_classic.retrievers import ContextualCompressionRetriever
from app.models.chunk import IngestChunk
from qdrant_client import QdrantClient
from langchain_qdrant import RetrievalMode, QdrantVectorStore, FastEmbedSparse
from qdrant_client.http.models import Distance, VectorParams, SparseVectorParams, models
from langchain_core.documents import Document


class HybridRetriever:
    def __init__(self):
        self.client = self._build_client()
        dense, sparse = self._init_embeddings()
        self._create_collection(self.client)

        self.vector_store = self._build_vector_store(self.client, dense, sparse)
        base_retriever = self.vector_store.as_retriever(search_kwargs={"k": 50})

        reranker = self._build_reranker()
        self.retriever = ContextualCompressionRetriever(
            base_compressor=reranker,
            base_retriever=base_retriever,
        )

    @staticmethod
    def _build_client():
        return QdrantClient(QDRANT_URL, prefer_grpc=True)

    @staticmethod
    def _init_embeddings():
        sparse = FastEmbedSparse(model_name=SPARSE_MODEL_NAME)
        return DENSE_MODEL, sparse

    @staticmethod
    def _create_collection(client: QdrantClient):
        collections = client.get_collections().collections
        exists = any(c.name == COLLECTION_NAME for c in collections)
        if exists:
            return

        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                "dense": VectorParams(size=1024, distance=Distance.COSINE)
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(
                    index=models.SparseIndexParams(on_disk=False)
                )
            },
        )

    @staticmethod
    def _build_vector_store(client, dense, sparse):
        return QdrantVectorStore(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding=dense,
            sparse_embedding=sparse,
            retrieval_mode=RetrievalMode.HYBRID,
            vector_name="dense",
            sparse_vector_name="sparse",
        )

    @staticmethod
    def _build_reranker():
        return CrossEncoderReranker(model=RERANKER, top_n=3)

    def add_documents(self, chunks: list[IngestChunk]):
        docs = []
        for chunk in chunks:
            doc = Document(
                page_content=chunk.text,
                metadata={
                    "document_id": chunk.document_id,
                    "chunk_id": chunk.chunk_id,
                    "chunk_index": chunk.chunk_index,
                    "document_name": chunk.document_name,
                    **chunk.metadata,
                    "created_at": chunk.created_at,
                }
            )
            docs.append(doc)

        self.vector_store.add_documents(docs)

    def query(self, question: str):
        return self.retriever.invoke(question)

    def query_without_rerank(self, question: str):
        return self.retriever.base_retriever.invoke(question)

    def get_chunks_text_by_ids(self, chunk_ids: List[str]) -> str:
        if not chunk_ids:
            return "No source chunks found for this node."

        points, _ = self.client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.chunk_id",
                        match=models.MatchAny(any=chunk_ids),
                    )
                ]
            ),
            with_payload=True,
            with_vectors=False
        )

        texts = []
        for idx, p in enumerate(points):
            payload = p.payload
            metadata = payload.get("metadata", {})

            content = payload.get("page_content", "Brak treści chunka")
            doc_name = metadata.get("document_name", "Nazwa dokumentu nieznana")
            section = metadata.get("section", "Brak nazwy sekcji")
            created_at = metadata.get("created_at", "Brak daty utworzenia")

            content = content.replace(f"{section}\n", "")

            header = (f"--- CHUNK nr {idx} ---\n"
                      f"DOKUMENT: {doc_name}\n"
                      f"SEKCJA: {section}\n"
                      f"DATA UTW.: {created_at}\n"
                      f"{'-' * 20}")

            texts.append(f"{header}\n{content}")

        return "\n\n".join(texts)

    def get_chunk_documents_by_ids(self, chunk_ids: List[str]) -> List[Document]:
        """Like get_chunks_text_by_ids, but returns List[Document] for reranking."""
        if not chunk_ids:
            return []

        points, _ = self.client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.chunk_id",
                        match=models.MatchAny(any=chunk_ids),
                    )
                ]
            ),
            with_payload=True,
            with_vectors=False,
        )

        docs = []
        for p in points:
            payload  = p.payload
            metadata = payload.get("metadata", {})
            content  = payload.get("page_content", "")
            section  = metadata.get("section", "")
            content  = content.replace(f"{section}\n", "")

            docs.append(Document(page_content=content, metadata=metadata))

        return docs
