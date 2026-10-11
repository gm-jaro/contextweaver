from langchain_huggingface import HuggingFaceEmbeddings
from app.config.costants import RERANK_MODEL_NAME, DENSE_MODEL_NAME
from langchain_community.cross_encoders import HuggingFaceCrossEncoder


# v7: queries (embed_query) get the instruction prefix required by sdadas/stella-pl-retrieval-8k
# ("Instruct: Given a web search query, retrieve relevant passages that answer the query.\nQuery: ").
# Documents, node names and facts are embedded with embed_documents, without the prefix.
DENSE_MODEL = HuggingFaceEmbeddings(model=DENSE_MODEL_NAME,
                                    model_kwargs={"trust_remote_code": True, "device": "cuda"},
                                    query_encode_kwargs={"prompt_name": "s2p_query"})

RERANKER = HuggingFaceCrossEncoder(model_name=RERANK_MODEL_NAME, model_kwargs={'trust_remote_code': True,
                                                                               'device': 'cpu'})
