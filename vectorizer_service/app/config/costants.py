RERANK_MODEL_NAME = "sdadas/polish-reranker-roberta-v3"
SPARSE_MODEL_NAME = "gm-jaro/sdadas-polish-splade-onnx"
DENSE_MODEL_NAME = "sdadas/stella-pl-retrieval-8k"
COLLECTION_NAME = "vectorstore"
RAG_QUEUE_NAME="ingestor_queue"
GRAPH_QUEUE_NAME="graph_queue"
SUMMARY_QUEUE_NAME="summary_queue"
MONGO_DB_NAME="chunks"
PROJECT_NAME="vectorizer_service"
NEO4J_SPARSE_INDEX="node_fulltext_index"
NEO4J_DENSE_INDEX="node_embedding_index"
NEO4J_REL_DENSE_INDEX="rel_embedding_index"
NEO4J_REL_SPARSE_INDEX="rel_fulltext_index"

# GRAPH_CHUNKS mode
GRAPH_CHUNK_HOPS = 2  # BFS depth
GRAPH_CHUNK_DECAY = 0.5  # discount per hop
GRAPH_CHUNK_TOP_K_NODES = 5  # top seed nodes per query
GRAPH_CHUNK_CANDIDATES = 30  # how many chunks to rerank
GRAPH_CHUNK_MIN_SCORE = 0.01  # cut off chunks with a marginal score

# --- v7: graph retrieval ---
GRAPH_RETRIEVAL_VERSION = "v7"          # "v6" = old path (find_nearest_nodes + 1-hop + 15 triples)
NEO4J_NODE_NAME_DENSE_INDEX = "node_name_embedding_index"   # embedding of the name only (no type prefix)
NEO4J_NODE_NAME_SPARSE_INDEX = "node_name_fulltext_index"   # fulltext on the name only
NEO4J_FACT_DENSE_INDEX = "rel_fact_embedding_index"         # embedding of the verbalized triple
V7_SEED_VEC_K = 15            # vector index candidates per entity
V7_SEED_FT_K = 15             # fulltext candidates per entity
V7_SEED_PER_ENTITY = 3        # max seeds per question entity
V7_SEED_MAX = 8               # max seeds in total
V7_SEED_MIN_CE = 0.20         # min cross-encoder score (question, node name)
V7_SEED_REL_CE = 0.35         # seed must have >= REL_CE * best score within the entity
V7_SEED_MIN_DENSE = 0.80      # min name similarity (Neo4j score, cosine -> [0,1]) when there is no fulltext hit
V7_FACT_K = 30                # fact index candidates
V7_FACT_TOP = 6               # direct facts after reranking
V7_FACT_MIN_CE = 0.30
V7_HOPS = 2
V7_HOP_DECAY = 0.5
V7_PER_SEED_LIMIT = 40        # edges per seed (1-hop), sorted by chunk count
V7_PER_NODE_LIMIT_2HOP = 12   # edges per intermediate node (2-hop)
V7_HUB_MAX_DEGREE = 40        # nodes with a higher degree are not expanded in 2-hop
V7_POOL_MAX = 300             # max triples to rerank
V7_BUDGET_TRIPLES = 40        # triples in the context
V7_CONTEXT_MAX_CHARS = 12000
V7_CE_WEIGHT = 0.7            # reranker weight vs structure (seed_score * decay^hop)
V7_EVIDENCE_MAX_CHARS = 350

import os
# Graph context variants (switched by environment variables at service start, for experiments):
#   V7_EVIDENCE=0  -> triples only, no source sentences (isolates the graph retrieval effect)
#   V7_EXTRA=0     -> no extra sentences picked for the question (v7.1)
V7_INCLUDE_EVIDENCE = os.environ.get("V7_EVIDENCE", "1") == "1"
V7_INCLUDE_EXTRA_SENTENCES = os.environ.get("V7_EXTRA", "1") == "1"

# --- v7.1 ---
V7_SEED_EXACT_SCORE = 0.6     # exact match on entity name/alias = seed regardless of the cross-encoder
V7_EVIDENCE_WINDOW_MAX = 480  # source sentence + the next sentence, if they fit in this limit
V7_EXTRA_CHUNKS = 5           # how many articles (by triple relevance) to take extra sentences from
V7_EXTRA_SENT_PER_CHUNK = 2   # how many sentences per article (cross-encoder against the question)
V7_EXTRA_MIN_CE = 0.30
