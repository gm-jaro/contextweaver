import asyncio
import json
from loguru import logger
from typing import List, Dict, Optional
import redis.asyncio as redis
from app.clients import model_client
from app.config.costants import GRAPH_QUEUE_NAME, SUMMARY_QUEUE_NAME
from app.config.settings import REDIS_URL, MAX_RETRIES
from app.dto import Relation
from app.repo import retriever
from app.repo.chunk_repo import ChunkRepository
from app.repo.graph_repo import GraphRepository


def extract_unique_nodes(subgraph: List[Relation]) -> List[Dict[str, str]]:

    seen = set()
    nodes = []
    for rel in subgraph:
        for name, ntype in [(rel.head, rel.head_type), (rel.tail, rel.tail_type)]:
            key = (name.strip().lower(), ntype.strip().lower())
            if key not in seen:
                seen.add(key)
                nodes.append({"name": name.strip(), "type": ntype.strip()})
    return nodes


async def enrich_candidates_with_source_texts(nearest_map: Dict[str, List[Dict]],
                                              graph_repo: GraphRepository) -> tuple[Dict[str, List[Dict]], Dict[int, str]]:
    enriched: Dict[str, List[Dict]] = {}
    source_text_cache: Dict[str, Optional[str]] = {}  # real_id -> text
    text_to_idx: Dict[str, int] = {}  # text -> index in the library
    source_library: Dict[int, str] = {}  # index -> text (for the prompt)

    for query_key, candidates in nearest_map.items():
        enriched_candidates = []
        for candidate in candidates:
            real_id = candidate.get("id", "")

            if real_id not in source_text_cache:
                chunk_ids = await graph_repo.get_node_chunk_ids_async(real_id)
                source_text_cache[real_id] = retriever.get_chunks_text_by_ids(chunk_ids) if chunk_ids else None

            text = source_text_cache[real_id]
            if text and text not in text_to_idx:
                idx = len(text_to_idx)
                text_to_idx[text] = idx
                source_library[idx] = text

            enriched_candidates.append({"short_id": candidate["short_id"], "name": candidate["name"],
                                        "type": candidate["type"], "source_ref": text_to_idx[text] if text else None})

        enriched[query_key] = enriched_candidates

    return enriched, source_library

def state_diff(before: dict, after: dict):
    """Return minimal diff of two dicts."""
    return {k: after[k] for k in after if before.get(k) != after.get(k)}


def safe_serialize(obj):
    """Convert objects like AIMessage to dicts for JSON logging."""
    if isinstance(obj, list):
        return [safe_serialize(o) for o in obj]
    elif hasattr(obj, "model_dump"):
        return obj.model_dump()
    elif isinstance(obj, dict):
        return {k: safe_serialize(v) for k, v in obj.items()}
    else:
        return obj


async def try_enqueue_summary(document_id: str, chunk_repo: ChunkRepository, redis_client: redis.Redis):
    total_chunks = chunk_repo.get_chunks_count(document_id)
    ready_chunks = chunk_repo.get_ready_chunks_count(document_id)  # ingest_status="DONE" & graph_status="DONE"

    if ready_chunks == total_chunks:
        chunks = chunk_repo.get_chunks(document_id)
        for chunk in chunks:
            await redis_client.xadd(SUMMARY_QUEUE_NAME, {"chunk_id": chunk["chunk_id"], "text": chunk["text"]})  # TODO: remove logic or get text from qdrant


async def run_graph_worker():
    redis_client = redis.from_url(REDIS_URL)
    chunk_repo = ChunkRepository()
    graph_repo = GraphRepository()

    logger.info(f"Consuming graph jobs from {GRAPH_QUEUE_NAME}")
    last_id = "0"  # 0 - all messages $ - messages from start the of worker

    try:
        while True:
            streams = await redis_client.xread({GRAPH_QUEUE_NAME: last_id}, block=0)
            for _, entries in streams or []:
                for entry_id, entry in entries:
                    last_id = entry_id
                    try:
                        raw_data = json.loads(entry[b"data"].decode("utf-8"))
                        text = raw_data["text"]
                        chunk_id = raw_data["chunk_id"]
                        retries = raw_data.get("retries", 0)
                        # document_id = raw_data["document_id"]

                        logger.debug(f"\n\nText:\n {text}\n\n")
                        logger.info(f"Processing chunk {chunk_id} (Attempt {retries})")

                        try:
                            chunk_graph = await model_client.build_graph(text)  # "Sub"-graph - graph from specific chunk
                            input_subgraph = chunk_graph.graph
                            input_nodes = extract_unique_nodes(input_subgraph)  # [{"name": "Putin", "type": "Osoba"}, {"name": "Moskwa", "type": "Miasto"}, ...]

                            uuid_map: Dict[str, str] = {}
                            nearest_map = await graph_repo.find_nearest_nodes_with_short_ids_async(input_nodes, top_k=3, uuid_map=uuid_map)

                            # chunks_summary = await model_client.summarize(text)

                            enriched_nearest_map, source_library = await enrich_candidates_with_source_texts(nearest_map, graph_repo)

                            resolution_response = await model_client.resolve_node_identity(chunk_text=text,
                                                                                           input_nodes=input_nodes,
                                                                                           candidates_map=enriched_nearest_map,
                                                                                           source_library=source_library)

                            id_map: Dict[str, Optional[str]] = {}
                            for r in resolution_response['resolutions']:
                                if r['matched_id']:
                                    real_id = uuid_map.get(r['matched_id'])
                                    if not real_id:
                                        logger.warning(f"Could not resolve short_id '{r['matched_id']}' for '{r['input_name']}', treating as new node")
                                    id_map[r['input_name']] = real_id
                                else:
                                    id_map[r['input_name']] = None

                            for rel in chunk_graph.graph:
                                graph_repo.upsert_triplet(
                                    head=rel.head, head_type=rel.head_type,
                                    relation=rel.relation,
                                    tail=rel.tail, tail_type=rel.tail_type,
                                    chunk_id=chunk_id,
                                    existing_head_id=id_map.get(rel.head),
                                    existing_tail_id=id_map.get(rel.tail)
                                )

                            logger.success(f"Finished subgraph fusion for chunk {chunk_id}")

                            # Set status for summary job
                            chunk_repo.update_chunk_graph_status(chunk_id=chunk_id, status="DONE")

                            # try_enqueue_summary(document_id, chunk_repo, redis_client)

                            await redis_client.xdel(GRAPH_QUEUE_NAME, entry_id)
                            logger.success(f"Chunk {chunk_id} processed successfully")

                        except Exception as fusion_error:
                            logger.error(f"Fusion failed on chunk {chunk_id}: {fusion_error}")
                            graph_repo.rollback_chunk_changes(chunk_id)

                            if retries < MAX_RETRIES:
                                logger.info(f"Retrying chunk {chunk_id}...")
                                raw_data["retries"] = retries + 1

                                await redis_client.xadd(GRAPH_QUEUE_NAME, {"data": json.dumps(raw_data)})
                            else:
                                logger.error(f"Max retries reached for chunk {chunk_id}")
                                chunk_repo.update_chunk_graph_status(chunk_id=chunk_id, status="ERROR")

                            await redis_client.xdel(GRAPH_QUEUE_NAME, entry_id)

                    except Exception as e:
                        logger.exception(f"Failed to process graph job {entry_id}: {e}")
    finally:
        await redis_client.close()
        await graph_repo.aclose()
        graph_repo.close()


if __name__ == "__main__":
    asyncio.run(run_graph_worker())
