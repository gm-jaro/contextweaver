import asyncio
from typing import List
from fastapi import HTTPException
from loguru import logger
from app import RERANKER
from app.clients import model_client
from app.config.settings import MAX_DOCS
from app.config import costants
from app.dto import QuestionGraphResponse
from app.repo import graph_repo, retriever


async def get_context_for_questions(questions: str | List[str], rag_mode: str) -> str:
    if isinstance(questions, str):
        questions = [questions]

    if rag_mode == "GRAPH":
        try:
            questions_graph: QuestionGraphResponse = await model_client.generate_questions_graph(questions)
            if costants.GRAPH_RETRIEVAL_VERSION == "v7":
                return await _get_graph_context_v7(questions, questions_graph)
            return _get_graph_context_v6(questions, questions_graph)

        except Exception as e:
            logger.exception("Error while querying graph repository: {}", e)
            return ""

    elif rag_mode == "VANILLA":
        loop = asyncio.get_event_loop()

        results_per_query = await asyncio.gather(
            *[loop.run_in_executor(None, retriever.query_without_rerank, q) for q in questions]
        )

        all_docs = []
        seen = set()
        for docs in results_per_query:
            for doc in docs:
                h = hash(doc.page_content)
                if h not in seen:
                    seen.add(h)
                    all_docs.append(doc)

        def rerank_docs(docs, query):
            return sorted(docs, key=lambda doc: RERANKER.score([[query, doc.page_content]])[0], reverse=True)

        all_docs = await loop.run_in_executor(None, rerank_docs, all_docs, questions[0])
        all_results = all_docs[:MAX_DOCS]

        formatted_contexts = []
        for i, doc in enumerate(all_results):
            doc_title = doc.metadata.get("document_name") or "NIEZNANY"
            section_title = doc.metadata.get("section") or "NIEZNANA"
            text = doc.page_content.strip().replace("\t", " ")

            formatted_contexts.append(
                f"\n\nPoczątek dokumentu nr {i}\n\n"
                f"Tytuł dokumentu źródłowego: '{doc_title}'\n\n"
                f"Tytuł sekcji w dokumencie źródłowym: '{section_title}'\n\n"
                f"Fragment dokumentu: '{text}'\n\n"
                f"Koniec dokumentu nr {i}\n\n"
            )

        return "".join(formatted_contexts)

    elif rag_mode == "GRAPH_CHUNK":
        """        
        Strategy (Simplified HippoRAG + KG²RAG):
          1. Seed nodes with similarity score from find_nearest_nodes
          2. BFS multi-hop expansion
          3. Score decay per hop: seed=1.0, 1-hop=0.5, 2-hop=0.25
          4. Score aggregation per chunk: score_chunk = SUM score_node (for every node from that chunk)
          5. Pre-filter: top CHUNK_CANDIDATES chunks
          6. Cross-encoder rerank -> top MAX_DOCS
          
          The mode combines:
          - entity extraction from questions (like GRAPH)
          - multi-hop graph expansion with decay (HippoRAG)
          - chunk retrieval via node->chunk mapping
          - cross-encoder rerank (like VANILLA)
        """
        # HippoRAG with BFS decay + https://arxiv.org/pdf/2502.06864
        return await _get_graph_chunk_context(questions)

    else:
        raise HTTPException(status_code=400, detail="Invalid rag mode")


def _entities_from_question_graph(questions_graph: QuestionGraphResponse) -> List[dict]:
    entities = [e.model_dump() for e in (questions_graph.entities or [])]
    if entities:
        return entities
    out, seen = [], set()
    for n in questions_graph.nodes:
        name, tp = (n.split(": ", 1) + ["UNKNOWN"])[:2] if ": " in n else (n, "UNKNOWN")
        if name and name not in seen:
            seen.add(name)
            out.append({"name": name, "type": tp, "aliases": []})
    return out


async def _get_graph_context_v7(questions: List[str], questions_graph: QuestionGraphResponse) -> str:
    """v7: canonical entities -> seeds (name index, RRF, filter against the question) + facts from the triple index -> 2-hop expansion -> reranking of the whole pool -> source sentences."""
    entities = _entities_from_question_graph(questions_graph)
    question = questions[0]
    retrieval_question = questions_graph.canonical_question or question
    loop = asyncio.get_event_loop()

    seeds = await loop.run_in_executor(None, graph_repo.find_seed_nodes_v7, entities, retrieval_question)
    facts = await loop.run_in_executor(None, graph_repo.find_facts_v7, retrieval_question)
    logger.debug("v7 entities={} seeds={} facts={}", [(e["name"], e["type"]) for e in entities],
                 [(s["name"], s["type"], round(s["score"], 3)) for s in seeds],
                 [(f["head"], f["rel"], f["tail"], round(f["ce"], 3)) for f in facts])
    if not seeds and not facts:
        logger.warning("v7: no seeds or facts for question: {}", question[:80])
        return ""
    return await loop.run_in_executor(None, graph_repo.build_context_v7, seeds, facts, retrieval_question)


def _get_graph_context_v6(questions: List[str], questions_graph: QuestionGraphResponse) -> str:
    """v2-v6 path (kept for comparison)."""
    rels_map = graph_repo.find_nearest_relations(questions_graph.relations, top_k=5) if questions_graph.relations else {}

    all_rel_results = []
    for q_rels in rels_map.values():
        all_rel_results.extend(q_rels)

    node_queries = []
    seen_nodes = set()
    for r in all_rel_results:
        for key in ("head", "tail"):
            name = r.get(key)
            type_ = r.get(f"{key}_type", "UNKNOWN")
            if name and name not in seen_nodes:
                seen_nodes.add(name)
                node_queries.append({"name": name, "type": type_})

    for n in questions_graph.nodes:
        if n not in seen_nodes:
            seen_nodes.add(n)
            if ": " in n:
                parts = n.split(": ", 1)
                name, tp = parts[0], parts[1]
            else:
                name, tp = n, "UNKNOWN"

            if name not in seen_nodes:
                seen_nodes.add(name)
                node_queries.append({"name": name, "type": tp})

    nodes_res = {}
    if node_queries:
        nodes_res = graph_repo.find_nearest_nodes(node_queries, top_k=5)

    return graph_repo.build_nearest_subgraphs(nodes_res, all_rel_results, max_triples=5, max_nodes=5,
                                              hops=1, query=questions[0])


async def _get_graph_chunk_context(questions: List[str]) -> str:
    """GRAPH_CHUNK: node->chunk multi-hop retrieval + rerank."""

    # Entity extraction
    try:
        questions_graph: QuestionGraphResponse = await model_client.generate_questions_graph(questions)

    except Exception as e:
        logger.exception("Entity extraction failed: {}", e)
        return ""

    if not questions_graph.nodes and not questions_graph.relations:
        logger.warning("GRAPH_CHUNK: no entities extracted, falling back to empty context")
        return ""

    node_queries = []
    seen = set()
    for n in questions_graph.nodes:
        if n in seen:
            continue
        seen.add(n)
        if ": " in n:
            name, tp = n.split(": ", 1)
        else:
            name, tp = n, "UNKNOWN"
        node_queries.append({"name": name, "type": tp})

    try:
        nodes_res = {}
        if costants.GRAPH_RETRIEVAL_VERSION == "v7":
            entities = _entities_from_question_graph(questions_graph)
            seeds = graph_repo.find_seed_nodes_v7(entities, questions_graph.canonical_question or questions[0])
            nodes_res = {"v7": [{"id": s["id"], "name": s["name"], "type": s["type"], "rerank_score": s["score"]} for s in seeds]}
        elif node_queries:
            nodes_res = graph_repo.find_nearest_nodes(node_queries, top_k=costants.GRAPH_CHUNK_TOP_K_NODES)

        if not nodes_res:
            logger.warning("GRAPH_CHUNK: no seed nodes found in graph")
            return ""

        # Multi-hop BFS -> chunk scores
        chunk_scores: dict[str, float] = graph_repo.get_multi_hop_chunks_with_scores(seed_nodes_res=nodes_res,
                                                                                     hops=costants.GRAPH_CHUNK_HOPS,
                                                                                     hop_decay=costants.GRAPH_CHUNK_DECAY,
                                                                                     top_k_nodes=costants.GRAPH_CHUNK_TOP_K_NODES)

    except Exception as e:
        logger.exception("GRAPH_CHUNK: multi-hop traversal failed: {}", e)
        return ""

    if not chunk_scores:
        logger.warning("GRAPH_CHUNK: no chunks found via graph traversal")
        return ""

    # Pre-filter - top GRAPH_CHUNK_CANDIDATES by graph score
    sorted_chunks = sorted([(cid, s) for cid, s in chunk_scores.items() if s >= costants.GRAPH_CHUNK_MIN_SCORE],
                           key=lambda x: x[1], reverse=True)[:costants.GRAPH_CHUNK_CANDIDATES]

    candidate_ids = [cid for cid, _ in sorted_chunks]

    # Fetch chunk text
    docs = retriever.get_chunk_documents_by_ids(candidate_ids)
    if not docs:
        logger.warning("GRAPH_CHUNK: no documents retrieved for {} candidate IDs", len(candidate_ids))
        return ""

    # Cross-encoder rerank
    pairs = [[questions[0], doc.page_content] for doc in docs]
    scores = RERANKER.score(pairs)
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    top_docs = [doc for doc, _ in ranked[:MAX_DOCS]]

    # Formatting (same as VANILLA for consistency)
    formatted = []
    for i, doc in enumerate(top_docs):
        doc_title = doc.metadata.get("document_name") or "NIEZNANY"
        section_title = doc.metadata.get("section") or "NIEZNANA"
        text = doc.page_content.strip().replace("\t", " ")

        formatted.append(
            f"\n\nPoczątek dokumentu nr {i}\n\n"
            f"Tytuł dokumentu źródłowego: '{doc_title}'\n\n"
            f"Tytuł sekcji w dokumencie źródłowym: '{section_title}'\n\n"
            f"Fragment dokumentu: '{text}'\n\n"
            f"Koniec dokumentu nr {i}\n\n"
        )

    logger.info("GRAPH_CHUNK: {} seed nodes -> {} unique chunks -> {} candidates -> {} returned",
                sum(len(v) for v in nodes_res.values()),
                len(chunk_scores),
                len(candidate_ids),
                len(top_docs))

    return "".join(formatted)

