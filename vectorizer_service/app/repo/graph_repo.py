import json
import re
from typing import List, Dict, Optional
from loguru import logger
from neo4j import GraphDatabase, AsyncGraphDatabase
from app import DENSE_MODEL, RERANKER
from app.config.costants import (NEO4J_SPARSE_INDEX, NEO4J_DENSE_INDEX, NEO4J_REL_DENSE_INDEX, NEO4J_REL_SPARSE_INDEX,
                                 NEO4J_NODE_NAME_DENSE_INDEX, NEO4J_NODE_NAME_SPARSE_INDEX, NEO4J_FACT_DENSE_INDEX,
                                 V7_SEED_VEC_K, V7_SEED_FT_K, V7_SEED_PER_ENTITY, V7_SEED_MAX, V7_SEED_MIN_CE, V7_SEED_REL_CE,
                                 V7_SEED_MIN_DENSE, V7_FACT_K, V7_FACT_TOP, V7_FACT_MIN_CE, V7_HOPS, V7_HOP_DECAY,
                                 V7_PER_SEED_LIMIT, V7_PER_NODE_LIMIT_2HOP, V7_HUB_MAX_DEGREE, V7_POOL_MAX,
                                 V7_BUDGET_TRIPLES, V7_CONTEXT_MAX_CHARS, V7_CE_WEIGHT, V7_EVIDENCE_MAX_CHARS,
                                 V7_SEED_EXACT_SCORE, V7_EVIDENCE_WINDOW_MAX, V7_EXTRA_CHUNKS, V7_EXTRA_SENT_PER_CHUNK,
                                 V7_EXTRA_MIN_CE, V7_INCLUDE_EVIDENCE, V7_INCLUDE_EXTRA_SENTENCES)
from app.config.settings import NEO4J_URL, NEO4J_USER, NEO4J_PASSWORD
from app.dto import Relation


class GraphRepository:
    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URL, auth=(NEO4J_USER, NEO4J_PASSWORD))
        self.async_driver = AsyncGraphDatabase.driver(NEO4J_URL, auth=(NEO4J_USER, NEO4J_PASSWORD))
        self.dense = DENSE_MODEL

    def _embed_text(self, text: str) -> list[float]:
        # Embeddings stored in the graph (type: name, relation label) - without the instruction prefix.
        return self.dense.embed_documents([text])[0]

    def _embed_node(self, name: str, type_: str) -> list[float]:
        return self._embed_text(f"{type_}: {name}")

    def close(self):
        self.driver.close()

    async def aclose(self):
        await self.async_driver.close()

    @staticmethod
    def escape_lucene(text: str) -> str:
        return re.sub(r'([+\-!(){}[\]^"~*?:\\/])', r'\\\1', text)

    @staticmethod
    def _shorten_id(original_id: str, id_type: str, mapping: Dict[str, str]) -> str:
        """Returns a short ID (e.g. n1, r1)."""
        for short, real in mapping.items():
            if real == original_id:
                return short

        count = sum(1 for k in mapping.keys() if k.startswith(id_type))
        short_id = f"{id_type}{count + 1}"
        mapping[short_id] = original_id
        return short_id

    def get_node_chunk_ids(self, node_id: str) -> List[str]:
        """Gets the list of chunk ids assigned to a node."""
        query = "MATCH (n) WHERE elementId(n) = $id RETURN n.chunk_ids as chunk_ids"
        with self.driver.session() as session:
            res = session.run(query, id=node_id).single()
            if res and res['chunk_ids']:
                return res['chunk_ids']
            return []

    async def get_node_chunk_ids_async(self, node_id: str) -> List[str]:
        query = "MATCH (n) WHERE elementId(n) = $id RETURN n.chunk_ids as chunk_ids"
        async with self.async_driver.session() as session:
            async def do_query(tx):
                result = await tx.run(query, id=node_id)
                return await result.single()

            res = await session.execute_read(do_query)
            if res and res['chunk_ids']:
                return res['chunk_ids']
            return []

    def _get_expanded_node_ids_with_depth(self, start_node_ids: list[str], hops: int = 2, limit: int = 200) -> dict[str, int]:
        """Same logic as _expand_subgraph, but returns {node_elementId: min_hop_distance} instead of formatted text lines."""
        if not start_node_ids or hops < 1:
            return {}

        cypher = f"""
            MATCH (start)
            WHERE elementId(start) IN $ids
            MATCH path = (start)-[*1..{hops}]-(neighbor)
            RETURN elementId(neighbor) AS node_id,
                   min(length(path))   AS min_dist
            LIMIT $limit
        """

        result = {nid: 0 for nid in start_node_ids}  # seed = depth 0

        with self.driver.session() as session:
            for row in session.run(cypher, ids=start_node_ids, limit=limit).data():
                nid = row["node_id"]
                dist = row["min_dist"]
                if nid not in result or dist < result[nid]:
                    result[nid] = dist

        return result

    def _bfs_fallback(self, seed_ids: list[str], hops: int) -> list[dict]:
        """Iterative BFS when variable-length path is unavailable."""
        visited: dict[str, int] = {}  # node_id -> min_dist
        frontier = seed_ids[:]

        for depth in range(hops + 1):
            if not frontier:
                break

            cypher = """
            UNWIND $ids AS nid
            MATCH (n) WHERE n.id = nid OR elementId(n) = nid
            OPTIONAL MATCH (n)-[]-(neighbor)
            RETURN n.id AS node_id, n.chunk_ids AS chunk_ids,
                   collect(DISTINCT neighbor.id) AS neighbor_ids
            """
            records = self.driver.execute_query(cypher, ids=frontier).records

            next_frontier = []
            for r in records:
                nid = r["node_id"]
                if nid not in visited:
                    visited[nid] = depth
                if depth < hops:
                    for nb in (r["neighbor_ids"] or []):
                        if nb and nb not in visited:
                            next_frontier.append(nb)

            frontier = list(set(next_frontier))

        #  chunk_ids for all visited nodes
        all_ids = list(visited.keys())
        if not all_ids:
            return []

        cypher2 = """
        UNWIND $ids AS nid
        MATCH (n) WHERE n.id = nid OR elementId(n) = nid
        WHERE n.chunk_ids IS NOT NULL AND size(n.chunk_ids) > 0
        RETURN n.id AS node_id, n.chunk_ids AS chunk_ids
        """
        records2 = self.driver.execute_query(cypher2, ids=all_ids).records

        return [
            {
                "node_id":   r["node_id"],
                "chunk_ids": r["chunk_ids"],
                "min_dist":  visited.get(r["node_id"], 0),
            }
            for r in records2
        ]

    def get_multi_hop_chunks_with_scores(self, seed_nodes_res: dict, hops: int = 2, hop_decay: float = 0.5, top_k_nodes: int = 5) -> dict[str, float]:
        """seed_nodes_res -> BFS expansion -> {chunk_id: aggregated_score}"""

        # 1. Seed nodes with similarity score
        seed_scores: dict[str, float] = {}   # {elementId: sim_score}

        for q_results in seed_nodes_res.values():
            for node in q_results[:top_k_nodes]:
                nid   = node.get("id") or node.get("elementId")
                score = float(node.get("rerank_score") or node.get("score") or 1.0)
                if nid and (nid not in seed_scores or score > seed_scores[nid]):
                    seed_scores[nid] = score

        if not seed_scores:
            return {}

        # 2. BFS - reuse the existing traversal logic
        node_depths = self._get_expanded_node_ids_with_depth(start_node_ids=list(seed_scores.keys()), hops=hops)
        # seed nodes must be in the map too (depth=0)
        node_depths.update({nid: 0 for nid in seed_scores if nid not in node_depths})
        all_node_ids = list(node_depths.keys())

        # 3. Chunk_ids for all nodes in one query
        cypher_chunks = """
            UNWIND $ids AS nid
            MATCH (n) WHERE elementId(n) = nid
            AND n.chunk_ids IS NOT NULL AND size(n.chunk_ids) > 0
            RETURN elementId(n) AS node_id, n.chunk_ids AS chunk_ids
        """
        with self.driver.session() as session:
            rows = session.run(cypher_chunks, ids=all_node_ids).data()

        # 4. Score aggregation per chunk
        chunk_scores: dict[str, float] = {}

        for row in rows:
            nid = row["node_id"]
            chunk_ids = row["chunk_ids"]
            depth = node_depths.get(nid, hops)

            base = seed_scores.get(nid, 1.0)   # seed similarity or 1.0 for expanded
            score = base * (hop_decay ** depth)

            for cid in chunk_ids:
                chunk_scores[cid] = chunk_scores.get(cid, 0.0) + score

        return chunk_scores

    def rollback_chunk_changes(self, chunk_id: str):
        """Rolls back changes for chunk_id."""
        with self.driver.session() as session:
            # Relations rollback
            session.run("""
                MATCH ()-[r:RELATED]-()
                WHERE $cid IN r.chunk_ids
                WITH r, [id IN r.chunk_ids WHERE id <> $cid] AS updated_ids
                FOREACH (_ IN CASE WHEN size(updated_ids) = 0 THEN [1] ELSE [] END | 
                    DELETE r
                )
                FOREACH (_ IN CASE WHEN size(updated_ids) > 0 THEN [1] ELSE [] END | 
                    SET r.chunk_ids = updated_ids
                )
            """, cid=chunk_id)

            # Nodes rollback
            session.run("""
                MATCH (n:Node)
                WHERE $cid IN n.chunk_ids
                WITH n, [id IN n.chunk_ids WHERE id <> $cid] AS updated_ids
                FOREACH (_ IN CASE WHEN size(updated_ids) = 0 THEN [1] ELSE [] END | 
                    DETACH DELETE n
                )
                FOREACH (_ IN CASE WHEN size(updated_ids) > 0 THEN [1] ELSE [] END | 
                    SET n.chunk_ids = updated_ids
                )
            """, cid=chunk_id)

        logger.warning(f"Rollback completed for chunk {chunk_id}")

    def update_chunk_graph(self, chunk_id: str, subgraph: List[Relation]):
        """Updates Neo4j with the "sub"-graph (graph of a specific chunk)."""
        with self.driver.session() as session:
            for rel in subgraph:
                head_emb_calc = self._embed_node(rel.head, rel.head_type)
                tail_emb_calc = self._embed_node(rel.tail, rel.tail_type)
                rel_emb_calc = self._embed_text(rel.relation)
                session.execute_write(self._merge_relation, chunk_id, rel,
                                      head_emb_calc,
                                      tail_emb_calc,
                                      rel_emb_calc)

    def find_nearest_nodes(self, nodes: List[Dict[str, str]], top_k: int = 8) -> Dict[str, List[Dict]]:
        results = {}
        initial_k = top_k * 5
        hybrid_query = """
        CALL db.index.vector.queryNodes($vector_index, $k, $embedding)
        YIELD node, score
        WITH collect({node: node, score: score}) AS v

        CALL db.index.fulltext.queryNodes($fulltext_index, $text_query, {limit: $k})
        YIELD node, score
        WITH v, collect({node: node, score: score}) AS f

        WITH v + f AS allResults
        UNWIND allResults AS r
        WITH r.node AS n, max(r.score) AS finalScore
        RETURN elementId(n) AS id, n.name AS name, n.type AS type, finalScore AS score
        ORDER BY score DESC LIMIT $k
        """
        with self.driver.session() as session:
            for node in nodes:
                emb = self._embed_node(node['name'], node['type'])
                query_string = f"{node['type']}: {node['name']}"
                tokens = query_string.replace(":", "").split()
                text_query = " ".join([f"{self.escape_lucene(t)}~" for t in tokens])

                raw_res = session.run(
                    hybrid_query,
                    vector_index=NEO4J_DENSE_INDEX,
                    fulltext_index=NEO4J_SPARSE_INDEX,
                    k=initial_k,
                    embedding=emb,
                    text_query=text_query
                ).data()

                if not raw_res:
                    results[query_string] = []
                    continue

                pairs = [[query_string, f"{r['type']}: {r['name']}"] for r in raw_res]
                scores = RERANKER.score(pairs)
                for i, score in enumerate(scores):
                    raw_res[i]['rerank_score'] = float(score)

                final_nodes = sorted(raw_res, key=lambda x: x['rerank_score'], reverse=True)[:top_k]
                results[query_string] = final_nodes

        return results


    async def find_nearest_nodes_async(self, nodes: List[Dict[str, str]], top_k: int = 8) -> Dict[str, List[Dict]]:
        results = {}
        initial_k = top_k * 5
        hybrid_query = """
        CALL db.index.vector.queryNodes($vector_index, $k, $embedding)
        YIELD node, score
        WITH collect({node: node, score: score}) AS v

        CALL db.index.fulltext.queryNodes($fulltext_index, $text_query, {limit: $k})
        YIELD node, score
        WITH v, collect({node: node, score: score}) AS f

        WITH v + f AS allResults
        UNWIND allResults AS r
        WITH r.node AS n, max(r.score) AS finalScore
        RETURN elementId(n) AS id, n.name AS name, n.type AS type, finalScore AS score
        ORDER BY score DESC LIMIT $k
        """
        async with self.async_driver.session() as session:
            for node in nodes:
                emb = self._embed_node(node['name'], node['type'])
                query_string = f"{node['type']}: {node['name']}"
                tokens = query_string.replace(":", "").split()
                text_query = " ".join([f"{self.escape_lucene(t)}~" for t in tokens])

                async def do_query(tx, emb=emb, text_query=text_query):
                    result = await tx.run(
                        hybrid_query,
                        vector_index=NEO4J_DENSE_INDEX,
                        fulltext_index=NEO4J_SPARSE_INDEX,
                        k=initial_k,
                        embedding=emb,
                        text_query=text_query
                    )
                    return await result.data()

                raw_res = await session.execute_read(do_query)
                if not raw_res:
                    results[query_string] = []
                    continue

                pairs = [[query_string, f"{r['type']}: {r['name']}"] for r in raw_res]
                scores = RERANKER.score(pairs)
                for i, score in enumerate(scores):
                    raw_res[i]['rerank_score'] = float(score)

                final_nodes = sorted(raw_res, key=lambda x: x['rerank_score'], reverse=True)[:top_k]
                results[query_string] = final_nodes

        return results

    def find_nearest_relations(self, relations: List[str], top_k: int = 8) -> Dict[str, List[Dict]]:
        results = {}
        initial_k = top_k * 5
        hybrid_query = """
        CALL db.index.vector.queryRelationships($vector_index, $k, $embedding)
        YIELD relationship, score
        WITH collect({rel: relationship, score: score}) AS v

        CALL db.index.fulltext.queryRelationships($fulltext_index, $text_query, {limit: $k})
        YIELD relationship, score
        WITH v, collect({rel: relationship, score: score}) AS f

        WITH v + f AS allResults
        UNWIND allResults AS r
        WITH r.rel AS rel, max(r.score) AS finalScore
        RETURN elementId(startNode(rel)) AS head_id,
               startNode(rel).name AS head,
               startNode(rel).type AS head_type,
               rel.type AS relation_name,
               elementId(rel) AS relation_id,
               elementId(endNode(rel)) AS tail_id,
               endNode(rel).name AS tail,
               endNode(rel).type AS tail_type,
               finalScore AS score
        ORDER BY score DESC LIMIT $k
        """

        with self.driver.session() as session:
            for query_text in relations:
                emb = self._embed_text(query_text)
                tokens = query_text.split()
                text_query = " ".join([f"{self.escape_lucene(t)}~" for t in tokens])

                raw_res = session.run(
                    hybrid_query,
                    vector_index=NEO4J_REL_DENSE_INDEX,
                    fulltext_index=NEO4J_REL_SPARSE_INDEX,
                    k=initial_k,
                    embedding=emb,
                    text_query=text_query
                ).data()

                if not raw_res:
                    results[query_text] = []
                    continue

                pairs = []
                for r in raw_res:
                    doc_text = f"{r['head']} ({r['head_type']}) - {r['relation_name']} -> {r['tail']} ({r['tail_type']})"
                    pairs.append([query_text, doc_text])

                scores = RERANKER.score(pairs)
                for i, score in enumerate(scores):
                    raw_res[i]['rerank_score'] = float(score)

                final_rels = sorted(raw_res, key=lambda x: x['rerank_score'], reverse=True)[:top_k]
                results[query_text] = final_rels

        return results

    def _expand_subgraph(self, start_node_ids: list,
                         hops: int, limit: int,
                         mapping: Dict[str, str] = None,
                         query: str = "",
                         max_results: int = 15, with_ids: bool = True) -> List[str]:
        """Expands the context by 'hops' hops from the given start nodes."""
        if mapping is None: mapping = {}

        if not start_node_ids or hops < 1:
            return []

        # Traversal query
        # DISTINCT r, so relations are not duplicated if the graph has cycles
        cypher = f"""
            MATCH (start)
            WHERE elementId(start) IN $ids
            MATCH path = (start)-[r*1..{hops}]-(end)
            UNWIND relationships(path) AS rel
            WITH DISTINCT rel
            RETURN elementId(startNode(rel)) as head_id,
                   startNode(rel).name AS head,
                   startNode(rel).type AS head_type,
                   elementId(rel) as rel_id,
                   rel.type AS relation,
                   elementId(endNode(rel)) as tail_id,
                   endNode(rel).name AS tail,
                   endNode(rel).type AS tail_type
            LIMIT $limit
        """

        raw_results = []
        with self.driver.session() as session:
            for res in session.run(cypher, ids=start_node_ids, limit=limit).data():
                s_head_id = self._shorten_id(res['head_id'], 'n', mapping)
                s_tail_id = self._shorten_id(res['tail_id'], 'n', mapping)
                s_rel_id = self._shorten_id(res['rel_id'], 'r', mapping)
                raw_results.append({
                    **res,
                    "s_head_id": s_head_id,
                    "s_tail_id": s_tail_id,
                    "s_rel_id":  s_rel_id,
                })

        if query and len(raw_results) > max_results:
            pairs = [
                [query, f"{r['head']} ({r['head_type']}) - {r['relation']} -> {r['tail']} ({r['tail_type']})"]
                for r in raw_results
            ]
            scores = RERANKER.score(pairs)
            raw_results = [r for _, r in sorted(zip(scores, raw_results), key=lambda x: x[0], reverse=True)]

        if with_ids:
            return [
                f"[ID: {r['s_head_id']}] ({r['head_type']}: {r['head']}) "
                f"--[ID: {r['s_rel_id']} {r['relation']}]--> "
                f"[ID: {r['s_tail_id']}] ({r['tail_type']}: {r['tail']})"
                for r in raw_results[:max_results]
            ]

        else:
            return [
                f"({r['head_type']}: {r['head']}) "
                f"--[{r['relation']}]--> "
                f"({r['tail_type']}: {r['tail']})"
                for r in raw_results[:max_results]
            ]

    def build_nearest_subgraphs(self, nearest_nodes: dict, nearest_relations: list,
                                max_triples: int = 5, max_nodes: int = 8, hops: int = 2, query: str = "") -> str:

        context_parts = []
        collected_ids = set()

        # Relations handling
        # if nearest_relations:
        #     sorted_rels = sorted(
        #         nearest_relations,
        #         key=lambda r: r.get("rerank_score", 0),
        #         reverse=True
        #     )[:max_triples]
        #
        #     lines = []
        #     for idx, rel in enumerate(sorted_rels):
        #         if 'head_id' in rel: collected_ids.add(rel['head_id'])
        #         if 'tail_id' in rel: collected_ids.add(rel['tail_id'])
        #
        #         head = rel.get("head", "UNKNOWN")
        #         head_type = rel.get("head_type", "Node")
        #         # head_id = rel.get("head_id", "UNKNOWN")
        #
        #         tail = rel.get("tail", "UNKNOWN")
        #         tail_type = rel.get("tail_type", "Node")
        #         # tail_id = rel.get("tail_id", "UNKNOWN")
        #
        #         rel_name = rel.get("relation_name", rel.get("relation", "UNKNOWN"))
        #         # rel_id = rel.get("relation_id", "UNKNOWN")
        #
        #         lines.append(
        #             f"Znaleziona relacja {idx}: "
        #             f"({head_type}: {head}) --[{rel_name}]--> ({tail_type}: {tail})"
        #         )
        #
        #     if lines:
        #         context_parts.append("BEZPOŚREDNIE TRAFIENIA (RELACJE):\n" + "\n".join(lines))

        # Nodes handling
        if nearest_nodes:
            all_nodes = []
            for _, items in nearest_nodes.items():
                all_nodes.extend(items)

            # Deduplication by name and type
            seen = set()
            unique_nodes = []
            for n in all_nodes:
                nid = n.get('id')
                if nid and nid not in seen:
                    seen.add(nid)
                    unique_nodes.append(n)
                    collected_ids.add(nid)

            unique_nodes = sorted(unique_nodes, key=lambda n: n.get("rerank_score", 0), reverse=True)[:max_nodes]

            lines = []
            for idx, n in enumerate(unique_nodes):
                lines.append(f"Znaleziony Byt {idx}: {n.get('name')} (Typ: {n.get('type')})")

            if lines:
                context_parts.append("BEZPOŚREDNIE TRAFIENIA (BYTY):\n" + "\n".join(lines))

        # 'Hop' logic (Graph Expansion)
        if hops > 0 and collected_ids:
            neighborhood = self._expand_subgraph(list(collected_ids), hops=hops, limit=60, query=query, max_results=15,
                                                 with_ids=False)

            if neighborhood:
                context_parts.append(
                    f"\nSZERSZY KONTEKST (Sąsiedztwo {hops}-hop):\n" +
                    "Legenda: (<Type>: <Name>) --[<RelType>]--> ...\n" +
                    "\n".join(neighborhood)
                )

        if not context_parts:
            return ""

        return "\n\n".join(context_parts)

    @staticmethod
    def _merge_relation(tx, chunk_id: str, rel: Relation, head_emb: list[float], tail_emb: list[float], rel_emb: list[float]):
        """Creates or updates head and tail, then the relation between them."""
        # MERGE head node
        tx.run(
            """
            MERGE (h:Node {name: $head_name, type: $head_type})
            ON CREATE SET
                h.chunk_ids = [$chunk_id],
                h.embedding = $embedding,
                h.text = $text
            ON MATCH SET h.chunk_ids = CASE WHEN NOT $chunk_id IN h.chunk_ids THEN h.chunk_ids + $chunk_id ELSE h.chunk_ids END
            """,
            head_name=rel.head,
            head_type=rel.head_type,
            chunk_id=chunk_id,
            embedding=head_emb,
            text=f"{rel.head_type}: {rel.head}"
        )

        # MERGE tail node
        tx.run(
            """
            MERGE (t:Node {name: $tail_name, type: $tail_type})
            ON CREATE SET
                t.chunk_ids = [$chunk_id],
                t.embedding = $embedding,
                t.text = $text
            ON MATCH SET t.chunk_ids = CASE WHEN NOT $chunk_id IN t.chunk_ids THEN t.chunk_ids + $chunk_id ELSE t.chunk_ids END
            """,
            tail_name=rel.tail,
            tail_type=rel.tail_type,
            chunk_id=chunk_id,
            embedding=tail_emb,
            text=f"{rel.tail_type}: {rel.tail}"
        )

        # MERGE relation
        tx.run(
            """
            MATCH (h:Node {name: $head_name, type: $head_type})
            MATCH (t:Node {name: $tail_name, type: $tail_type})
            MERGE (h)-[r:RELATED]->(t)
            ON CREATE SET
                r.chunk_ids = [$chunk_id],
                r.type = $original_rel_type,
                r.embedding = $rel_embedding
            ON MATCH SET r.chunk_ids = CASE WHEN NOT $chunk_id IN r.chunk_ids THEN r.chunk_ids + $chunk_id ELSE r.chunk_ids END
            """,
            head_name=rel.head,
            head_type=rel.head_type,
            tail_name=rel.tail,
            tail_type=rel.tail_type,
            chunk_id=chunk_id,
            original_rel_type=rel.relation,
            rel_embedding=rel_emb
        )

    # --- AGENT METHODS (TOOLS IMPLEMENTATION) ---

    def get_schema_summary(self) -> str:
        """[Action: read_graph_schema] Gets unique node types (property 'type') and relation types."""
        query = """
        CALL () {
            MATCH (n:Node) RETURN collect(DISTINCT n.type) as nodeTypes
        }
        CALL () {
            MATCH ()-[r]->() RETURN collect(DISTINCT r.type) as relTypes
        }
        RETURN nodeTypes, relTypes
        """
        with self.driver.session() as session:
            res = session.run(query).single()
            if res:
                return f"Node Types: {res['nodeTypes']}\nRelation Types: {res['relTypes']}"
            return "Schema is empty."

    def search_neighborhood(self, query: str, mapping: Dict[str, str], top_k: int = 3, hops: int = 1) -> str:
        """[Action: search_neighborhood] Finds the nearest nodes and returns their neighborhood as text."""
        # 1. Find entry points (nodes)
        nodes_to_search = [{"name": query, "type": "Node"}]
        nearest_nodes_map = self.find_nearest_nodes(nodes_to_search, top_k)

        # 2. Extract the IDs of the found nodes
        collected_ids = []
        for nodes_list in nearest_nodes_map.values():
            for node in nodes_list:
                if 'id' in node:
                    collected_ids.append(node['id'])

        if not collected_ids:
            return f"No nodes matching the query ‘{query}’ were found."

        # 3. Get the neighborhood (Traversing)
        neighborhood_triples = self._expand_subgraph(collected_ids, hops=hops, limit=20, mapping=mapping)

        if not neighborhood_triples:
            # If there are nodes, but they have no relations
            found_names = [n['name'] for n_list in nearest_nodes_map.values() for n in n_list]
            return f"Nodes found: {found_names}, but they have no relations within a radius of {hops} hops."

        # 4. Format the result for the Agent
        output = [f"Graph surroundings for query ‘{query}’:"]
        for line in neighborhood_triples:
            output.append(f"- {line}")

        return "\n".join(output)

    async def find_nearest_nodes_with_short_ids_async(self, nodes: List[Dict[str, str]], top_k: int = 3,
                                          uuid_map: Dict[str, str] = None) -> Dict[str, List[Dict]]:
        """Like find_nearest_nodes, but each candidate gets a 'short_id' field and uuid_map is filled in-place (short_id -> real elementId)."""
        if uuid_map is None:
            uuid_map = {}

        raw_results = await self.find_nearest_nodes_async(nodes, top_k=top_k)

        enriched = {}
        for key, candidates in raw_results.items():
            enriched_candidates = []
            for candidate in candidates:
                real_id = candidate.get("id", "")
                short_id = self._shorten_id(real_id, "n", uuid_map)
                enriched_candidates.append({**candidate, "short_id": short_id})
            enriched[key] = enriched_candidates

        return enriched

    def get_node_details(self, node_id: str) -> str:
        """[Action: inspect_node_details] Returns the full properties of a node by ID."""
        query = "MATCH (n) WHERE elementId(n) = $id RETURN properties(n) as props"
        with self.driver.session() as session:
            res = session.run(query, id=node_id).single()
            if res:
                props = res['props']
                # Drop embedding and chunk_ids from the view, they are unreadable for the LLM
                if 'embedding' in props:
                    del props['embedding']
                if 'chunk_ids' in props:
                    del props['chunk_ids']

                return json.dumps(props, indent=2, ensure_ascii=False)
            return f"Node with ID {node_id} not found."

    def merge_nodes(self, node_ids: List[str]) -> str:
        """[Action: merge_nodes] Merges nodes into one."""
        if len(node_ids) < 2:
            return "Error: Need at least 2 nodes to merge."

        query = """
        MATCH (n:Node) WHERE elementId(n) IN $ids
        WITH collect(n) as nodes
        CALL apoc.refactor.mergeNodes(nodes, {
            properties: "discard",
            mergeRels: true
        }) YIELD node
        RETURN elementId(node) as new_id, node.name as name
        """
        try:
            with self.driver.session() as session:
                res = session.run(query, ids=node_ids).single()
                return f"Successfully merged nodes into {res['name']}."
        except Exception as e:
            return f"Error merging nodes: {str(e)}"

    def deduplicate_relations_by_ids(self, relation_ids: List[str]) -> str:
        """[Action: deduplicate_relations] Merges the given relations (by ID) into one."""
        if len(relation_ids) < 2:
            return "Error: Need at least 2 relations to merge."

        query = """
        MATCH ()-[r]->() WHERE elementId(r) IN $ids
        WITH collect(r) as rels
        CALL apoc.refactor.mergeRelationships(rels, {properties: "overwrite"})
        YIELD rel
        RETURN elementId(rel) as new_id, rel.type as type
        """
        try:
            with self.driver.session() as session:
                res = session.run(query, ids=relation_ids).single()
                return f"Successfully merged {len(relation_ids)} relations into one {res['type']}."
        except Exception as e:
            return f"Error merging relations: {str(e)}"

    def upsert_triplet(self, head: str, head_type: str, relation: str, tail: str, tail_type: str,
                       chunk_id: Optional[str] = "manual_fix", existing_head_id: Optional[str] = None,
                       existing_tail_id: Optional[str] = None,) -> str:
        """[Action: add_triplet] Adds or updates a triplet."""
        try:
            relation = relation.strip().replace(" ", "_").upper()

            # Generate embeddings
            head_emb = self._embed_node(head, head_type)
            tail_emb = self._embed_node(tail, tail_type)
            rel_emb = self._embed_text(relation)

            rel_obj = Relation(head=head, head_type=head_type, relation=relation, tail=tail, tail_type=tail_type)
            # v7: embeddings of names (without type) and of the verbalized fact
            head_name_emb, tail_name_emb = self.dense.embed_documents([head, tail])
            fact_text = self._verbalize(head, head_type, relation, tail, tail_type)
            fact_emb = self._embed_text(fact_text)

            with self.driver.session() as session:
                session.execute_write(self._merge_relation_v4, chunk_id, rel_obj, head_emb, tail_emb, rel_emb,
                                      existing_head_id,
                                      existing_tail_id,
                                      head_name_emb, tail_name_emb, fact_emb, fact_text)
            return f"Successfully upserted triplets."
        except Exception as e:
            return f"Error adding triplet: {str(e)}"

    @staticmethod
    def _merge_relation_v4(tx, chunk_id: str, rel: "Relation", head_emb: List[float], tail_emb: List[float],
                           rel_emb: List[float], existing_head_id: Optional[str], existing_tail_id: Optional[str],
                           head_name_emb: Optional[List[float]] = None, tail_name_emb: Optional[List[float]] = None,
                           fact_emb: Optional[List[float]] = None, fact_text: Optional[str] = None):

        # HEAD
        if existing_head_id:
            tx.run("""
                MATCH (h:Node) WHERE elementId(h) = $existing_head_id
                SET h.chunk_ids = CASE WHEN NOT $chunk_id IN coalesce(h.chunk_ids, [])
                                  THEN coalesce(h.chunk_ids, []) + $chunk_id ELSE h.chunk_ids END
            """, existing_head_id=existing_head_id, chunk_id=chunk_id)
        else:
            tx.run("""
                MERGE (h:Node {name: $head_name, type: $head_type})
                ON CREATE SET h.chunk_ids = [$chunk_id], h.embedding = $embedding, h.text = $text,
                              h.embedding_name = $embedding_name
                ON MATCH SET h.chunk_ids = CASE WHEN NOT $chunk_id IN h.chunk_ids
                             THEN h.chunk_ids + $chunk_id ELSE h.chunk_ids END
            """, head_name=rel.head, head_type=rel.head_type,
                   embedding=head_emb, text=f"{rel.head_type}: {rel.head}", chunk_id=chunk_id,
                   embedding_name=head_name_emb)

        # TAIL
        if existing_tail_id:
            tx.run("""
                MATCH (t:Node) WHERE elementId(t) = $existing_tail_id
                SET t.chunk_ids = CASE WHEN NOT $chunk_id IN coalesce(t.chunk_ids, [])
                                  THEN coalesce(t.chunk_ids, []) + $chunk_id ELSE t.chunk_ids END
            """, existing_tail_id=existing_tail_id, chunk_id=chunk_id)
        else:
            tx.run("""
                MERGE (t:Node {name: $tail_name, type: $tail_type})
                ON CREATE SET t.chunk_ids = [$chunk_id], t.embedding = $embedding, t.text = $text,
                              t.embedding_name = $embedding_name
                ON MATCH SET t.chunk_ids = CASE WHEN NOT $chunk_id IN t.chunk_ids
                             THEN t.chunk_ids + $chunk_id ELSE t.chunk_ids END
            """, tail_name=rel.tail, tail_type=rel.tail_type,
                   embedding=tail_emb, text=f"{rel.tail_type}: {rel.tail}", chunk_id=chunk_id,
                   embedding_name=tail_name_emb)

        # RELATION: MATCH by existing_id, if given; otherwise by name + type
        head_match = "MATCH (h:Node) WHERE elementId(h) = $existing_head_id" if existing_head_id \
            else "MATCH (h:Node {name: $head_name, type: $head_type})"
        tail_match = "MATCH (t:Node) WHERE elementId(t) = $existing_tail_id" if existing_tail_id \
            else "MATCH (t:Node {name: $tail_name, type: $tail_type})"

        tx.run(f"""
            {head_match}
            {tail_match}
            MERGE (h)-[r:RELATED {type: $relation}]->(t)
            ON CREATE SET r.chunk_ids = [$chunk_id], r.embedding = $rel_embedding,
                          r.embedding_fact = $fact_embedding, r.fact_text = $fact_text
            ON MATCH SET r.chunk_ids = CASE WHEN NOT $chunk_id IN r.chunk_ids
                         THEN r.chunk_ids + $chunk_id ELSE r.chunk_ids END
        """, head_name=rel.head, head_type=rel.head_type,
               tail_name=rel.tail, tail_type=rel.tail_type,
               existing_head_id=existing_head_id, existing_tail_id=existing_tail_id,
               chunk_id=chunk_id, relation=rel.relation, rel_embedding=rel_emb,
               fact_embedding=fact_emb, fact_text=fact_text)

    def update_node_properties(self, node_id: str, new_name: str = None, new_type: str = None) -> str:
        """[Action: update_node_properties] Updates a node's name or type."""
        try:
            with self.driver.session() as session:
                # 1. Get the current data, if any parameter is None
                curr = session.run("MATCH (n) WHERE elementId(n) = $id RETURN n.name as name, n.type as type", id=node_id).single()
                if not curr:
                    return f"Node {node_id} not found."

                final_name = new_name if new_name else curr['name']
                final_type = new_type if new_type else curr['type']

                # 2. Recompute the embedding
                new_emb = self._embed_node(final_name, final_type)
                new_text = f"{final_type}: {final_name}"

                # 3. Update in the database
                query = """
                MATCH (n) WHERE elementId(n) = $id
                SET n.name = $name,
                    n.type = $type,
                    n.text = $text,
                    n.embedding = $emb
                RETURN n.name
                """
                session.run(query, id=node_id, name=final_name, type=final_type, text=new_text, emb=new_emb)

            return f"Successfully updated given node. New state: {final_name} ({final_type})"
        except Exception as e:
            return f"Error updating node: {str(e)}"

    def split_node(self, original_node_id: str, new_name: str, new_type: str, relation_ids_to_move: List[str]) -> str:
        """[Action: split_node_and_move_relations] Creates a new node and moves the given relations from the old node to it."""
        try:
            # Compute the embedding for the new node
            new_emb = self._embed_node(new_name, new_type)
            new_text = f"{new_type}: {new_name}"

            query = """
            MATCH (old) WHERE elementId(old) = $old_id

            // Create new node
            CREATE (new_node:Node {
                name: $new_name,
                type: $new_type,
                text: $new_text,
                embedding: $new_emb,
                created_from_split: $old_id
            })

            // Move relations
            WITH old, new_node
            MATCH (old)-[r]-(neighbor)
            WHERE elementId(r) IN $rel_ids
            CALL apoc.refactor.to(r, new_node) YIELD input, output

            RETURN count(input) as moved_count
            """
            with self.driver.session() as session:
                res = session.run(query,
                                  old_id=original_node_id,
                                  new_name=new_name,
                                  new_type=new_type,
                                  new_text=new_text,
                                  new_emb=new_emb,
                                  rel_ids=relation_ids_to_move).single()

            return f"Successfully split node. Created '{new_name}' and moved {res['moved_count']} relations."
        except Exception as e:
            return f"Error splitting node: {str(e)}"

    def redirect_relation(self, relation_id: str, new_target_node_id: str) -> str:
        """[Action: redirect_relation] Changes the end node of a relation (End Node) to another one."""
        query = """
        MATCH ()-[r]->(), (new_target)
        WHERE elementId(r) = $rid AND elementId(new_target) = $ntid
        WITH r, new_target
        CALL apoc.refactor.to(r, new_target) YIELD input, output
        RETURN startNode(output).name as s, output.type as rel_name, endNode(output).name as t
        """
        try:
            with self.driver.session() as session:
                res = session.run(query, rid=relation_id, ntid=new_target_node_id).single()
                if res:
                    return f"Redirected relation '{res['rel_name']}' to point to: {res['t']}"
                return "Could not redirect relation (ids might be wrong)."
        except Exception as e:
            return f"Error redirecting relation: {str(e)}"

    def delete_element(self, element_id: str, is_node: bool = True) -> str:
        """[Action: delete_element] Deletes a node (with its relations) or a single relation."""
        try:
            with self.driver.session() as session:
                if is_node:
                    # DETACH DELETE removes the node and all relations attached to it
                    query = "MATCH (n) WHERE elementId(n) = $id DETACH DELETE n"
                    msg = f"Successfully deleted given NODE and its relationships."
                else:
                    query = "MATCH ()-[r]->() WHERE elementId(r) = $id DELETE r"
                    msg = f"Successfully deleted given RELATION."

                session.run(query, id=element_id)
                return msg
        except Exception as e:
            return f"Error deleting element: {str(e)}"


    # ------------------------------------------------------------------
    # v7: seeding by name (RRF + filter against the question), fact index,
    #     2-hop expansion with reranking of the whole pool, source sentences next to triples
    # ------------------------------------------------------------------
    TYPE_GROUPS = {
        "Osoba": ("osoba", "człowiek", "polityk", "autor", "ekspert", "dziennikarz", "analityk", "prezes", "minister",
                  "przedsiębiorca", "hodowca", "dyrektor"),
        "Organizacja": ("organizacja", "instytucja", "firma", "spółka", "partia", "bank", "ministerstwo", "urząd",
                        "medium", "agencja", "fundusz", "koalicja", "grupa", "stadnina", "koncern", "giełda", "uczelnia",
                        "związek", "sieć", "przedsiębiorstwo", "resort", "rząd", "sąd", "trybunał", "komisja", "sojusz"),
        "Lokalizacja": ("lokalizacja", "kraj", "państwo", "miasto", "miejsce", "region", "województwo", "obwód",
                        "kontynent", "strefa", "rzeka", "morze", "obszar", "teren", "port", "poligon", "nadleśnictwo",
                        "gmina", "powiat"),
        "Wydarzenie": ("wydarzenie", "zdarzenie", "akcja", "operacja", "wybory", "konflikt", "wojna", "aukcja",
                       "decyzja", "kryzys", "pandemia", "spotkanie", "protest", "demonstracja", "referendum"),
        "Czas": ("czas", "rok", "data", "okres", "miesiąc", "kwartał", "termin", "dzień", "dekada"),
        "Wskaźnik": ("wskaźnik", "procent", "kwota", "liczba", "wartość", "cena", "kurs", "indeks", "stopa", "udział",
                     "poziom", "wolumen", "koszt", "opłata", "pkt", "zmiana", "wzrost", "spadek", "prognoza"),
        "Pojęcie": ("pojęcie", "koncepcja", "zjawisko", "kategoria", "sektor", "rynek", "towar", "produkt", "usługa",
                    "dokument", "ustawa", "umowa", "program", "projekt", "proces", "technologia", "obiekt", "polityka",
                    "strategia", "plan", "raport", "badanie", "surowiec", "energia", "branża"),
    }

    @classmethod
    def _type_group(cls, t: Optional[str]) -> Optional[str]:
        lt = (t or "").strip().lower()
        if not lt:
            return None
        for group, words in cls.TYPE_GROUPS.items():
            for w in words:
                if lt == w or lt.startswith(w):
                    return group
        return None

    @staticmethod
    def _verbalize(head: str, head_type: str, rel: str, tail: str, tail_type: str) -> str:
        rel_txt = (rel or "").replace("_", " ").lower()
        return f"{head} ({head_type}) {rel_txt} {tail} ({tail_type})"

    @staticmethod
    def _clean_doc_title(document_name) -> str:
        name = re.sub(r"\.(pdf|docx?|html?|txt)$", "", str(document_name or ""), flags=re.I)
        name = re.sub(r"^\d{5,}-", "", name)
        return name.replace("-", " ").strip() or "nieznany dokument"

    def _embed_doc(self, text: str) -> list[float]:
        return self.dense.embed_documents([text])[0]

    def _embed_query_text(self, text: str) -> list[float]:
        """Query with the model's instruction prefix (s2p_query) - for matching question -> name/fact."""
        return self.dense.embed_query(text)

    def _fulltext_query_for_names(self, names: List[str]) -> str:
        parts = []
        for nm in names:
            nm = (nm or "").strip().replace('"', " ")
            if not nm:
                continue
            parts.append(f'"{self.escape_lucene(nm)}"^2')
            for tok in re.findall(r"\w+", nm):
                if len(tok) < 3:
                    continue
                t = self.escape_lucene(tok)
                parts.append(f"{t}~1" if len(tok) >= 6 else t)
        return " OR ".join(parts)

    def find_seed_nodes_v7(self, entities: List[Dict], question: str) -> List[Dict]:
        """For each question entity: candidates from the name index (vector + fulltext, RRF fusion), then a relevance filter against the whole question (cross-encoder) and a limit per entity / in total."""
        vec_q = """
        CALL db.index.vector.queryNodes($idx, $k, $emb) YIELD node, score
        RETURN elementId(node) AS id, node.name AS name, node.type AS type, score, COUNT{(node)--()} AS deg
        """
        ft_q = """
        CALL db.index.fulltext.queryNodes($idx, $q, {limit: $k}) YIELD node, score
        RETURN elementId(node) AS id, node.name AS name, node.type AS type, score, COUNT{(node)--()} AS deg
        """
        seeds: Dict[str, Dict] = {}
        with self.driver.session() as session:
            for ent in entities:
                name = (ent.get("name") or "").strip()
                if not name:
                    continue
                names = [name] + [a for a in (ent.get("aliases") or []) if a and a.strip()]
                egroup = self._type_group(ent.get("type"))
                cands: Dict[str, Dict] = {}

                for nm in names[:3]:
                    emb = self._embed_doc(nm)
                    rows = session.run(vec_q, idx=NEO4J_NODE_NAME_DENSE_INDEX, k=V7_SEED_VEC_K, emb=emb).data()
                    for rank, row in enumerate(rows):
                        c = cands.setdefault(row["id"], {**row, "rrf": 0.0, "dense": 0.0, "ft": False})
                        c["rrf"] += 1.0 / (60 + rank)
                        c["dense"] = max(c["dense"], float(row["score"]))

                ft_query = self._fulltext_query_for_names(names)
                if ft_query:
                    try:
                        rows = session.run(ft_q, idx=NEO4J_NODE_NAME_SPARSE_INDEX, q=ft_query, k=V7_SEED_FT_K).data()
                        for rank, row in enumerate(rows):
                            c = cands.setdefault(row["id"], {**row, "rrf": 0.0, "dense": 0.0, "ft": False})
                            c["rrf"] += 1.0 / (60 + rank)
                            c["ft"] = True
                    except Exception as e:
                        logger.warning(f"v7 fulltext query failed for {name!r}: {e}")

                if not cands:
                    continue

                pool = [c for c in cands.values() if c["ft"] or c["dense"] >= V7_SEED_MIN_DENSE]
                if not pool:
                    pool = sorted(cands.values(), key=lambda c: c["dense"], reverse=True)[:3]
                pool = sorted(pool, key=lambda c: c["rrf"], reverse=True)[:10]
                max_rrf = max(c["rrf"] for c in pool) or 1.0

                pairs = [[question, f"{c['name']} ({c['type']})"] for c in pool]
                ce_scores = RERANKER.score(pairs)
                for c, s in zip(pool, ce_scores):
                    c["ce"] = float(s)
                    c["type_bonus"] = 0.05 if (egroup and self._type_group(c["type"]) == egroup) else 0.0
                    c["rrf_n"] = c["rrf"] / max_rrf
                names_l = {n.strip().lower() for n in names}
                for c in pool:
                    c["exact"] = (c["name"] or "").strip().lower() in names_l
                best_ce = max(c["ce"] for c in pool)
                kept = [c for c in pool if c["exact"] or c["ce"] >= max(V7_SEED_MIN_CE, V7_SEED_REL_CE * best_ce)]
                kept = sorted(kept, key=lambda c: (c["exact"], c["ce"] + c["type_bonus"] + 0.15 * c["rrf_n"]), reverse=True)[:V7_SEED_PER_ENTITY]

                logger.debug("v7 seeds for {!r}: {}", name, [(c["name"], c["type"], round(c["ce"], 3), round(c["dense"], 3), c["ft"]) for c in pool[:6]])

                for c in kept:
                    score = min(1.0, max(c["ce"] + c["type_bonus"], V7_SEED_EXACT_SCORE if c.get("exact") else 0.0))
                    if c["id"] not in seeds or seeds[c["id"]]["score"] < score:
                        seeds[c["id"]] = {"id": c["id"], "name": c["name"], "type": c["type"], "score": score,
                                          "deg": c["deg"], "entity": name}

        return sorted(seeds.values(), key=lambda s: s["score"], reverse=True)[:V7_SEED_MAX]

    def find_facts_v7(self, question: str, top_k: int = V7_FACT_TOP) -> List[Dict]:
        """Question -> index of verbalized triples (facts), then cross-encoder against the question."""
        q = """
        CALL db.index.vector.queryRelationships($idx, $k, $emb) YIELD relationship AS r, score
        RETURN elementId(r) AS rel_id, r.type AS rel, r.chunk_ids AS chunk_ids, score,
               elementId(startNode(r)) AS head_id, startNode(r).name AS head, startNode(r).type AS head_type,
               elementId(endNode(r)) AS tail_id, endNode(r).name AS tail, endNode(r).type AS tail_type
        """
        try:
            emb = self._embed_query_text(question)
            with self.driver.session() as session:
                rows = session.run(q, idx=NEO4J_FACT_DENSE_INDEX, k=V7_FACT_K, emb=emb).data()
        except Exception as e:
            logger.warning(f"v7 fact search failed: {e}")
            return []
        if not rows:
            return []
        pairs = [[question, self._verbalize(r["head"], r["head_type"], r["rel"], r["tail"], r["tail_type"])] for r in rows]
        for r, s in zip(rows, RERANKER.score(pairs)):
            r["ce"] = float(s)
        rows = [r for r in rows if r["ce"] >= V7_FACT_MIN_CE]
        return sorted(rows, key=lambda r: r["ce"], reverse=True)[:top_k]

    def _expand_v7(self, seeds: List[Dict]) -> Dict[str, Dict]:
        """1-hop from seeds (limit per seed, edges with more chunks first) and 2-hop from neighbors, skipping hubs."""
        if not seeds:
            return {}
        one_hop_q = """
        UNWIND $seeds AS s
        MATCH (a:Node) WHERE elementId(a) = s.id
        MATCH (a)-[r:RELATED]-(b)
        WITH s, r, b ORDER BY size(coalesce(r.chunk_ids, [])) DESC
        WITH s, collect({r: r, b: b})[..$per] AS items
        UNWIND items AS it
        WITH s, it.r AS r, it.b AS b
        RETURN s.score AS seed_score, elementId(r) AS rel_id, r.type AS rel, r.chunk_ids AS chunk_ids,
               elementId(startNode(r)) AS head_id, startNode(r).name AS head, startNode(r).type AS head_type,
               elementId(endNode(r)) AS tail_id, endNode(r).name AS tail, endNode(r).type AS tail_type,
               elementId(b) AS nb_id, COUNT{(b)--()} AS nb_deg
        """
        two_hop_q = """
        UNWIND $nodes AS s
        MATCH (a:Node) WHERE elementId(a) = s.id
        MATCH (a)-[r:RELATED]-(b)
        WHERE NOT elementId(b) IN $exclude
        WITH s, r, b ORDER BY size(coalesce(r.chunk_ids, [])) DESC
        WITH s, collect({r: r, b: b})[..$per] AS items
        UNWIND items AS it
        WITH s, it.r AS r, it.b AS b
        RETURN s.score AS seed_score, elementId(r) AS rel_id, r.type AS rel, r.chunk_ids AS chunk_ids,
               elementId(startNode(r)) AS head_id, startNode(r).name AS head, startNode(r).type AS head_type,
               elementId(endNode(r)) AS tail_id, endNode(r).name AS tail, endNode(r).type AS tail_type
        """
        pool: Dict[str, Dict] = {}
        seed_ids = {s["id"] for s in seeds}
        frontier: Dict[str, float] = {}
        with self.driver.session() as session:
            rows = session.run(one_hop_q, seeds=[{"id": s["id"], "score": float(s["score"])} for s in seeds],
                               per=V7_PER_SEED_LIMIT).data()
            for row in rows:
                struct = float(row["seed_score"])
                t = pool.get(row["rel_id"])
                if t is None or t["struct"] < struct:
                    pool[row["rel_id"]] = {**row, "struct": struct, "hop": 1}
                nb, nb_deg = row["nb_id"], row["nb_deg"]
                if nb not in seed_ids and nb_deg <= V7_HUB_MAX_DEGREE:
                    frontier[nb] = max(frontier.get(nb, 0.0), struct * V7_HOP_DECAY)

            if V7_HOPS >= 2 and frontier:
                nodes = sorted(({"id": k, "score": v} for k, v in frontier.items()), key=lambda x: x["score"], reverse=True)[:60]
                rows = session.run(two_hop_q, nodes=nodes, exclude=list(seed_ids), per=V7_PER_NODE_LIMIT_2HOP).data()
                for row in rows:
                    struct = float(row["seed_score"])
                    t = pool.get(row["rel_id"])
                    if t is None:
                        pool[row["rel_id"]] = {**row, "struct": struct, "hop": 2}
                    elif t["struct"] < struct:
                        t["struct"] = struct
        return pool

    def _attach_evidence_v7(self, triples: List[Dict]) -> Dict[str, Dict]:
        """For each triple picks the sentence from the source chunk that contains head and tail (matching on token prefixes, numbers exactly)."""
        from app.repo import retriever  # local import: avoids the circular import app.repo -> graph_repo

        chunk_ids = sorted({c for t in triples for c in (t.get("chunk_ids") or [])})
        if not chunk_ids:
            return {}
        try:
            docs = retriever.get_chunk_documents_by_ids(chunk_ids)
        except Exception as e:
            logger.warning(f"v7 evidence: failed to fetch chunks: {e}")
            return {}

        def norm(w: str) -> Optional[str]:
            w = w.lower()
            if w.isdigit() or len(w) >= 3:
                return w[:5] if len(w) >= 6 else w
            return None

        def toks(s: str) -> set:
            return {n for n in (norm(w) for w in re.findall(r"\w+", s or "")) if n}

        def window(sent: str, wanted: set) -> str:
            if len(sent) <= V7_EVIDENCE_MAX_CHARS:
                return sent
            positions = [m.start() for m in re.finditer(r"\w+", sent) if norm(m.group()) in wanted]
            if not positions:
                return sent[:V7_EVIDENCE_MAX_CHARS]
            center = (min(positions) + max(positions)) // 2
            start = max(0, min(center - V7_EVIDENCE_MAX_CHARS // 2, len(sent) - V7_EVIDENCE_MAX_CHARS))
            end = min(len(sent), start + V7_EVIDENCE_MAX_CHARS)
            return ("…" if start > 0 else "") + sent[start:end].strip() + ("…" if end < len(sent) else "")

        by_chunk: Dict[str, Dict] = {}
        for d in docs:
            cid = d.metadata.get("chunk_id")
            text = re.sub(r"\s+", " ", d.page_content or "")
            sents = [x.strip() for x in re.split(r"(?<=[.!?])\s+(?=[A-ZĄĆĘŁŃÓŚŹŻ0-9„\"(])", text) if x.strip()]
            by_chunk[cid] = {"title": self._clean_doc_title(d.metadata.get("document_name")),
                             "sents": [(x, toks(x)) for x in sents], "used": set()}

        for t in triples:
            ht, tt = toks(t["head"]), toks(t["tail"])
            best, best_score, title, best_cid, best_idx = None, 0.0, None, None, None
            for cid in (t.get("chunk_ids") or []):
                info = by_chunk.get(cid)
                if not info:
                    continue
                title = title or info["title"]
                for idx, (x, st) in enumerate(info["sents"]):
                    hm, tm = ht & st, tt & st
                    if not ht or not tt:
                        continue
                    if hm and tm:
                        score = len(hm) / len(ht) + len(tm) / len(tt)
                    elif hm or tm:
                        side = hm or tm
                        if not (len(side) >= 2 or any(w.isdigit() for w in side)):
                            continue
                        score = 0.5 * (len(hm) / len(ht) + len(tm) / len(tt))
                    else:
                        continue
                    score -= 0.0003 * len(x)
                    if score > best_score:
                        best_score, best, best_cid, best_idx = score, x, cid, idx
            if best is not None:
                info = by_chunk[best_cid]
                info["used"].add(best_idx)
                # v7.1: two-sentence window - the next sentence often completes the fact (effect, purpose, value)
                if best_idx + 1 < len(info["sents"]):
                    nxt = info["sents"][best_idx + 1][0]
                    if len(best) + len(nxt) + 1 <= V7_EVIDENCE_WINDOW_MAX:
                        best = best + " " + nxt
                        info["used"].add(best_idx + 1)
                t["evidence"] = best if len(best) <= V7_EVIDENCE_WINDOW_MAX else window(best, ht | tt)
                t["source_chunk"] = best_cid
            else:
                t["evidence"] = None
            t["doc_title"] = title
        return by_chunk

    def _extra_sentences_v7(self, question: str, triples: List[Dict], by_chunk: Dict[str, Dict]) -> List[tuple]:
        """v7.1: from the articles the most relevant triples come from, picks extra sentences chosen by the cross-encoder against the question (skips sentences already used as triple sources)."""
        if not by_chunk:
            return []
        order: List[str] = []
        for t in triples:
            for cid in (t.get("chunk_ids") or []):
                if cid in by_chunk and cid not in order:
                    order.append(cid)
        cands = []
        for cid in order[:V7_EXTRA_CHUNKS]:
            info = by_chunk[cid]
            for idx, (x, _) in enumerate(info["sents"]):
                if idx in info["used"] or len(x) < 40 or len(x) > 600:
                    continue
                cands.append((cid, idx, x))
        if not cands:
            return []
        scores = RERANKER.score([[question, x] for _, _, x in cands])
        per_chunk: Dict[str, List[tuple]] = {}
        for (cid, idx, x), sc in zip(cands, scores):
            if float(sc) >= V7_EXTRA_MIN_CE:
                per_chunk.setdefault(cid, []).append((float(sc), idx, x))
        out = []
        for cid in order[:V7_EXTRA_CHUNKS]:
            top = sorted(per_chunk.get(cid, []), reverse=True)[:V7_EXTRA_SENT_PER_CHUNK]
            for sc, idx, x in sorted(top, key=lambda z: z[1]):
                out.append((x, by_chunk[cid]["title"], sc))
        return out

    def build_context_v7(self, seeds: List[Dict], facts: List[Dict], question: str) -> str:
        pool = self._expand_v7(seeds)
        for f in facts:  # direct facts from the index: structure = relevance to the question
            t = pool.get(f["rel_id"])
            if t is None:
                pool[f["rel_id"]] = {**f, "struct": float(f["ce"]), "hop": 0}
            else:
                t["struct"] = max(t["struct"], float(f["ce"]))
        if not pool:
            return ""

        cands = sorted(pool.values(), key=lambda t: t["struct"], reverse=True)[:V7_POOL_MAX]
        pairs = [[question, self._verbalize(t["head"], t["head_type"], t["rel"], t["tail"], t["tail_type"])] for t in cands]
        for t, s in zip(cands, RERANKER.score(pairs)):
            t["ce"] = float(s)
            t["final"] = V7_CE_WEIGHT * t["ce"] + (1.0 - V7_CE_WEIGHT) * float(t["struct"])
        top = sorted(cands, key=lambda t: t["final"], reverse=True)[:V7_BUDGET_TRIPLES]
        by_chunk = (self._attach_evidence_v7(top) or {}) if V7_INCLUDE_EVIDENCE else {}
        extra = self._extra_sentences_v7(question, top, by_chunk) if (V7_INCLUDE_EVIDENCE and V7_INCLUDE_EXTRA_SENTENCES) else []

        logger.info("v7 graph context: seeds={} facts={} pool={} -> {} triples (hop0={}, hop1={}, hop2={}, with_evidence={})",
                    len(seeds), len(facts), len(pool), len(top),
                    sum(1 for t in top if t["hop"] == 0), sum(1 for t in top if t["hop"] == 1),
                    sum(1 for t in top if t["hop"] == 2), sum(1 for t in top if t.get("evidence")))

        lines = ["BEZPOŚREDNIE TRAFIENIA (BYTY):"]
        lines += [f"- {s['name']} (Typ: {s['type']})" for s in seeds] or ["- brak"]
        lines += ["", "FAKTY Z GRAFU WIEDZY (od najbardziej trafnych):",
                  "Legenda: (<Typ>: <Nazwa>) --[<RELACJA>]--> (<Typ>: <Nazwa>); pod faktem zdanie źródłowe [tytuł artykułu]"]
        total = sum(len(l) for l in lines)
        for i, t in enumerate(top, 1):
            block = f"{i}. ({t['head_type']}: {t['head']}) --[{t['rel']}]--> ({t['tail_type']}: {t['tail']})"
            if t.get("evidence"):
                block += f"\n   Źródło: „{t['evidence']}” [{t.get('doc_title') or 'nieznany dokument'}]"
            elif t.get("doc_title"):
                block += f"\n   [{t['doc_title']}]"
            if total + len(block) > V7_CONTEXT_MAX_CHARS:
                break
            lines.append(block)
            total += len(block)
        if extra:
            lines += ["", "DODATKOWE ZDANIA Z ARTYKUŁÓW ŹRÓDŁOWYCH (wybrane pod pytanie):"]
            for x, title, sc in extra:
                block = f"- „{x}” [{title}]"
                if total + len(block) > V7_CONTEXT_MAX_CHARS + 2500:
                    break
                lines.append(block)
                total += len(block)
        return "\n".join(lines)


if __name__ == '__main__':
    import pprint
    g = GraphRepository()
    try:
        # print("--- TEST: Similar Node Retrieval ONLY ---")
        #
        # article_1_nodes = [
        #     {"name": "OpenAI", "type": "Organizacja"},
        #     {"name": "Polska", "type": "Kraj"},
        #     {"name": "ChatGPT", "type": "Produkt"},
        #     {"name": "AI Summit Warsaw 2025", "type": "Wydarzenie"},
        #     {"name": "Anna Kowalska", "type": "Osoba"},
        #     {"name": "NCBR", "type": "Organizacja"}
        # ]
        #
        # article_2_nodes = [
        #     {"name": "OpenAI", "type": "Organizacja"},
        #     {"name": "ChatGPT", "type": "Produkt"},
        #     {"name": "AI Summit Warsaw", "type": "Wydarzenie"},
        #     {"name": "dr Anna Kowalska", "type": "Osoba"},
        #     {"name": "Narodowe Centrum Badań i Rozwoju", "type": "Organizacja"}
        # ]
        #
        # print("\n[1] Article 1 - nearest nodes")
        # nodes_map_1 = g.find_nearest_nodes(article_1_nodes, top_k=5)
        #
        # print("\n[2] Article 2 - nearest nodes")
        # nodes_map_2 = g.find_nearest_nodes(article_2_nodes, top_k=5)
        #
        # print("\n--- RESULTS ARTICLE 1 ---")
        # pprint.pprint(nodes_map_1)
        #
        # print("\n--- RESULTS ARTICLE 2 ---")
        # pprint.pprint(nodes_map_2)
        #
        # print("\n[3] Checking overlap (recall test)")
        #
        # def extract_names(nodes_map):
        #     names = set()
        #     for v in nodes_map.values():
        #         for n in v:
        #             names.add(f"{n['type']}: {n['name']}")
        #     return names
        #
        # names_1 = extract_names(nodes_map_1)
        # names_2 = extract_names(nodes_map_2)
        #
        # common = names_1.intersection(names_2)
        #
        # print("\n--- COMMON NODES ---")
        # for c in common:
        #     print(c)
        #
        # print(f"\nOverlap: {len(common)}")

        pprint.pprint(g.find_nearest_nodes([{"name": "ceny materiałów budowlanych", "type": "Wskaźnik"}]))
        pprint.pprint(g.find_nearest_nodes([{"name": "ceny surowców", "type": "Ceny"}]))
        pprint.pprint(g.find_nearest_nodes([{"name": "wzrost cen surowców", "type": "Koncepcja"}]))
        pprint.pprint(g.find_nearest_nodes([{"name": "12", "type": "Wskaźnik"}]))

        print("_"*100)
        pprint.pprint(g.build_nearest_subgraphs(g.find_nearest_nodes([{"name": "ceny materiałów budowlanych", "type": "Wskaźnik"}]), [], query="Jakie były zmiany cen materiałów budowlanych w 2021 roku?"))
        print("_"*100)
        pprint.pprint(g.build_nearest_subgraphs(g.find_nearest_nodes([{"name": "ceny surowców", "type": "Ceny"}]), [], query="Jakie były zmiany cen materiałów budowlanych w 2021 roku?"))
        print("_"*100)
        pprint.pprint(g.build_nearest_subgraphs(g.find_nearest_nodes([{"name": "wzrost cen surowców", "type": "Koncepcja"}]), [], query="Jakie były zmiany cen materiałów budowlanych w 2021 roku?"))
        print("_"*100)
        pprint.pprint(g.build_nearest_subgraphs(g.find_nearest_nodes([{"name": "12", "type": "Wskaźnik"}]), [], query="Jakie były zmiany cen materiałów budowlanych w 2021 roku?"))

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

    finally:
        g.close()
