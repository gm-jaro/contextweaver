"""v7: builds (without destroying existing data) extra embeddings and indexes in Neo4j"""
import sys, os, time
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from loguru import logger
from neo4j import GraphDatabase
from app import DENSE_MODEL
from app.config.settings import NEO4J_URL, NEO4J_USER, NEO4J_PASSWORD
from app.config.costants import NEO4J_NODE_NAME_DENSE_INDEX, NEO4J_NODE_NAME_SPARSE_INDEX, NEO4J_FACT_DENSE_INDEX
from app.repo.graph_repo import GraphRepository

BATCH = 64


def main(force: bool = False):
    driver = GraphDatabase.driver(NEO4J_URL, auth=(NEO4J_USER, NEO4J_PASSWORD))
    t0 = time.time()
    with driver.session() as s:
        where = "" if force else "WHERE n.embedding_name IS NULL"
        nodes = s.run(f"MATCH (n:Node) {where} RETURN elementId(n) AS id, n.name AS name").data()
        logger.info(f"Nodes to embed: {len(nodes)}")
        for i in range(0, len(nodes), BATCH):
            batch = nodes[i:i + BATCH]
            embs = DENSE_MODEL.embed_documents([b["name"] or "" for b in batch])
            s.run("UNWIND $rows AS row MATCH (n:Node) WHERE elementId(n) = row.id SET n.embedding_name = row.emb",
                  rows=[{"id": b["id"], "emb": e} for b, e in zip(batch, embs)])
            if (i // BATCH) % 10 == 0:
                logger.info(f"  nodes {min(i + BATCH, len(nodes))}/{len(nodes)}")

        where = "" if force else "WHERE r.embedding_fact IS NULL"
        rels = s.run(f"""MATCH (h:Node)-[r:RELATED]->(t:Node) {where}
                         RETURN elementId(r) AS id, h.name AS head, h.type AS head_type, r.type AS rel,
                                t.name AS tail, t.type AS tail_type""").data()
        logger.info(f"Relations to embed: {len(rels)}")
        for i in range(0, len(rels), BATCH):
            batch = rels[i:i + BATCH]
            texts = [GraphRepository._verbalize(b["head"], b["head_type"], b["rel"], b["tail"], b["tail_type"]) for b in batch]
            embs = DENSE_MODEL.embed_documents(texts)
            s.run("UNWIND $rows AS row MATCH ()-[r:RELATED]->() WHERE elementId(r) = row.id "
                  "SET r.embedding_fact = row.emb, r.fact_text = row.text",
                  rows=[{"id": b["id"], "emb": e, "text": tx} for b, e, tx in zip(batch, embs, texts)])
            if (i // BATCH) % 10 == 0:
                logger.info(f"  relations {min(i + BATCH, len(rels))}/{len(rels)}")

        dim = len(DENSE_MODEL.embed_documents(["test"])[0])
        s.run(f"""CREATE VECTOR INDEX {NEO4J_NODE_NAME_DENSE_INDEX} IF NOT EXISTS FOR (n:Node) ON (n.embedding_name)
                  OPTIONS {{indexConfig: {{`vector.dimensions`: {dim}, `vector.similarity_function`: 'cosine'}}}}""")
        s.run(f"CREATE FULLTEXT INDEX {NEO4J_NODE_NAME_SPARSE_INDEX} IF NOT EXISTS FOR (n:Node) ON EACH [n.name]")
        s.run(f"""CREATE VECTOR INDEX {NEO4J_FACT_DENSE_INDEX} IF NOT EXISTS FOR ()-[r:RELATED]-() ON (r.embedding_fact)
                  OPTIONS {{indexConfig: {{`vector.dimensions`: {dim}, `vector.similarity_function`: 'cosine'}}}}""")
        s.run("CALL db.awaitIndexes(300)")
        idx = s.run("SHOW INDEXES YIELD name, type, state RETURN name, type, state").data()
        logger.info(f"Indexes: {[(i['name'], i['type'], i['state']) for i in idx]}")
    driver.close()
    logger.success(f"Done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main(force="--force" in sys.argv)
