import re
from langchain_ollama import ChatOllama
from typing import Optional, List, Dict
from langchain_openai import ChatOpenAI
from loguru import logger
from app.config.settings import LLM_MODEL_NAME, REASONING, OPENAI_API_KEY
from app.models.graph_dto import NodeResolutionResponse, NodeResolution
from app.prompts.node_dedup_v4 import ND_V4_PROMPT
from app.prompts.node_types_for_question_graph import NODE_CLASSIFIER_PROMPT
from app.prompts.nodes import NODES_EXTRACTION_PROMPT
from app.prompts.question_graph import QUESTION_RELATIONS_EXTRACTION_PROMPT
from app.prompts.relations import build_prompt
from app.utils.text import extract_json_from_text, extract_json_object
from app.prompts.question_analysis import QUESTION_ANALYSIS_PROMPT


class GraphBuilder:
    def __init__(self, model_name: str = LLM_MODEL_NAME):
        if OPENAI_API_KEY and OPENAI_API_KEY.strip():
            self.llm = ChatOpenAI(model=LLM_MODEL_NAME, temperature=0.0, api_key=OPENAI_API_KEY)
            logger.info(f"Using OpenAI model: {LLM_MODEL_NAME}")
        else:
            self.llm = ChatOllama(model=model_name, temperature=0.0, reasoning=REASONING, num_ctx=131000, num_predict=-2, keep_alive="30h")
            logger.info(f"Using Ollama model: {LLM_MODEL_NAME}")

    async def extract_nodes(self, text: str) -> List[str]:
        logger.info("Extracting nodes")

        nodes_extraction_chain = NODES_EXTRACTION_PROMPT | self.llm
        nodes_extracted_content_raw = await nodes_extraction_chain.ainvoke({"text": text})
        nodes = [node.strip() for node in nodes_extracted_content_raw.content.split(";")]
        logger.debug(f"Extracted {len(nodes)} nodes")
        return nodes

    async def extract_relations(self, text: str, nodes: str, rels: Optional[str] = None) -> dict:
        logger.info("Extracting relations")

        rels_generator_prompt = build_prompt(nodes, rels)
        rels_generator_chain = rels_generator_prompt | self.llm

        rels_raw = await rels_generator_chain.ainvoke({"input": text})
        raw_content = "".join(rels_raw.content)

        extracted_json = extract_json_from_text(raw_content)
        if extracted_json:
            logger.debug(f"Number of extracted relations: {len(extracted_json)}")
        else:
            logger.debug("No relations extracted")

        return extracted_json

    # Graph – all-in-one
    async def build_graph(self, text: str) -> dict:
        logger.info(f"Building full graph. Text length: {len(text)} characters.")

        nodes_list = await self.extract_nodes(text)
        nodes_str = "; ".join(nodes_list)

        relations = await self.extract_relations(text, nodes_str)
        return {"graph": relations}

    ALLOWED_QUESTION_TYPES = {"Osoba", "Organizacja", "Lokalizacja", "Wydarzenie", "Czas", "Pojęcie", "Technologia", "Obiekt", "Wskaźnik"}

    async def analyze_questions(self, questions: List[str]) -> dict:
        """v7: one LLM call -> canonical entities (with periphrases resolved), aliases, keywords, answer type and the question rewritten without periphrases."""
        text = "\n".join(questions)
        chain = QUESTION_ANALYSIS_PROMPT | self.llm
        data: dict = {}
        try:
            raw = await chain.ainvoke({"text": text})
            data = extract_json_object(raw.content) or {}
        except Exception as e:
            logger.warning(f"analyze_questions: failed to parse the LLM response ({e}); falling back to node extraction")

        entities, nodes, seen = [], [], set()
        for e in data.get("entities", []) or []:
            if not isinstance(e, dict):
                continue
            name = str(e.get("name", "")).strip()
            tp = str(e.get("type", "")).strip() or "Pojęcie"
            if tp not in self.ALLOWED_QUESTION_TYPES:
                tp = "Pojęcie"
            if not name or name.lower() in seen:
                continue
            seen.add(name.lower())
            aliases = [str(a).strip() for a in (e.get("aliases") or []) if str(a).strip() and str(a).strip().lower() != name.lower()]
            entities.append({"name": name, "type": tp, "aliases": aliases[:4]})
            nodes.append(f"{name}: {tp}")

        if not entities:  # fallback: old node extraction from the question text
            for name in await self.extract_nodes(text):
                if name and name.lower() not in seen:
                    seen.add(name.lower())
                    entities.append({"name": name, "type": "Pojęcie", "aliases": []})
                    nodes.append(f"{name}: Pojęcie")

        keywords = [str(k).strip() for k in (data.get("keywords") or []) if str(k).strip()][:6]
        answer_type = str(data.get("answer_type") or "").strip() or None
        canonical = str(data.get("canonical_question") or "").strip() or None

        logger.debug(f"analyze_questions: entities={[(e['name'], e['type']) for e in entities]} keywords={keywords} "
                     f"answer_type={answer_type} canonical={canonical!r}")
        return {"nodes": nodes, "relations": [], "entities": entities, "keywords": keywords,
                "answer_type": answer_type, "canonical_question": canonical}

    async def gen_questions_graph_for_qa(self, questions_str: str) -> dict:
        """Version v2-v6 (3 LLM calls). Kept for comparison."""
        raw_nodes = await self.extract_nodes(questions_str)
        nodes_to_classify = "; ".join(raw_nodes)
        classifier_chain = NODE_CLASSIFIER_PROMPT | self.llm
        classified_raw = await classifier_chain.ainvoke({"nodes": "List: " + nodes_to_classify})
        classified_pairs = re.findall(r"\s*([^;]+?)\s*\(\s*([^)]+?)\s*\)\s*(?:;|$)", classified_raw.content)
        nodes_with_types = [f"{name.strip()}: {ctype.strip()}" for name, ctype in classified_pairs]

        q_rels_extraction_chain = QUESTION_RELATIONS_EXTRACTION_PROMPT | self.llm
        q_rels_raw = await q_rels_extraction_chain.ainvoke({"text": questions_str})
        q_rels_list = [rel.strip().upper() for rel in q_rels_raw.content.split("\n") if rel.strip()]

        return {"nodes": nodes_with_types, "relations": q_rels_list}


    async def resolve_node_identity(self, chunk_text: str, input_nodes: List[Dict[str, str]],
                                    candidates_map: Dict[str, List[Dict]],
                                    source_library: Dict[int, str]) -> NodeResolutionResponse:

        logger.info(f"Resolving identity for {len(input_nodes)} nodes")

        if source_library:
            source_lines = [f"[SOURCE {idx}]:\n{txt}" for idx, txt in source_library.items()]  # txt[:MAX_SOURCE_LENGTH]
            source_section = "SOURCE LIBRARY (referenced by candidates below):\n\n" + "\n\n".join(source_lines)
        else:
            source_section = "SOURCE LIBRARY: (empty - no existing nodes with source chunks found)"

        input_entities_lines = []
        for node in input_nodes:
            input_entities_lines.append(f'- "{node["name"]}" (type: {node["type"]})')
        input_entities_str = "\n".join(input_entities_lines)

        candidates_lines = []
        for query_key, candidates in candidates_map.items():
            raw_name = query_key.split(":", 1)[-1].strip()

            if not candidates:
                candidates_lines.append(f'"{raw_name}" -> no candidates found in graph\n')
                continue

            cand_blocks = []
            for c in candidates:
                ref = c.get("source_ref")
                source_part = f" source_ref={ref}" if ref is not None else " source_ref=none"
                cand_blocks.append(
                    f'  - short_id="{c["short_id"]}" name="{c["name"]}" type="{c["type"]}"'
                    f'{source_part}'
                )

            candidates_lines.append(f'"{raw_name}" candidates:\n' + "\n".join(cand_blocks))

        candidates_str = "\n\n".join(candidates_lines)

        resolution_chain = ND_V4_PROMPT | self.llm
        raw = await resolution_chain.ainvoke({"chunk_text": chunk_text, "input_entities": input_entities_str,
                                              "source_section": source_section, "candidates": candidates_str})

        raw_content = raw.content
        logger.debug(f"Raw LLM resolution response:\n{raw_content}")

        parsed = extract_json_from_text(raw_content)
        if not parsed or not isinstance(parsed, list):
            logger.warning("resolve_node_identity: failed to parse LLM output, returning empty resolutions")
            return NodeResolutionResponse(resolutions=[])

        resolutions = []
        for item in parsed:
            try:
                resolutions.append(NodeResolution(**item))
            except Exception as e:
                logger.warning(f"Skipping malformed resolution item {item}: {e}")

        logger.info(f"Resolved {len(resolutions)} nodes - {sum(1 for r in resolutions if r.matched_id)} matched, "
                    f"{sum(1 for r in resolutions if not r.matched_id)} new")

        return NodeResolutionResponse(resolutions=resolutions)
