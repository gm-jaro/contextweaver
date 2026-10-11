from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, START, END
from langchain_core.tools import tool
from loguru import logger
from app.service.ner import NER, RelationsPipeline
import wikipedia


wikipedia.set_lang("pl")
ner = NER("pl")
rel = RelationsPipeline(ner)


class GraphState(TypedDict):
    text: str
    raw_entities: List[str]
    unique_entities: List[str]
    linked_entities: List[Dict]
    relations: Dict
    final_graph: Dict


@tool
def ner_extract(text: str) -> list[str]:
    """Extract raw entities using HuggingFace pipeline."""
    logger.info("TOOL: Running NER extraction")
    entities = [e['word'] for e in ner(text)]
    return entities

# TODO: Add node to split Uniwersytetu Jagiellońskiego w Krakowie into Uniwersytetu Jagiellońskiego and Krakowie (two separate NEs)

@tool
def search_duplicates(raw_entities: list[str]) -> list[str]:
    """Clean and deduplicate entities."""
    logger.info("TOOL: Deduplicating entities")
    unique_entities = sorted(list(set(raw_entities)))
    return unique_entities


@tool
def wiki_link(entities: list[str]) -> list[dict]:
    """Link entities to Wikipedia."""
    # TODO: This function should eliminate duplicates like WOŚP when wikipedia.search returns same results for "WOŚP" and "Wielka Orkiestra Świątecznej Pomocy"
    #  Duplicated values (other than main search results) should be treated as aliases and added to wiki_data['aliases'] (maybe by search_duplicates function, for which separate node is not needed - deduplication in wiki_link on the fly)
    logger.info("TOOL: Linking to Wiki")
    linked = []
    for ent in entities:
        wiki_data = {"entity": ent, "wiki_name": None, "url": None, "wiki_id": None}
        try:
            results = wikipedia.search(ent, results=1)
            if results:
                page_title = results[0]
                wiki_data.update({
                    "wiki_name": page_title,
                    "url": f"https://pl.wikipedia.org/wiki/{page_title.replace(' ', '_')}",
                    "ent_id": f"wiki_{hash(page_title)}"
                })
        except Exception as e:
            logger.warning(f"Wiki error for {ent}: {e}")

        linked.append(wiki_data)
    return linked

@tool
def extract_relations(text: str, entities: list[str]) -> dict:
    """Extract relations between entities using LLM."""
    logger.info("TOOL: Extracting relations via LLM")
    relations = rel(text, entities)
    return relations

@tool
def assemble_graph(entities: list[dict], relations: dict) -> dict:
    """Assemble the final JSON graph structure."""
    logger.info("TOOL: Assembling final graph")

    nodes = []
    for item in entities:
        nodes.append({
            "id": item["entity"],
            "metadata": {
                "wiki_url": item["url"],
                "wiki_id": item["wiki_id"]
            }
        })

    edges = relations.get("relations", [])

    return {
        "graph": {
            "nodes": nodes,
            "edges": edges
        }
    }


def node_ner(state: GraphState):
    text = state["text"]
    raw = ner_extract.invoke(text)
    return {"raw_entities": raw}


def node_dedup(state: GraphState):
    raw = state["raw_entities"]
    unique = search_duplicates.invoke({"raw_entities": raw})
    return {"unique_entities": unique}


def node_link(state: GraphState):
    unique = state["unique_entities"]
    linked = wiki_link.invoke({"entities": unique})
    return {"linked_entities": linked}


def node_relations(state: GraphState):
    text = state["text"]
    unique = state["unique_entities"]
    rels = extract_relations.invoke({"text": text, "entities": unique})
    return {"relations": rels}


def node_assemble(state: GraphState):
    linked = state["linked_entities"]
    rels = state["relations"]
    final = assemble_graph.invoke({"entities": linked, "relations": rels})
    return {"final_graph": final}


workflow = StateGraph(GraphState)

workflow.add_node("ner", node_ner)
workflow.add_node("deduplicate", node_dedup)
workflow.add_node("wiki_linker", node_link)
workflow.add_node("relation_extractor", node_relations)
workflow.add_node("assembler", node_assemble)

workflow.add_edge(START, "ner")
workflow.add_edge("ner", "deduplicate")
workflow.add_edge("deduplicate", "wiki_linker")
workflow.add_edge("wiki_linker", "relation_extractor")
workflow.add_edge("relation_extractor", "assembler")
workflow.add_edge("assembler", END)


app = workflow.compile()


if __name__ == "__main__":
    demo_text = """W dniu 14 marca 2024 roku, profesor Adam Szymański, 
    wybitny językoznawca z Uniwersytetu Jagiellońskiego w Krakowie, 
    otrzymał Nagrodę Nobla. W swoim przemówieniu cytował: 
    "The only true wisdom is in knowing you know nothing," (Sokrates). 
    Następnie ogłosił, że przekaże 50% nagrody, czyli około 4.5 mln PLN, 
    fundacji Wielka Orkiestra Świątecznej Pomocy (WOŚP)."""

    print("--- Starting graph processing ---")

    result = app.invoke({"text": demo_text})

    print("\n--- Final JSON Graph ---")
    import json

    print(json.dumps(result["final_graph"], indent=2, ensure_ascii=False))

    try:
        app.get_graph().draw_mermaid_png(output_file_path="state_graph_pipeline.png")
        print("Saved the graph visualization to file: state_graph_pipeline.png")

    except Exception as e:
        print(e)