from typing import List, Annotated
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from app.repo import graph_repo, retriever
from langgraph.prebuilt import InjectedState


def resolve_id(short_id: str, uuid_map: dict) -> str:
    return uuid_map.get(short_id, short_id)


@tool
def read_graph_schema_tool() -> str:
    """
    [Action: read_graph_schema]
    Retrieves a summary of the graph schema, including unique node types (labels) and relationship types.
    Use this to understand the ontology and vocabulary of the graph.

    Returns:
        str: A text-formatted list of unique Node and Relationship types present in Neo4j knowledge graph.
    """
    return graph_repo.get_schema_summary()

@tool
def search_neighborhood_tool(query: str, uuid_map: Annotated[dict, InjectedState("uuid_map")]) -> str:
    """
    [Action: search_neighborhood]
    Searches for nodes matching the query (using vector/keyword search) and returns their immediate neighborhood (1-hop).
    Search for a SINGLE entity name to see its local graph. Do not provide full sentences.
    Use this to visualize the graph context around a specific concept or entity before making changes.

    Args:
        query (str): The name or concept (e.g. Node name) to search for (e.g. "Vladimir Putin").

    Returns:
        str: Text representation of the local graph (Nodes and Relations with Short IDs).
    """
    return graph_repo.search_neighborhood(query, mapping=uuid_map, top_k=3, hops=1)

@tool
def inspect_node_details_tool(node_id: str, uuid_map: Annotated[dict, InjectedState("uuid_map")]) -> str:
    """
    [Action: inspect_node_details]
    Returns all properties of a specific node given its elementId.
    Use this to verify details (e.g. dates, URLs) before merging or deleting.

    Args:
        node_id (str): The Short ID of the node (e.g. 'n1').

    Returns:
        str: Json-formatted string of node properties.
    """
    real_id = resolve_id(node_id, uuid_map)
    return graph_repo.get_node_details(real_id)

@tool
def read_source_chunk_tool(node_id: str, uuid_map: Annotated[dict, InjectedState("uuid_map")]) -> str:
    """
    [Action: read_source_chunk]
    Retrieves the original text fragments (chunks) that were used to create a specific node.
    CRITICAL: Use this when you need more context about a node or to verify if a fact was correctly extracted.

    Args:
        node_id (str): The Short ID of the node (e.g. 'n1').

    Returns:
        str: The raw text content of the source chunks.
    """
    real_node_id = resolve_id(node_id, uuid_map)

    chunk_ids = graph_repo.get_node_chunk_ids(real_node_id)
    if not chunk_ids:
        return f"Node {node_id} has no source chunk information."

    source_text = retriever.get_chunks_text_by_ids(chunk_ids)
    return source_text

@tool
def merge_nodes_tool(node_ids: List[str], uuid_map: Annotated[dict, InjectedState("uuid_map")]) -> str:
    """
    [Action: merge_nodes]
    Merges multiple nodes into a single node. Relationships are preserved and moved to the primary node.
    Use this for deduplication (e.g. merging "USA" and "United States").

    Args:
        node_ids (List[str]): A list of Short IDs to merge (e.g. ['n1', 'n5']).
                              The first ID ('n1') becomes the primary node; 'n5' is merged into it.

    Returns:
        str: Success message or error.
    """
    real_ids = [resolve_id(nid, uuid_map) for nid in node_ids]
    return graph_repo.merge_nodes(real_ids)

@tool
def deduplicate_relations_tool(relation_ids: List[str], uuid_map: Annotated[dict, InjectedState("uuid_map")]) -> str:
    """
    [Action: deduplicate_relations]
    Merges multiple relationships (specified by their IDs) into a single relationship.
    Use this when you see similar edges between the same two nodes.

    Args:
        relation_ids (List[str]): A list of Short IDs of relationships to merge (e.g. ['r1', 'r2']).

    Returns:
        str: Confirmation message.
    """
    real_ids = [resolve_id(rid, uuid_map) for rid in relation_ids]
    return graph_repo.deduplicate_relations_by_ids(real_ids)

@tool
def add_triplet_tool(head: str, head_type: str, relation: str, tail: str, tail_type: str, config: RunnableConfig) -> str:
    """
    [Action: add_triplet]
    Adds a new fact (triplet) to the graph.
    If nodes do not exist, they are created. If they exist, the relationship is added/merged.
    Use this to insert new knowledge from the input chunk.

    Args:
        head (str): Name of the start node in Polish (e.g. "Warszawa").
        head_type (str): Type of the start node in Polish (e.g. "Miasto").
        relation (str): Name of the relationship in Polish (e.g. "STOLICA"). Use UPPERCASE and underscores.
        tail (str): Name of the end node in Polish (e.g. "Polska").
        tail_type (str): Type of the end node in Polish (e.g. "Kraj").

    Returns:
        str: Confirmation message.
    """
    chunk_id = config.get("configurable", {}).get("chunk_id")
    return graph_repo.upsert_triplet(head, head_type, relation, tail, tail_type, chunk_id=chunk_id)

@tool
def update_node_properties_tool(node_id: str, uuid_map: Annotated[dict, InjectedState("uuid_map")],
                                new_name: str = None, new_type: str = None) -> str:
    """
    [Action: update_node_properties]
    Updates a node's name or type.
    Use this to correct typos or normalize names (e.g. changing "Jan" to "Jan Kowalski").

    Args:
        node_id (str): The Short ID of the node to update (e.g. 'n1').
        new_name (str, optional): The new name to set. Defaults to None.
        new_type (str, optional): The new type/label to set. Defaults to None.

    Returns:
        str: Success message showing the new state of the node.
    """
    real_id = resolve_id(node_id, uuid_map)
    return graph_repo.update_node_properties(real_id, new_name, new_type)

@tool
def split_node_and_move_relations_tool(aggregated_node_id: str, new_name: str, new_type: str,
                                       relation_ids_to_move: List[str],
                                       uuid_map: Annotated[dict, InjectedState("uuid_map")]) -> str:
    """
    [Action: split_node_and_move_relations]
    Splits a 'super-node' into two. Creates a NEW node and moves specific relations to it.

    Args:
        aggregated_node_id (str): The Short ID of the existing node to split (e.g. 'n1').
        new_name (str): Name for the NEW node being created.
        new_type (str): Type for the NEW node.
        relation_ids_to_move (List[str]): List of Short IDs of relationships (e.g. ['r1', 'r3'])
                                          that should be moved from 'n1' to the new node.

    Returns:
        str: Summary of the split operation.
    """
    real_node_id = resolve_id(aggregated_node_id, uuid_map)
    real_rel_ids = [resolve_id(rid, uuid_map) for rid in relation_ids_to_move]
    return graph_repo.split_node(real_node_id, new_name, new_type, real_rel_ids)

@tool
def redirect_relation_tool(relation_id: str, new_target_node_id: str,
                           uuid_map: Annotated[dict, InjectedState("uuid_map")]) -> str:
    """
    [Action: redirect_relation]
    Redirects an existing relationship to point to a different target node.
    Use this when a relationship points to the wrong entity (e.g. 'Capital' pointing to 'Russia' instead of 'Moscow').

    Args:
        relation_id (str): The Short ID of the relationship to move (e.g. 'r1').
        new_target_node_id (str): The Short ID of the correct target node (e.g. 'n5').

    Returns:
        str: Confirmation of the redirection.
    """
    real_rel_id = resolve_id(relation_id, uuid_map)
    real_node_id = resolve_id(new_target_node_id, uuid_map)
    return graph_repo.redirect_relation(real_rel_id, real_node_id)

@tool
def delete_element_tool(element_id: str, uuid_map: Annotated[dict, InjectedState("uuid_map")],
                        is_node: bool = True) -> str:
    """
    [Action: delete_element]
    Deletes a node (and its relationships) OR a single relationship.

    Args:
        element_id (str): The Short ID to delete (e.g. 'n1' or 'r1').
        is_node (bool): Set True if deleting a Node ('n...'). Set False if deleting a Relationship ('r...').

    Returns:
        str: Confirmation message.
    """
    real_id = resolve_id(element_id, uuid_map)
    return graph_repo.delete_element(real_id, is_node)

@tool
def do_nothing_tool() -> str:
    """
    [Action: do_nothing]
    Signals that the graph is consistent with the input data and no further actions are needed.
    Call this to finish the process.

    Returns:
        str: Final status message.
    """
    return "Process finished."


tools = [
    read_graph_schema_tool,
    search_neighborhood_tool,
    inspect_node_details_tool,
    read_source_chunk_tool,
    merge_nodes_tool,
    deduplicate_relations_tool,
    add_triplet_tool,
    update_node_properties_tool,
    split_node_and_move_relations_tool,
    redirect_relation_tool,
    delete_element_tool,
    do_nothing_tool
]
