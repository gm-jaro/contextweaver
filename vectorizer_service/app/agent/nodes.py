import json
from loguru import logger
from langchain_openai import ChatOpenAI
from app.agent.state import AgentState
from app.agent.tools import tools
from app.config.settings import LLM_MODEL_NAME, NUM_OF_AGENT_ITERATIONS, LLM_AGENT_NUM_CTX, LLM_AGENT_REASONING, \
    OPENAI_API_KEY
from app.repo import graph_repo
from langchain_core.messages import SystemMessage
from langchain_ollama import ChatOllama


def retrieve_context_node(state: AgentState):
    input_data = state["input_subgraph"]
    schema = state.get("schema_summary", "")

    if not schema:
        schema = graph_repo.get_schema_summary()

    current_uuid_map = {}  # state.get("uuid_map", {}).copy() - save uuid_map between agent sessions

    nodes_to_search = []
    for rel in input_data:
        nodes_to_search.append({"name": rel["head"], "type": rel["head_type"]})
        nodes_to_search.append({"name": rel["tail"], "type": rel["tail_type"]})

    nearest_nodes = graph_repo.find_nearest_nodes(nodes_to_search, top_k=3)

    ids_to_expand = set()
    for node_list in nearest_nodes.values():
        for node in node_list:
            if 'id' in node:
                ids_to_expand.add(node['id'])

    context_lines = graph_repo._expand_subgraph(list(ids_to_expand), hops=1, limit=30, mapping=current_uuid_map)

    if context_lines:
        context_str = "CURRENT GRAPH STATE (NEO4J):\n" + "\n".join(context_lines)
    else:
        context_str = "CURRENT GRAPH STATE: No relevant existing nodes found (Graph might be empty)."

    return {"graph_context": context_str, "schema_summary": schema, "uuid_map": current_uuid_map}


def curator_agent_node(state: AgentState):
    iteration = state.get("iterations", 0)
    input_data = state["input_subgraph"]
    context = state["graph_context"]
    schema = state.get("schema_summary", "Schema not available.")

    if OPENAI_API_KEY and OPENAI_API_KEY.strip():
        model = ChatOpenAI(model=LLM_MODEL_NAME, temperature=0.0, api_key=OPENAI_API_KEY)
        logger.info(f"Using OpenAI model: {LLM_MODEL_NAME}")
    else:
        model = ChatOllama(model=LLM_MODEL_NAME, temperature=0.0, reasoning=LLM_AGENT_REASONING, num_ctx=LLM_AGENT_NUM_CTX,
                           num_predict=-2, keep_alive="30h")
        logger.info(f"Using Ollama model: {LLM_MODEL_NAME}")


    model_with_tools = model.bind_tools(tools)

    system_msg = f"""You are a Knowledge Graph Curator Expert.
    Your goal is to maintain absolute consistency and cleanliness of the Neo4j database while integrating new knowledge.
    
    ID MAPPING SYSTEM (CRITICAL)
    - To save tokens, long Neo4j IDs are mapped to Short IDs: 'n' for nodes (e.g., n1, n2) and 'r' for relationships (e.g., r1).
    - ALWAYS use these Short IDs in your tool calls. The system will translate them back to database IDs automatically.
    - If you need to know more about a node or its origin, use its Short ID.

    LANGUAGE & FORMATTING REQUIREMENT:
    - All node types (head_type, tail_type), and relationship names MUST be in POLISH.
    - If you encounter English terms in the Input relationship or node type, translate them to Polish before adding or updating.
    - RELATIONSHIPS MUST BE UPPERCASE: Use "MIESZKA_W" instead of "mieszka w".
    - Use underscores (_) instead of spaces for relationship names.
    - Consistency in Polish terminology is critical (e.g., use "Osoba" instead of "Person").

    ONTOLOGY & SCHEMA COMPLIANCE (Guideline):
    You have access to the "GLOBAL SCHEMA" below.
    1. PRIORITIZE EXISTING TYPES: Always check the Global Schema first. If a suitable Node Type or Relationship Type exists there, USE IT.
       - Example: If the schema contains "Osoba", do NOT create a new type "Ludzie" or "Person".
    2. NEW TYPES ALLOWED: If the Input introduces a completely new concept that cannot be mapped to existing types in the Schema, you ARE ALLOWED to create a new type.
    3. NAMING CONVENTION: Use Polish language.

    ERROR HANDLING & RECOVERY:
    If a tool execution fails (returns an error message), you must self-correct:
    - "Node not found": The ID might be wrong or the node was merged/deleted. Use `search_neighborhood_tool` to find the correct entity ID again.
    - "Merge failed": Check if you are trying to merge nodes that are already merged. Refresh context.
    - "Recursion Limit": If you feel stuck in a loop, STOP. If the graph is mostly correct, use `do_nothing_tool`.
    - Infinite Adding: If you see the same triple in 'Input' and 'Current Graph', DO NOT add it again.

    YOUR TASK:
    You have a list of new facts (INPUT SUBGRAPH) and the current state of the database (CURRENT GRAPH CONTEXT).
    You must integrate the Input into the Context, resolving conflicts and duplicates. 
    Try to do all the work in shortest number of iterations possible - preferably 8-12.

    MANDATORY OPERATING PROTOCOL (Step-by-Step):

    PHASE 1: DIAGNOSIS AND CLEANING (Highest Priority)
    Before adding anything new, fix what you already see in the Context.
    1. NODE DUPLICATES:
       - Do you see "Władimir Putin" (ID: n123) and "Putin" (ID: n456) in the context?
       - If it's the same entity -> USE `merge_nodes_tool([n123, n456])`.
       - IMPORTANT: Do not merge nodes of different types (e.g., City and Person) unless it's an obvious error.

    2. RELATIONSHIP DUPLICATES:
       - Do you see multiple arrows of the similar or same type between the same nodes?
       - USE `deduplicate_relations_tool`.

    3. INCORRECT CONNECTIONS:
       - Does a relationship point to the wrong node (e.g., "Stolica" -> "Rosja" instead of "Moskwa")?
       - USE `redirect_relation_tool`.

    PHASE 2: INTEGRATION (Only when Context is clean)
    1. If the Input contains facts not present in the Context -> USE `add_triplet_tool`.
    2. If the Input confirms existing relationships -> DO NOTHING (avoid redundancy).

    PHASE 3: METADATA UPDATES
    - If a node is named "Jan" and Input says "Jan Kowalski" -> USE `update_node_properties_tool`.

    CRITICAL RULES:
    - BATCH OPERATIONS: If you identify multiple independent issues (e.g., 5 pairs of duplicates), generate ALL necessary tool calls in a SINGLE message. Do not wait.
    - DO NOT GUESS IDs. Use only IDs visible in the CURRENT GRAPH CONTEXT section.
    - FINISH FAST: If the INPUT SUBGRAPH is already reflected in the CURRENT GRAPH CONTEXT, you MUST use do_nothing_tool immediately.
    - DO NOT UNDO YOUR WORK: Do not delete or redirect relations you have just created in previous steps unless you explicitly identified a mistake in your reasoning.
    - AVOID SEARCH LOOPS: If search_neighborhood_tool returns the same information twice, stop searching and decide based on what you have.
    - If the Context is empty, simply add the triplets from the Input.
    - If the work is done and the Input is fully reflected in the graph -> USE `do_nothing_tool`.

    --- INPUT DATA ---
    INPUT SUBGRAPH (New knowledge):
    {json.dumps(input_data, indent=2, ensure_ascii=False)}

    --- GLOBAL SCHEMA (Available Types) ---
    {schema}

    --- CURRENT DATABASE STATE ---
    CURRENT GRAPH CONTEXT (As seen in the DB):
    {context}

    Iteration: {iteration}/{NUM_OF_AGENT_ITERATIONS}"""

    response = model_with_tools.invoke([SystemMessage(content=system_msg)] + state["messages"])

    return {
        "messages": [response],
        "iterations": iteration + 1
    }
