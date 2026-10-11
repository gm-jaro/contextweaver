from typing import Annotated, TypedDict, Dict
from langchain_core.messages import BaseMessage
from typing import List
import operator


class AgentState(TypedDict):
    input_subgraph: List[Dict]  # The new data to integrate
    graph_context: str  # The current view of the graph
    messages: Annotated[List[BaseMessage], operator.add] # Chat history
    schema_summary: str  # Global list of node/rel types
    iterations: int  # Safety counter
    uuid_map: Dict[str, str]  # Neo4j ShortIDs <-> entityIds map