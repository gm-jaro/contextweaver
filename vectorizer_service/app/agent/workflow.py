from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from app.agent.nodes import retrieve_context_node, curator_agent_node
from app.agent.state import AgentState
from app.agent.tools import tools
from app.config.settings import NUM_OF_AGENT_ITERATIONS


workflow = StateGraph(AgentState)

# Nodes
workflow.add_node("retrieve_context", retrieve_context_node)
workflow.add_node("curator", curator_agent_node)
workflow.add_node("tools", ToolNode(tools))

# Edges
workflow.add_edge(START, "retrieve_context")
workflow.add_edge("retrieve_context", "curator")

# Router
def router(state: AgentState):
    iteration = state.get("iterations", 0)
    last_message = state["messages"][-1]

    if iteration > NUM_OF_AGENT_ITERATIONS:
        return END

    if last_message.tool_calls:
        if last_message.tool_calls[0]["name"] == "do_nothing_tool":
            return END
        return "tools"

    return END


workflow.add_conditional_edges("curator", router, {"tools": "tools", END: END})
workflow.add_edge("tools", "retrieve_context")

app = workflow.compile()
