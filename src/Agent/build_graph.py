from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

# Import from your newly modularized files
from src.Agent.state import AgentState
from src.Agent.agent import chatbot_node, route_tools, tools


def build_graph():
    """Constructs and compiles the LangGraph state machine with short-term memory."""
    workflow = StateGraph(AgentState)

    # Add the core nodes
    workflow.add_node("chatbot", chatbot_node)

    # LangGraph's prebuilt ToolNode safely handles execution and standard errors
    workflow.add_node("tools", ToolNode(tools=tools))

    # Define the ReAct loop edges
    workflow.add_edge(START, "chatbot")
    workflow.add_conditional_edges("chatbot", route_tools)
    workflow.add_edge("tools", "chatbot")

    checkpointer = InMemorySaver()

    # Compile the graph with the checkpointer attached
    return workflow.compile(checkpointer=checkpointer)


# Instantiate the compiled graph so it can be imported by your FastAPI endpoints
trendly_agent = build_graph()