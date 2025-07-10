from langgraph.graph import START, StateGraph, END
from gaia_agent.agents.reflection.state import (
    AgentState,
    AgentStateInput,
    AgentStateOutput,
)
from gaia_agent.agents.reflection.nodes import (
    assistant,
    get_tool_node,
    reflection,
    ready_to_answer,
)
from langgraph.prebuilt import tools_condition
from gaia_agent.common.nodes import validate_answer

# Build the graph
builder = StateGraph(AgentState, input=AgentStateInput, output=AgentStateOutput)

# Define nodes
builder.add_node("assistant", assistant)
builder.add_node("tools", get_tool_node)
builder.add_node("validate_answer", validate_answer)
builder.add_node("reflection", reflection)

# Define edges for the standard flow
builder.add_edge(START, "assistant")

# Conditional edges from assistant
builder.add_conditional_edges(
    "assistant",
    tools_condition,
    {
        "tools": "tools",  # Route to tools if needed
        # END: "END"  # Route to end if no tools needed
        END: "reflection",  # Route to validate_answer if no tools needed
    },
)

# From tools back to assistant to process tool results
builder.add_edge("tools", "assistant")
builder.add_conditional_edges(
    "reflection",
    ready_to_answer,
    {"validate_answer": "validate_answer", "assistant": "assistant"},
)
builder.add_edge("validate_answer", END)

# Set up memory for conversation persistence
# memory = MemorySaver()
graph = builder.compile()
