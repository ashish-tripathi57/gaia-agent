from langgraph.graph import START, StateGraph, END
from gaia_agent.agents.plan_execute.state import (
    AgentState,
    AgentStateInput,
    AgentStateOutput,
)
from gaia_agent.agents.plan_execute.nodes import (
    react_agent,
    planner,
    replanner,
    should_end,
)
from gaia_agent.common.nodes import validate_answer

# Build the graph
builder = StateGraph(AgentState, input=AgentStateInput, output=AgentStateOutput)

# Define nodes
builder.add_node("react_agent", react_agent)
builder.add_node("validate_answer", validate_answer)
builder.add_node("planner", planner)
builder.add_node("replanner", replanner)

# Define edges for the standard flow
builder.add_edge(START, "planner")
builder.add_edge("planner", "react_agent")
builder.add_edge("react_agent", "replanner")
builder.add_conditional_edges(
    "replanner",
    should_end,
    {"__end__": "validate_answer", "react_agent": "react_agent"},
)
builder.add_edge("validate_answer", END)

# Set up memory for conversation persistence
# memory = MemorySaver()
graph = builder.compile()
