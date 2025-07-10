from langgraph.graph import START, StateGraph, END
from gaia_agent.agents.multi_agent.state import (
    AgentState,
    AgentStateInput,
    AgentStateOutput,
)
from gaia_agent.agents.multi_agent.nodes import (
    supervisor,
    research_agent,
    wikipedia_agent,
)
from gaia_agent.common.nodes import validate_answer

# Build the graph
builder = StateGraph(AgentState, input=AgentStateInput, output=AgentStateOutput)

# Define nodes
# builder.add_node("react_agent", react_agent)
# builder.add_node("validate_answer", validate_answer)
builder.add_node("supervisor", supervisor)
builder.add_node("research_agent", research_agent)
builder.add_node("wikipedia_agent", wikipedia_agent)
builder.add_node("validation_agent", validate_answer)
# builder.add_node("replanner", replanner)

# Define edges for the standard flow
builder.add_edge(START, "supervisor")
builder.add_edge("research_agent", "supervisor")
builder.add_edge("wikipedia_agent", "supervisor")
builder.add_edge("validation_agent", END)
# builder.add_edge("react_agent", "replanner")
# builder.add_conditional_edges("replanner", should_end, {"__end__": "validate_answer", "react_agent": "react_agent"})
# builder.add_edge("validate_answer", END)

# Set up memory for conversation persistence
# memory = MemorySaver()
graph = builder.compile()
