from langgraph.graph import START, StateGraph, END
from gaia_agent.agents.multi_agent.state import (
    AgentState,
    AgentStateInput,
    AgentStateOutput,
)
from gaia_agent.agents.multi_agent.nodes import (
    supervisor,
    web_search_agent,
    wikipedia_agent,
    visual_agent,
    audio_agent,
    youtube_video_agent,
    excel_agent,
    python_agent,
)
from gaia_agent.common.nodes import validate_answer

# Build the graph
builder = StateGraph(AgentState, input=AgentStateInput, output=AgentStateOutput)

# Define nodes
builder.add_node("supervisor", supervisor)
builder.add_node("web_search_agent", web_search_agent)
builder.add_node("wikipedia_agent", wikipedia_agent)
builder.add_node("visual_agent", visual_agent)
builder.add_node("audio_agent", audio_agent)
builder.add_node("youtube_video_agent", youtube_video_agent)
builder.add_node("excel_agent", excel_agent)
builder.add_node("python_agent", python_agent)
builder.add_node("validation_agent", validate_answer)


# Define edges for the standard flow
builder.add_edge(START, "supervisor")
builder.add_edge("web_search_agent", "supervisor")
builder.add_edge("wikipedia_agent", "supervisor")
builder.add_edge("visual_agent", "supervisor")
builder.add_edge("audio_agent", "supervisor")
builder.add_edge("youtube_video_agent", "supervisor")
builder.add_edge("excel_agent", "supervisor")
builder.add_edge("python_agent", "supervisor")
builder.add_edge("validation_agent", END)

# Set up memory for conversation persistence
# memory = MemorySaver()
graph = builder.compile()
