from typing import Optional
from langgraph.graph import MessagesState

# Define the state type with annotations
class AgentState(MessagesState):
    system_message: str
    last_ai_message: str
    question: str
    final_answer: str
    error: Optional[str]

class AgentStateInput(MessagesState):
    system_message: str
    question: str

class AgentStateOutput(MessagesState):
    final_answer: str
    error: Optional[str]