from typing import Optional, List, Tuple, Annotated, Dict
from langgraph.graph import MessagesState
import operator
from pydantic import BaseModel, Field
from typing import Literal


class Task(BaseModel):
    """Task to perform."""

    agent_name: Literal[
        "validation_agent",
        "web_search_agent",
        "wikipedia_agent",
        "visual_agent",
        "audio_agent",
        "excel_agent",
        "python_agent",
    ] = Field(description="Agent to perform task.")
    agent_task: str = Field(description="Task to perform.")


class Tasks(BaseModel):
    """Tasks to perform."""

    agent_tasks: List[Task] = Field(
        description="tasks to perform, should be in sorted order"
    )


# Define the state type with annotations
class AgentState(MessagesState):
    system_message: str
    last_ai_message: str
    question: str
    final_answer: str
    agent_tasks: Tasks
    past_steps: Annotated[List[Tuple], operator.add]
    error: Optional[str]


class AgentStateInput(MessagesState):
    system_message: str
    question: str


class AgentStateOutput(MessagesState):
    final_answer: str
    error: Optional[str]
