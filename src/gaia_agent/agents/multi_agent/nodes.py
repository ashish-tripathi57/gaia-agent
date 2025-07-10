from typing import Dict, List, Union, Literal
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_core.messages.utils import get_buffer_string
from langchain_core.prompts import ChatPromptTemplate
from gaia_agent.common.tools import wikipedia_search_html, website_scrape, web_search, visual_model, audio_model, run_python, excel_tool
from gaia_agent.common.prompts import load_prompt
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import ToolNode, create_react_agent
from gaia_agent.agents.multi_agent.state import AgentState
import inspect
import importlib.util
from langgraph.types import Command
from types import ModuleType
from pathlib import Path
import logging
logger = logging.getLogger(__name__)


class Tasks(BaseModel):
    """Tasks to perform."""
    agent_tasks: List[Dict[str, str]] = Field(
        description="tasks to perform, should be in sorted order"
    )

class FinalAnswer(BaseModel):
    """Final answer to user."""
    final_answer: str = Field(
        description="Final answer to user."
    )

class Act(BaseModel):
    """Action to perform."""
    action: Union[FinalAnswer, Tasks] = Field(
        description="Action to perform. If you want to respond to user, use Response. "
        "If you need to further use tools to get the answer, use Plan."
    )

# tools = [web_search, wikipedia_search_html, website_scrape, visual_model, audio_model, run_python, excel_tool]

# load tools from file
def load_agent_from_file(file_path: str) -> List[ToolNode]:
    """
    Loads a Python file as a module and yields the name and docstring
    of each function found within it.
    """
    # Create a module spec from the file path
    spec = importlib.util.spec_from_file_location("my_module", file_path)
    
    # Create a module from the spec
    module = importlib.util.module_from_spec(spec)
    
    # Execute the module to make its contents available
    spec.loader.exec_module(module)
    
    # Use inspect to find all function members of the module
    for name, func in inspect.getmembers(module, inspect.isfunction):
        # The __doc__ attribute holds the docstring
        if func.__module__ == module.__name__:
            if "agent_node" in name:
                docstring = inspect.getdoc(func) or "No docstring found."
                yield name, docstring


# Planner node - generates responses
def supervisor(state: AgentState, config: Dict) -> Command[Literal["research_agent_node", "wikipedia_agent_node", "validation_agent"]]:
    logger.info("Supervisor node processing")
    
    parser = JsonOutputParser(pydantic_object=Tasks)

    current_dir = Path(__file__).parent
    agents_path = current_dir / "nodes.py"
    agents_str = "\n---\n".join([f"Agent: {agent_name}\nDescription: {docstring}" for agent_name, docstring in load_agent_from_file(agents_path)])
    agents_str += "\n---\nAgent: validation_agent\nDescription: Once you have determined the final answer to the user's question, pass it to this agent. The answer must be fully self-contained, including all relevant information without relying on prior context.\n---\n"
    tasks_str = "---\n".join(f"{key}\n{val}" for key, val in state["past_steps"])
    tasks_str += "\n---\n"

    llm = ChatGoogleGenerativeAI(
        model="gemma-3-27b-it",
        temperature=0,
        max_tokens=None,
        timeout=60,  # Added a timeout
        max_retries=2,
    )
    
    prompt_path = current_dir / ".." / ".." / "prompts"
    planner_prompt = load_prompt(prompt_path, "supervisor")
    prompt = ChatPromptTemplate.from_messages(
        [
            ("human", planner_prompt + "\n\n{format_instructions}"),
        ]
    )

    planner_chain = prompt | llm | parser
    response = planner_chain.invoke({"question": state["question"], "past_steps": tasks_str, "agents": agents_str, "format_instructions": parser.get_format_instructions()})
    (agent_name, agent_task), = response["agent_tasks"][0].items()
    if agent_name == "validation_agent":
        return Command(goto="validation_agent", update={"agent_tasks":response["agent_tasks"], "last_ai_message": agent_task})
    elif agent_name == "research_agent_node":
        return Command(goto="research_agent_node", update={"agent_tasks":response["agent_tasks"]})
    elif agent_name == "wikipedia_agent_node":
        return Command(goto="wikipedia_agent_node", update={"agent_tasks":response["agent_tasks"]})
    else:
        return Command(goto="__end__", update={"error":f"Invalid agent name {agent_name} found"})
    # return {"agent_tasks": response["agent_tasks"]}


def _execute_agent_task(state: AgentState, config: Dict, agent_name: str, tools: list):
    """Common execution logic for agent tasks."""
    try:
        task = state["agent_tasks"][0][agent_name]
        task_formatted = f"""{task}"""

        logger.info(f"{agent_name.replace('_', ' ').title()} task: {task}")

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite-preview-06-17",
            temperature=0,
            max_tokens=None,
            timeout=60,
            max_retries=2,
        )

        prompt = f"You are a {agent_name} with access to tools.\n\n After you're done with your tasks, respond to the supervisor directly. Respond ONLY with the results of your work, do NOT include ANY other text."
        agent_executor = create_react_agent(llm, tools, prompt=prompt, name=agent_name)
        agent_response = agent_executor.invoke({"messages": [("user", task_formatted)]})
        
        final_response = f"{agent_name}'s task: {task}\n\n{agent_name}'s response: {agent_response['messages'][-1].content}"
        logger.info(f"{agent_name} response: {agent_response['messages'][-1].content}")
        
        return {
            "past_steps": [(f"{agent_name}'s task: {task}", f"{agent_name}'s Response: {agent_response['messages'][-1].content}\n")],
            "messages": [HumanMessage(content=final_response)]
        }
    except Exception as e:
        logger.error(f"Error in {agent_name} node: {e}")
        return {
            "error": f"{agent_name} error: {str(e)}",
            "messages": [AIMessage(content="I encountered an error while generating a response. Please try again.")]
        }


def research_agent_node(state: AgentState, config: Dict):
    """Research agent node has access to the web and can use tools to complete tasks."""
    return _execute_agent_task(state, config, "research_agent_node", [web_search, website_scrape])


def wikipedia_agent_node(state: AgentState, config: Dict):
    """Wikipedia agent node has access to Wikipedia and can use tools to complete tasks."""
    return _execute_agent_task(state, config, "wikipedia_agent_node", [wikipedia_search_html])
